Let me read every relevant section carefully before explaining.Now I have read everything. Here is the complete explanation.

---

## What greedy seeding is and why it matters

### The problem with random starts

SA works by starting from some binary vector and making small changes (one bit flip at a time), accepting changes that lower energy and occasionally accepting ones that raise it (to escape local minima). The critical question is: where does that starting vector come from?

In the base code, every single read started from a completely random binary vector — every variable independently set to 0 or 1 with equal probability. With 39 variables in your PPLN-reduced instance (or 384 in the full instance), a random vector has no logical structure. Consider what a random start looks like in terms of the actual problem:

- Some RDCs might be assigned to 3 CDCs simultaneously
- Some RDCs might be assigned to none
- CDC `z_j` might be 0 while plants are sending supply to it
- Slack variables might be set to random values that contradict the supply/demand relationship

This is a deeply infeasible state. SA must first climb out of all those violations before it can even begin optimising the objective. With `num_sweeps=5000` single-bit flips, most of the budget gets consumed just trying to find any feasible point. Often it never does within the sweep budget, which is why the base code rarely found feasible solutions for larger instances.

---

### What the greedy seed does

`build_greedy_seed` constructs one specific binary vector using logical reasoning about the problem, not random chance. It runs three steps in order:

**Step 1 — Assign every RDC to exactly one CDC**

```python
for r in data.R:
    best_j = min(data.T_r[r], key=lambda j: data.b_jr.get((j, r), 1e9))
    sample[ymap[(best_j, r)]] = 1
```

For each regional distribution center `r`, it looks at all CDCs that are allowed to serve it (`data.T_r[r]`) and picks the one with the lowest transport cost `b_jr`. It sets that `y_{j,r}` variable to 1 and leaves all others at 0.

After this step, **constraint (13) is fully satisfied** — every RDC is assigned to exactly one CDC. `v1 = 0` in the seed.

**Step 2 — Activate every CDC that received an assignment**

```python
active_cdcs = set()
for (j, r), nm in ymap.items():
    if sample[nm] == 1:
        active_cdcs.add(j)
for j in active_cdcs:
    sample[zmap[j]] = 1
```

It scans through all `y_{j,r}` variables just set, collects the CDCs that appear, and sets `z_j = 1` for each of them. CDCs with no RDC assignments remain at `z_j = 0`.

This directly supports constraint (15): `sum(p_i * x_ij) <= q_j * z_j`. If a CDC is activated (`z_j=1`), the right-hand side is non-zero, which allows supply to flow through it legally.

**Step 3 — Assign the cheapest plant to each active CDC**

```python
for j in active_cdcs:
    best_i = min(data.N_c[j], key=lambda i: data.a_ij.get((i, j), 1e9))
    sample[xmap[(best_i, j)]] = 1
```

For each active CDC, it picks the plant with the lowest transport cost `a_ij` and sets `x_{i,j} = 1`.

**Slack variables — all set to 0**

```python
sample = {var: 0 for var in all_qubo_vars}
```

The seed initialises every variable in the entire QUBO to 0 first, then overrides the decision variables. This means all slack variables (`s1_j_0`, `s1_j_1`, ..., `s2_j_0`, `s2_j_1`, ...) start at 0. A slack value of 0 means "no slack needed" — the assumption is that the supply/demand balance is tight. SA will flip these bits during annealing to satisfy constraints (14) and (15) exactly.

---

### Why the seed must cover ALL variables, not just decision variables

This is the bug that caused the `ValueError: mismatch between variables in 'initial_states' and 'bqm'`.

Neal's `initial_states` parameter requires the `SampleSet` to contain **exactly** the same variable set as the BQM — every variable that appears in any QUBO entry must have a value in the initial state. The original broken version only set the 21 decision variables and omitted the 18 slack variables. Neal counted 39 variables in the BQM and 21 in the initial state and refused to proceed.

The fix was to first collect every variable that appears anywhere in Q:

```python
all_qubo_vars = set()
for (u, v) in Q:
    all_qubo_vars.add(u)
    all_qubo_vars.add(v)
sample = {var: 0 for var in all_qubo_vars}   # all 39 variables initialised
```

Then the greedy steps override only the decision variable subset. Slack variables remain at 0. Now the `SampleSet` contains all 39 variables and neal accepts it without complaint.

This is also why `Q` is passed as a parameter to `build_greedy_seed` — the function needs to know the complete variable set of the QUBO, which it cannot determine from `xmap`, `ymap`, `zmap` alone since those only cover decision variables.

---

### How it is wired into both samplers

**For neal:**

```python
initial_ss = dimod.SampleSet.from_samples(
    greedy_seed,          # dict with all 39 variables
    vartype=dimod.BINARY,
    energy=0.0
)
sampleset = sampler.sample_qubo(
    Q,
    num_reads      = num_reads,
    num_sweeps     = num_sweeps,
    initial_states = initial_ss,  # seeds one read
)
```

Neal receives one sample in `initial_states`. It uses that as the starting point for one of its reads. The remaining `num_reads - 1` reads start randomly as usual. This means one read in every APL iteration begins from a logically structured, near-feasible point.

**For NumpySASampler (fallback):**

```python
sample_list = sampler.sample_qubo(
    Q,
    num_reads     = num_reads,
    num_sweeps    = num_sweeps,
    initial_state = greedy_seed,   # used only for read index 0
    seed          = seed + t,
)
```

Inside `NumpySASampler.sample_qubo`:

```python
for read_idx in range(num_reads):
    if read_idx == 0 and initial_state is not None:
        x = np.array([float(initial_state.get(var_list[i], 0)) for i in range(n)])
    else:
        x = rng.integers(0, 2, size=n).astype(np.float64)
```

Only read index 0 uses the seed. All others are random. The behaviour is identical to the neal path.

---

### Why the seed is rebuilt every APL iteration

```python
for t in range(1, num_iters + 1):
    Q, xmap, ymap, zmap = build_qubo_with_slacks(data, P1, P2, P3)
    greedy_seed = build_greedy_seed(data, xmap, ymap, zmap, Q)
```

The QUBO is rebuilt at the start of each iteration because the penalty values P1, P2, P3 change between iterations. When penalties change, the diagonal and off-diagonal entries of Q change, but the **variable names** (`x_0_8`, `y_8_3`, `s1_8_0`, etc.) are regenerated fresh from `xmap`, `ymap`, `zmap`. The greedy seed must be rebuilt using these freshly generated name maps so the variable names in the seed exactly match the variable names in the new Q. If the seed were built once and reused, the variable names would still match (they are deterministic from the instance data), but rebuilding makes the dependency explicit and correct.