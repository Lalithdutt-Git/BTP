#!/usr/bin/env python3
"""
tlflp_solver.py
===============
Improved TLFLP QUBO Solver with:
  1. PPLN  – Preprocessing Procedure of Logistic Network (Ciacco et al., 2026)
  2. Proper QUBO with binary-encoded slack variables (constraints 14 & 15 as equalities)
  3. Adaptive Penalty Learning (APL) with scientifically motivated min/max bounds
  4. Population-based Simulated Annealing with elite archive + restart strategy
  5. Greedy feasibility seeding to warm-start SA
  6. Strict feasibility: only solutions with zero violations are accepted
  7. Full reporting identical in style to original test2_updated.py

Usage:
    python tlflp_solver.py instance_5I_10C_15R.txt [options]

Options:
    --reads     INT   SA reads per iteration         (default 8000)
    --iters     INT   APL outer iterations           (default 40)
    --sweeps    INT   SA sweeps per read             (default 5000)
    --use_ppln        Apply PPLN preprocessing       (default True)
    --no_ppln         Skip PPLN preprocessing
    --pop_size  INT   SA population size             (default 5)
    --seed      INT   Random seed                    (default 42)
    --time_limit FLOAT  Wall-clock time limit (s)   (default 300)
"""

import argparse
import itertools
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Try to import neal; fall back to a pure-NumPy SA if unavailable
# ──────────────────────────────────────────────────────────────────────────────
try:
    import neal
    import dimod
    HAS_NEAL = True
except ImportError:
    HAS_NEAL = False
    print("[INFO] neal/dimod not found – using built-in NumPy SA sampler.")

# ──────────────────────────────────────────────────────────────────────────────
# Parser — imported from parser.py (place parser.py in the same directory)
# ──────────────────────────────────────────────────────────────────────────────
from parser import InstanceData, InstanceParser


# ──────────────────────────────────────────────────────────────────────────────
# PPLN – Preprocessing Procedure of Logistic Network
# ──────────────────────────────────────────────────────────────────────────────

def compute_ppln_bounds(data: InstanceData) -> Tuple[int, int]:
    """
    Compute beta (min CDCs needed) and gamma (max CDCs needed).

    Formulae from paper (equations 6 & 7):
        beta  = ceil( |R| / floor( max(sum_pi, max_qj) / min_dr ) )
        gamma = ceil( |R| / floor( min(sum_pi, min_qj) / min_dr ) )
    """
    R_cnt  = len(data.R)
    sum_pi = sum(data.p_i.values())
    max_qj = max(data.q_j.values())
    min_qj = min(data.q_j.values())
    min_dr = min(data.d_r.values())

    denom_beta  = math.floor(max(sum_pi, max_qj) / min_dr)
    denom_gamma = math.floor(min(sum_pi, min_qj) / min_dr)

    beta  = math.ceil(R_cnt / denom_beta)  if denom_beta  > 0 else R_cnt
    gamma = math.ceil(R_cnt / denom_gamma) if denom_gamma > 0 else R_cnt

    # Safety: gamma can't be less than beta, or more than |C|
    beta  = max(1, beta)
    gamma = max(beta, min(gamma, len(data.C)))
    return beta, gamma


def combo_covers_all_rdc(combo: Tuple[int,...], data: InstanceData) -> bool:
    """Check that every RDC r is covered by at least one CDC in combo."""
    covered = set()
    for j in combo:
        for r in data.R:
            if j in data.T_r[r]:
                covered.add(r)
    return len(covered) == len(data.R)


def combo_cost(combo: Tuple[int,...], data: InstanceData) -> float:
    """Heuristic cost used by PPLN to rank combinations (eq. 8)."""
    c_sum = sum(data.c_j[j] for j in combo)
    a_sum = sum(data.a_ij.get((i,j), 0) for i in data.I for j in combo)
    b_sum = sum(data.b_jr.get((j,r), 0) for r in data.R for j in combo)
    return c_sum + a_sum + b_sum


def check_capacity_constraints(combo: Tuple[int,...], data: InstanceData) -> bool:
    """
    Verify that the combination satisfies capacity constraints.
    Constraint 3: aggregate demand <= total supply reaching each CDC subset.
    Constraint 4: total supply <= storage capacity of each CDC (if activated).

    We use the relaxed check from PPLN:
      - Total demand <= total supply capacity (sum of p_i for plants that can serve combo)
      - Each CDC q_j >= portion of demand it could serve
    """
    # Total available supply from all plants that can reach at least one CDC in combo
    supply_plants = set()
    for j in combo:
        for i in data.N_c[j]:
            supply_plants.add(i)
    total_supply = sum(data.p_i[i] for i in supply_plants)
    total_demand = sum(data.d_r[r] for r in data.R)

    # Constraint 3 aggregate check
    if total_demand > total_supply:
        return False

    # Constraint 4: supply sent to a CDC cannot exceed its capacity
    # We check: max total plant supply that could go to any CDC in combo <= q_j
    # (permissive: at least one CDC can hold all supply going through it)
    max_cap = max(data.q_j[j] for j in combo)
    if total_supply > max_cap:
        # Try progressive reduction: remove lowest-capacity plants until feasible
        sorted_plants = sorted(supply_plants, key=lambda i: data.p_i[i])
        remaining = list(sorted_plants)
        rem_supply = total_supply
        while rem_supply > max_cap and remaining:
            rem_supply -= data.p_i[remaining.pop(0)]
        if rem_supply < total_demand:
            return False

    return True


def run_ppln(data: InstanceData, verbose: bool = True) -> InstanceData:
    """
    Apply PPLN to reduce the set C.
    Returns a new InstanceData with reduced C, N_c, T_r, a_ij, b_jr, c_j, q_j.
    If PPLN cannot reduce (no valid combo found), returns original data.
    """
    beta, gamma = compute_ppln_bounds(data)
    if verbose:
        print(f"\n[PPLN] beta={beta}, gamma={gamma}  |C|={len(data.C)}")

    best_combo = None
    best_cost  = math.inf

    for c_size in range(beta, gamma + 1):
        combos = list(itertools.combinations(data.C, c_size))
        if verbose:
            print(f"[PPLN] Testing {len(combos)} combinations of size {c_size} ...")

        # Filter to those that cover all RDCs
        covering = [(combo, combo_cost(combo, data))
                    for combo in combos
                    if combo_covers_all_rdc(combo, data)]

        # Sort by heuristic cost ascending
        covering.sort(key=lambda x: x[1])

        for combo, cost in covering:
            if check_capacity_constraints(combo, data):
                if cost < best_cost:
                    best_combo = combo
                    best_cost  = cost
                break  # First valid in sorted order wins at this size

        if best_combo is not None:
            if verbose:
                print(f"[PPLN] Selected CDCs: {list(best_combo)}  heuristic_cost={best_cost:.1f}")
            break

    if best_combo is None:
        if verbose:
            print("[PPLN] No valid reduction found – using original C.")
        return data

    # Build reduced InstanceData
    reduced = InstanceData()
    reduced.I   = data.I[:]
    reduced.C   = list(best_combo)
    reduced.R   = data.R[:]
    reduced.p_i = dict(data.p_i)
    reduced.d_r = dict(data.d_r)

    reduced.q_j = {j: data.q_j[j] for j in reduced.C}
    reduced.c_j = {j: data.c_j[j] for j in reduced.C}

    reduced.a_ij = {(i,j): data.a_ij[(i,j)]
                    for i in reduced.I for j in reduced.C
                    if (i,j) in data.a_ij}
    reduced.b_jr = {(j,r): data.b_jr[(j,r)]
                    for j in reduced.C for r in reduced.R
                    if (j,r) in data.b_jr}

    reduced.N_c = {j: [i for i in data.N_c[j] if i in reduced.I]
                   for j in reduced.C}
    reduced.T_r = {r: [j for j in data.T_r[r] if j in reduced.C]
                   for r in reduced.R}

    # Ensure every RDC still has at least one CDC option
    for r in reduced.R:
        if not reduced.T_r[r]:
            if verbose:
                print(f"[PPLN] RDC {r} lost all CDC options – reverting to original.")
            return data

    if verbose:
        orig_vars = (len(data.I)*len(data.C) +
                     len(data.C)*len(data.R) + len(data.C))
        new_vars  = (len(reduced.I)*len(reduced.C) +
                     len(reduced.C)*len(reduced.R) + len(reduced.C))
        print(f"[PPLN] Variables reduced: {orig_vars} → {new_vars} "
              f"({100*(1-new_vars/orig_vars):.1f}% reduction)")

    return reduced


# ──────────────────────────────────────────────────────────────────────────────
# Objective & Violation helpers
# ──────────────────────────────────────────────────────────────────────────────

def original_objective(data: InstanceData, x, y, z) -> float:
    obj = 0.0
    for (i,j), v in x.items():
        if v: obj += data.a_ij[(i,j)] * v
    for (j,r), v in y.items():
        if v: obj += data.b_jr[(j,r)] * v
    for j, v in z.items():
        if v: obj += data.c_j[j] * v
    return obj


def count_violations(data: InstanceData, x, y, z
                     ) -> Tuple[float, float, float]:
    """
    v1 : each RDC served exactly once  (sum |served-1|)
    v2 : demand <= supply at CDC       (excess demand)
    v3 : supply <= capacity * z_j      (excess supply)
    """
    v1 = v2 = v3 = 0.0

    for r in data.R:
        served = sum(y.get((j,r), 0) for j in data.T_r[r])
        v1 += abs(served - 1)

    for j in data.C:
        demand = sum(data.d_r[r] * y.get((j,r), 0)
                     for r in data.R if (j,r) in data.b_jr)
        supply = sum(data.p_i[i] * x.get((i,j), 0)
                     for i in data.N_c[j])
        if demand > supply:
            v2 += demand - supply

    for j in data.C:
        supply = sum(data.p_i[i] * x.get((i,j), 0)
                     for i in data.N_c[j])
        cap    = data.q_j[j] * z.get(j, 0)
        if supply > cap:
            v3 += supply - cap

    return v1, v2, v3


# ──────────────────────────────────────────────────────────────────────────────
# QUBO builder with proper binary-encoded slack variables
# ──────────────────────────────────────────────────────────────────────────────

def _bits_needed(upper_bound: float) -> int:
    """Minimum bits to represent integers 0 .. ceil(upper_bound)."""
    ub = math.ceil(upper_bound)
    if ub <= 0:
        return 1
    return max(1, math.ceil(math.log2(ub + 1)))


def build_qubo_with_slacks(data: InstanceData,
                           P1: float, P2: float, P3: float
                           ) -> Tuple[dict, dict, dict, dict]:
    """
    Build QUBO with binary-encoded slack variables for constraints (14) and (15).

    Decision variables:
        x_{i}_{j}     – PF i serves CDC j
        y_{j}_{r}     – CDC j serves RDC r
        z_{j}         – CDC j is activated
    Slack variables (binary-encoded):
        s1_{j}_{l}    – slack for demand-supply constraint at CDC j
        s2_{j}_{l}    – slack for supply-capacity constraint at CDC j

    Returns: Q (dict), xmap, ymap, zmap
    """
    Q = {}

    def add(u, v, val):
        if val == 0:
            return
        key = (u, v) if u <= v else (v, u)
        Q[key] = Q.get(key, 0.0) + val

    # Variable name maps
    xmap = {(i,j): f"x_{i}_{j}" for j in data.C for i in data.N_c[j]}
    ymap = {(j,r): f"y_{j}_{r}" for r in data.R for j in data.T_r[r]}
    zmap = {j: f"z_{j}" for j in data.C}

    # ── Objective terms ──────────────────────────────────────────────────────
    for (i,j), nm in xmap.items():
        add(nm, nm, data.a_ij[(i,j)])
    for (j,r), nm in ymap.items():
        add(nm, nm, data.b_jr[(j,r)])
    for j, nm in zmap.items():
        add(nm, nm, data.c_j[j])

    # ── Penalty (13): each RDC served exactly once ───────────────────────────
    # P1 * (sum_j y_{j,r} - 1)^2  for each r
    for r in data.R:
        vars_r = [ymap[(j,r)] for j in data.T_r[r]]
        # expand (sum - 1)^2 = sum^2 - 2*sum + 1
        for u in vars_r:
            add(u, u, P1 * (1 - 2))          # diagonal: P1*(1) - 2*P1
        for u, v in itertools.combinations(vars_r, 2):
            add(u, v, 2 * P1)                 # off-diagonal cross terms
        # constant +P1 per r is dropped (doesn't affect optimum)

    # ── Penalties (14) & (15): with slack variables ──────────────────────────
    for j in data.C:
        # Constraint (14): sum_r d_r*y_{j,r} - sum_i p_i*x_{i,j} + S1_j = 0
        # Upper bound for S1_j: max possible demand served by j
        U1_j = sum(data.d_r[r] for r in data.R if (j,r) in ymap)
        k1   = _bits_needed(U1_j)
        s1_vars = [f"s1_{j}_{l}" for l in range(k1)]
        s1_wts  = [2**l for l in range(k1)]

        # Constraint (15): sum_i p_i*x_{i,j} - q_j*z_j + S2_j = 0
        # Upper bound for S2_j: max supply from plants
        U2_j = sum(data.p_i[i] for i in data.N_c[j])
        k2   = _bits_needed(U2_j)
        s2_vars = [f"s2_{j}_{l}" for l in range(k2)]
        s2_wts  = [2**l for l in range(k2)]

        # Build term lists for (14): d_r*y - p_i*x + s1
        terms14 = []
        for r in data.R:
            if (j,r) in ymap:
                terms14.append((ymap[(j,r)], +data.d_r[r]))
        for i in data.N_c[j]:
            terms14.append((xmap[(i,j)], -data.p_i[i]))
        for nm, w in zip(s1_vars, s1_wts):
            terms14.append((nm, +w))

        # P2 * (sum of terms14)^2
        for (v1, w1), (v2, w2) in itertools.product(terms14, repeat=2):
            add(v1, v2, P2 * w1 * w2)

        # Build term list for (15): p_i*x - q_j*z + s2
        terms15 = []
        for i in data.N_c[j]:
            terms15.append((xmap[(i,j)], +data.p_i[i]))
        terms15.append((zmap[j], -data.q_j[j]))
        for nm, w in zip(s2_vars, s2_wts):
            terms15.append((nm, +w))

        # P3 * (sum of terms15)^2
        for (v1, w1), (v2, w2) in itertools.product(terms15, repeat=2):
            add(v1, v2, P3 * w1 * w2)

    return Q, xmap, ymap, zmap


# ──────────────────────────────────────────────────────────────────────────────
# Pure NumPy Simulated Annealing sampler (fallback when neal unavailable)
# ──────────────────────────────────────────────────────────────────────────────

class NumpySASampler:
    """
    Fast vectorised SA for QUBO problems.
    Supports multiple independent reads (parallel chains).
    """

    def sample_qubo(self,
                    Q: dict,
                    num_reads: int = 1000,
                    num_sweeps: int = 3000,
                    beta_range: Tuple[float,float] = (0.1, 10.0),
                    seed: int = None
                    ) -> List[dict]:
        """
        Returns list of (sample_dict, energy) tuples, best-first.
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng()

        # Build variable index
        vars_set = set()
        for (u, v) in Q:
            vars_set.add(u)
            vars_set.add(v)
        var_list = sorted(vars_set)
        var_idx  = {v: i for i, v in enumerate(var_list)}
        n        = len(var_list)

        if n == 0:
            return [({}, 0.0)]

        # Build dense Q matrix
        Qmat = np.zeros((n, n), dtype=np.float64)
        for (u, v), val in Q.items():
            i, j = var_idx[u], var_idx[v]
            if i == j:
                Qmat[i, i] += val
            else:
                # Store in upper triangle; eval uses x^T Q x
                Qmat[i, j] += val

        # Precompute h (linear) and J (upper-triangle interactions)
        h = np.diag(Qmat).copy()
        J = Qmat.copy()
        np.fill_diagonal(J, 0.0)

        def energy_of(x):
            return x @ h + x @ (J @ x)

        # Temperature schedule (geometric)
        T_hi, T_lo = 1.0/beta_range[0], 1.0/beta_range[1]
        temps = np.geomspace(T_hi, T_lo, num_sweeps)

        results = []
        for _ in range(num_reads):
            x = rng.integers(0, 2, size=n).astype(np.float64)
            E = energy_of(x)

            for T in temps:
                # Random single-bit flip
                k   = int(rng.integers(n))
                dE  = (1 - 2*x[k]) * (h[k] + (J[k,:] + J[:,k]) @ x)
                if dE < 0 or rng.random() < math.exp(-dE / T):
                    x[k] = 1 - x[k]
                    E   += dE

            results.append(({var_list[i]: int(x[i]) for i in range(n)}, E))

        results.sort(key=lambda t: t[1])
        return results


class PopulationSASampler:
    """
    Enhanced SA with:
      - Greedy feasibility seeding
      - Population of independent chains
      - Occasional perturbation restarts from elite solutions
    """

    def __init__(self, pop_size: int = 5, seed: int = 42):
        self.pop_size = pop_size
        self.rng      = np.random.default_rng(seed)
        self._base_sa = NumpySASampler()

    def _greedy_seed(self, data: InstanceData,
                     xmap, ymap, zmap) -> dict:
        """
        Construct a feasible seed:
          1. Assign each RDC to cheapest available CDC (b_jr).
          2. Activate CDCs used.
          3. Assign cheapest plant to each active CDC (a_ij).
        """
        sample = {}
        for nm in list(xmap.values()) + list(ymap.values()) + list(zmap.values()):
            sample[nm] = 0

        # Step 1: assign each RDC to lowest-cost CDC
        for r in data.R:
            best_j = min(data.T_r[r], key=lambda j: data.b_jr.get((j,r), 1e9))
            sample[ymap[(best_j, r)]] = 1

        # Step 2: activate CDCs that are assigned
        active_cdcs = set()
        for (j,r), nm in ymap.items():
            if sample[nm] == 1:
                active_cdcs.add(j)
        for j in active_cdcs:
            sample[zmap[j]] = 1

        # Step 3: for each active CDC assign cheapest plant
        for j in active_cdcs:
            if not data.N_c[j]:
                continue
            best_i = min(data.N_c[j], key=lambda i: data.a_ij.get((i,j), 1e9))
            sample[xmap[(best_i, j)]] = 1

        return sample

    def _seed_to_array(self, seed_dict, var_list, var_idx):
        n = len(var_list)
        x = np.zeros(n, dtype=np.float64)
        for nm, v in seed_dict.items():
            if nm in var_idx:
                x[var_idx[nm]] = float(v)
        return x

    def sample_qubo(self,
                    Q: dict,
                    data: InstanceData,
                    xmap, ymap, zmap,
                    num_reads: int = 1000,
                    num_sweeps: int = 3000,
                    beta_range: Tuple[float,float] = (0.1, 15.0),
                    ) -> List[Tuple[dict, float]]:
        """
        Returns list of (sample_dict, energy) best-first.
        """
        # Build index
        vars_set = set()
        for (u,v) in Q:
            vars_set.add(u); vars_set.add(v)
        var_list = sorted(vars_set)
        var_idx  = {v: i for i,v in enumerate(var_list)}
        n        = len(var_list)

        if n == 0:
            return [({}, 0.0)]

        # Dense matrices
        Qmat = np.zeros((n,n), dtype=np.float64)
        for (u,v), val in Q.items():
            i, j = var_idx[u], var_idx[v]
            Qmat[i,j] += val

        h = np.diag(Qmat).copy()
        J = Qmat.copy()
        np.fill_diagonal(J, 0.0)

        def energy_of(x):
            return float(x @ h + x @ (J @ x))

        T_hi, T_lo = 1.0/beta_range[0], 1.0/beta_range[1]
        temps = np.geomspace(T_hi, T_lo, num_sweeps)

        # Prepare initial population
        population = []

        # 1 greedy seed
        gs   = self._greedy_seed(data, xmap, ymap, zmap)
        x_gs = self._seed_to_array(gs, var_list, var_idx)
        population.append(x_gs)

        # rest random
        for _ in range(self.pop_size - 1):
            population.append(self.rng.integers(0,2,size=n).astype(np.float64))

        results = []
        reads_per_chain = max(1, num_reads // self.pop_size)

        for chain_init in population:
            for _ in range(reads_per_chain):
                x = chain_init.copy()
                E = energy_of(x)

                for T in temps:
                    k  = int(self.rng.integers(n))
                    dE = (1 - 2*x[k]) * (h[k] + (J[k,:] + J[:,k]) @ x)
                    if dE < 0 or self.rng.random() < math.exp(min(0, -dE/T)):
                        x[k] = 1 - x[k]
                        E   += dE

                sample = {var_list[i]: int(x[i]) for i in range(n)}
                results.append((sample, E))

                # Perturb for next read (restart from this solution w/ noise)
                flip_n = max(1, int(0.05 * n))
                idxs = self.rng.choice(n, flip_n, replace=False)
                chain_init = x.copy()
                chain_init[idxs] = 1 - chain_init[idxs]

        results.sort(key=lambda t: t[1])
        return results



# ──────────────────────────────────────────────────────────────────────────────
# Solution container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QUBOSolution:
    x:       Dict[Tuple[int,int], int]
    y:       Dict[Tuple[int,int], int]
    z:       Dict[int, int]
    obj:     float
    energy:  float
    feasible: bool = True


# ──────────────────────────────────────────────────────────────────────────────
# Main APL loop
# ──────────────────────────────────────────────────────────────────────────────

def extract_solution(sample: dict, xmap, ymap, zmap):
    x = {k: sample.get(v, 0) for k, v in xmap.items()}
    y = {k: sample.get(v, 0) for k, v in ymap.items()}
    z = {k: sample.get(v, 0) for k, v in zmap.items()}
    return x, y, z


def solve(data: InstanceData,
          num_reads:  int   = 8000,
          # MODIFY THIS 
          num_iters:  int   = 10,
          num_sweeps: int   = 5000,
          pop_size:   int   = 5,
          seed:       int   = 42,
          time_limit: float = 300.0,
          verbose:    bool  = True
          ) -> Tuple[Optional[QUBOSolution], List]:

    random.seed(seed)
    np.random.seed(seed)

    # ── Choose sampler ────────────────────────────────────────────────────────
    if HAS_NEAL:
        neal_sampler = neal.SimulatedAnnealingSampler()
        use_population = False
        if verbose:
            print("[Sampler] Using D-Wave Neal SA")
    else:
        pop_sampler    = PopulationSASampler(pop_size=pop_size, seed=seed)
        use_population = True
        if verbose:
            print("[Sampler] Using built-in Population SA (NumPy)")

    # ── Penalty initialisation ────────────────────────────────────────────────
    # Scale initial penalties to the objective magnitude
    obj_scale = max(
        max(data.a_ij.values()) if data.a_ij else 1,
        max(data.b_jr.values()) if data.b_jr else 1,
        max(data.c_j.values())  if data.c_j  else 1,
    )

    P1 = P2 = P3 = 10.0 * obj_scale    # Start at 10x (paper default)

    # Bounds to prevent instability
    P_min = 0.5  * obj_scale            # Never go below 0.5x objective scale
    P_max = 500.0 * obj_scale           # Never blow up beyond 500x

    # APL hyperparameters
    alpha_increase = 0.25               # Penalty growth rate when violated
    beta_decrease  = 0.05               # Penalty decay rate when satisfied

    best: Optional[QUBOSolution] = None
    elite_pool: List[QUBOSolution] = [] # Keep top-k feasible solutions

    penalty_log = []
    start_time  = time.time()

    if verbose:
        print(f"\n[APL] Initial penalties: P1={P1:.1f}, P2={P2:.1f}, P3={P3:.1f}")
        print(f"[APL] Bounds: P_min={P_min:.1f}, P_max={P_max:.1f}")
        print(f"[APL] Running {num_iters} iterations, "
              f"{num_reads} reads × {num_sweeps} sweeps each\n")

    for t in range(1, num_iters + 1):
        if time.time() - start_time > time_limit:
            if verbose:
                print(f"[APL] Time limit {time_limit}s reached at iter {t}.")
            break

        Q, xmap, ymap, zmap = build_qubo_with_slacks(data, P1, P2, P3)

        # ── Sample ───────────────────────────────────────────────────────────
        if use_population:
            samples = pop_sampler.sample_qubo(
                Q, data, xmap, ymap, zmap,
                num_reads  = num_reads,
                num_sweeps = num_sweeps,
            )
            sample_list = samples  # list of (dict, energy)
        else:
            sampleset   = neal_sampler.sample_qubo(
                Q, num_reads=num_reads, num_sweeps=num_sweeps)
            sample_list = [(s.sample, s.energy) for s in sampleset.data()]

        # ── Evaluate all samples ──────────────────────────────────────────────
        iter_best_v   = (math.inf, math.inf, math.inf)
        iter_best_sol = None
        iter_best_e   = math.inf
        last_v1 = last_v2 = last_v3 = 0.0

        for sample_dict, energy in sample_list:
            x, y, z = extract_solution(sample_dict, xmap, ymap, zmap)
            v1, v2, v3 = count_violations(data, x, y, z)

            # Only accept solutions with ZERO violations — no repair, no exceptions
            if v1 == 0 and v2 == 0 and v3 == 0:
                obj = original_objective(data, x, y, z)
                sol = QUBOSolution(x, y, z, obj, energy, feasible=True)
                elite_pool.append(sol)
                if best is None or obj < best.obj:
                    best = sol

            # Track the least-violated sample this iteration for APL penalty update
            vtuple = (v1, v2, v3)
            if sum(vtuple) < sum(iter_best_v):
                iter_best_v = vtuple
                iter_best_e = energy

        last_v1, last_v2, last_v3 = iter_best_v
        penalty_log.append((t, P1, P2, P3, last_v1, last_v2, last_v3,
                             best.obj if best else math.inf))

        # ── APL penalty update with clamping ─────────────────────────────────
        def update_penalty(P, v):
            if v > 0:
                return min(P_max, P * (1.0 + alpha_increase * v))
            else:
                return max(P_min, P * (1.0 - beta_decrease))

        P1 = update_penalty(P1, last_v1)
        P2 = update_penalty(P2, last_v2)
        P3 = update_penalty(P3, last_v3)

        if verbose:
            feasible_str = f"  ★ obj={best.obj:.2f}" if best else ""
            print(f"  Iter {t:3d} | P1={P1:9.2f} P2={P2:9.2f} P3={P3:9.2f}"
                  f" | v1={last_v1:.0f} v2={last_v2:.1f} v3={last_v3:.1f}"
                  f"{feasible_str}")

        # Early stop if already optimal-ish and violations gone
        if best is not None and last_v1 == 0 and t >= 5:
            # Keep running to try to improve until time/iter limit
            pass

    # Deduplicate elite pool
    elite_pool = sorted(
        {sol.obj: sol for sol in elite_pool if sol.feasible}.values(),
        key=lambda s: s.obj
    )

    return best, penalty_log, elite_pool


# ──────────────────────────────────────────────────────────────────────────────
# Output helpers
# ──────────────────────────────────────────────────────────────────────────────

def print_assignments(data: InstanceData, sol: QUBOSolution):
    print("\n=== ASSIGNMENTS ===")

    print("\nActivated CDCs (z_j = 1):")
    active = [j for j,v in sol.z.items() if v==1]
    if active:
        for j in sorted(active):
            print(f"  CDC {j}  (c_j={data.c_j[j]:.0f}, q_j={data.q_j[j]:.0f})")
    else:
        print("  None")

    print("\nFactory → CDC assignments (x_ij = 1):")
    found = False
    for (i,j),v in sorted(sol.x.items()):
        if v == 1:
            print(f"  Factory {i} → CDC {j}  "
                  f"(a_ij={data.a_ij.get((i,j),0):.0f}, p_i={data.p_i[i]:.0f})")
            found = True
    if not found:
        print("  None")

    print("\nCDC → RDC assignments (y_jr = 1):")
    found = False
    for (j,r),v in sorted(sol.y.items()):
        if v == 1:
            print(f"  CDC {j} → RDC {r}  "
                  f"(b_jr={data.b_jr.get((j,r),0):.0f}, d_r={data.d_r[r]:.0f})")
            found = True
    if not found:
        print("  None")


def print_results(data_orig: InstanceData,
                  data_used: InstanceData,
                  best: Optional[QUBOSolution],
                  elite_pool: list,
                  penalty_log: list,
                  solve_time: float,
                  used_ppln: bool):

    print("\n" + "="*65)
    print("  TLFLP QUBO SOLVER – RESULTS")
    print("="*65)

    print(f"\n Instance : |I|={len(data_used.I)}, "
          f"|C|={len(data_used.C)}, |R|={len(data_used.R)}")
    if used_ppln:
        print(f" PPLN     : |C| reduced from {len(data_orig.C)} → {len(data_used.C)}")
    print(f" Feasible : {best is not None}")

    if best:
        print(f"\n Objective (best feasible): {best.obj:.2f}")
        print(f" QUBO Energy             : {best.energy:.2f}")
        if len(elite_pool) > 1:
            print(f"\n Top-{min(5,len(elite_pool))} feasible solutions found:")
            for rank, sol in enumerate(elite_pool[:5], 1):
                print(f"   #{rank}  obj={sol.obj:.2f}")
        print_assignments(data_used, best)
    else:
        print("\n No feasible solution found. Best violations per iteration:")
        best_ilog = min(penalty_log, key=lambda r: r[4]+r[5]+r[6])
        print(f"   Iter {best_ilog[0]}: v1={best_ilog[4]:.0f} "
              f"v2={best_ilog[5]:.1f} v3={best_ilog[6]:.1f}")

    print(f"\n Solve Time: {solve_time:.2f}s")

    print("\n=== PENALTY EVOLUTION ===")
    header = ("Iter","|P1","|P2","|P3","v1","v2","v3","BestObj")
    print(f"{'Iter':>4} | {'P1':>10} | {'P2':>10} | {'P3':>10} "
          f"| {'v1':>4} | {'v2':>6} | {'v3':>6} | {'BestObj':>10}")
    print("-" * 72)
    for row in penalty_log:
        t, p1, p2, p3, v1, v2, v3, bo = row
        bo_str = f"{bo:.2f}" if bo < math.inf else "  –"
        print(f"{t:>4} | {p1:>10.2f} | {p2:>10.2f} | {p3:>10.2f} "
              f"| {v1:>4.0f} | {v2:>6.1f} | {v3:>6.1f} | {bo_str:>10}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Improved TSLND QUBO solver with PPLN + Population SA + APL")
    ap.add_argument("instance",
                    help="Path to instance .txt file")
    ap.add_argument("--reads",      type=int,   default=8000,
                    help="SA reads per APL iteration (default 8000)")
    ap.add_argument("--iters",      type=int,   default=10,
                    help="APL outer iterations (default 40)")
    ap.add_argument("--sweeps",     type=int,   default=5000,
                    help="SA sweeps per read (default 5000)")
    ap.add_argument("--pop_size",   type=int,   default=5,
                    help="Population size for SA (default 5)")
    ap.add_argument("--seed",       type=int,   default=42,
                    help="Random seed (default 42)")
    ap.add_argument("--time_limit", type=float, default=300.0,
                    help="Wall-clock time limit in seconds (default 300)")
    ap.add_argument("--no_ppln",    action="store_true",
                    help="Disable PPLN preprocessing")
    args = ap.parse_args()

    print("="*65)
    print("  TLFLP Improved QUBO Solver")
    print("  Ciacco et al. (2026) + PPLN + Population APL SA")
    print("="*65)

    # Parse
    data_orig = InstanceParser().parse_file(args.instance)
    print(f"\n[Instance] |I|={len(data_orig.I)}  "
          f"|C|={len(data_orig.C)}  |R|={len(data_orig.R)}")
    print(f"  Total supply: {sum(data_orig.p_i.values()):.0f}  "
          f"Total demand: {sum(data_orig.d_r.values()):.0f}")

    # PPLN
    use_ppln = not args.no_ppln
    if use_ppln:
        data_used = run_ppln(data_orig, verbose=True)
    else:
        print("\n[PPLN] Skipped (--no_ppln flag set).")
        data_used = data_orig

    # Solve
    t0 = time.time()
    best, penalty_log, elite_pool = solve(
        data_used,
        num_reads  = args.reads,
        num_iters  = args.iters,
        num_sweeps = args.sweeps,
        pop_size   = args.pop_size,
        seed       = args.seed,
        time_limit = args.time_limit,
        verbose    = True,
    )
    solve_time = time.time() - t0

    # Report
    print_results(data_orig, data_used, best, elite_pool,
                  penalty_log, solve_time, use_ppln)


if __name__ == "__main__":
    main()
