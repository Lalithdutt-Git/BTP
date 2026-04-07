#!/usr/bin/env python3
"""
tslnd_solver_parallel.py
========================
High-Performance TLFLP QUBO Solver — optimised for high-RAM / multi-core machines.

Based on tslnd_solver_final.py with the following enhancements:
  1. Parallel SA reads — split across CPU cores via multiprocessing.Pool,
     each worker creates its own neal.SimulatedAnnealingSampler instance.
  2. Higher default budgets — more reads, sweeps, and iterations.
  3. CLI knobs — --workers, and higher defaults for --reads / --sweeps.

Uses ONLY D-Wave Neal for simulated annealing (no NumPy fallback).

Usage:
    python tslnd_solver_parallel.py instance_5I_10C_15R.txt [options]

Options:
    --reads      INT   Total SA reads per APL iteration   (default 16000)
    --iters      INT   APL outer iterations               (default 60)
    --sweeps     INT   SA sweeps per read                 (default 8000)
    --workers    INT   Number of parallel worker processes (default: CPU count)
    --no_ppln          Skip PPLN preprocessing
    --seed       INT   Random seed                        (default 42)
    --time_limit FLOAT Wall-clock time limit (s)          (default 600)
"""

import argparse
import itertools
import math
import multiprocessing
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import neal
import dimod

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
    """
    supply_plants = set()
    for j in combo:
        for i in data.N_c[j]:
            supply_plants.add(i)
    total_supply = sum(data.p_i[i] for i in supply_plants)
    total_demand = sum(data.d_r[r] for r in data.R)

    if total_demand > total_supply:
        return False

    max_cap = max(data.q_j[j] for j in combo)
    if total_supply > max_cap:
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

        covering = [(combo, combo_cost(combo, data))
                    for combo in combos
                    if combo_covers_all_rdc(combo, data)]

        covering.sort(key=lambda x: x[1])

        for combo, cost in covering:
            if check_capacity_constraints(combo, data):
                if cost < best_cost:
                    best_combo = combo
                    best_cost  = cost
                break

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
    for r in data.R:
        vars_r = [ymap[(j,r)] for j in data.T_r[r]]
        for u in vars_r:
            add(u, u, P1 * (1 - 2))
        for u, v in itertools.combinations(vars_r, 2):
            add(u, v, 2 * P1)

    # ── Penalties (14) & (15): with slack variables ──────────────────────────
    for j in data.C:
        # Constraint (14): sum_r d_r*y_{j,r} - sum_i p_i*x_{i,j} + S1_j = 0
        U1_j = sum(data.d_r[r] for r in data.R if (j,r) in ymap)
        k1   = _bits_needed(U1_j)
        s1_vars = [f"s1_{j}_{l}" for l in range(k1)]
        s1_wts  = [2**l for l in range(k1)]

        # Constraint (15): sum_i p_i*x_{i,j} - q_j*z_j + S2_j = 0
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
# Greedy seed builder
# ──────────────────────────────────────────────────────────────────────────────

def build_greedy_seed(data: InstanceData, xmap, ymap, zmap, Q: dict) -> dict:
    """
    Construct a smart starting point for SA covering EVERY variable in the QUBO
    (decision variables + slack variables).
    """
    all_qubo_vars = set()
    for (u, v) in Q:
        all_qubo_vars.add(u)
        all_qubo_vars.add(v)

    sample = {var: 0 for var in all_qubo_vars}

    # Step 1: assign each RDC to lowest-cost CDC
    for r in data.R:
        best_j = min(data.T_r[r], key=lambda j: data.b_jr.get((j, r), 1e9))
        sample[ymap[(best_j, r)]] = 1

    # Step 2: activate CDCs that received an RDC assignment
    active_cdcs = set()
    for (j, r), nm in ymap.items():
        if sample[nm] == 1:
            active_cdcs.add(j)
    for j in active_cdcs:
        sample[zmap[j]] = 1

    # Step 3: assign cheapest plant to each active CDC
    for j in active_cdcs:
        if not data.N_c[j]:
            continue
        best_i = min(data.N_c[j], key=lambda i: data.a_ij.get((i, j), 1e9))
        sample[xmap[(best_i, j)]] = 1

    return sample


# ──────────────────────────────────────────────────────────────────────────────
# Parallel Neal worker — top-level function so it is picklable
# ──────────────────────────────────────────────────────────────────────────────

def _neal_worker(args):
    """
    Worker function for multiprocessing.  Each worker creates its own
    neal.SimulatedAnnealingSampler instance (avoids pickle issues).

    Args (unpacked from tuple):
        Q:            QUBO dict {(u,v): coeff}
        greedy_seed:  dict {var_name: 0/1} — injected as initial_state
                      for the FIRST worker only (worker_id == 0)
        num_reads:    number of SA reads this worker should perform
        num_sweeps:   SA sweeps per read
        worker_id:    integer id of this worker (0-based)
        seed:         random seed for this worker

    Returns:
        list of (sample_dict, energy) tuples
    """
    Q, greedy_seed, num_reads, num_sweeps, worker_id, seed = args

    # Each worker creates its own fresh sampler
    sampler = neal.SimulatedAnnealingSampler()

    # Only worker 0 gets the greedy seed as initial state
    if worker_id == 0 and greedy_seed is not None:
        initial_ss = dimod.SampleSet.from_samples(
            greedy_seed,
            vartype=dimod.BINARY,
            energy=0.0
        )
        sampleset = sampler.sample_qubo(
            Q,
            num_reads      = num_reads,
            num_sweeps     = num_sweeps,
            initial_states = initial_ss,
            seed           = seed,
        )
    else:
        sampleset = sampler.sample_qubo(
            Q,
            num_reads  = num_reads,
            num_sweeps = num_sweeps,
            seed       = seed,
        )

    # Convert to plain dicts (picklable across process boundary)
    results = []
    for datum in sampleset.data():
        sample_dict = dict(datum.sample)
        results.append((sample_dict, float(datum.energy)))

    return results


def parallel_neal_sample(Q: dict,
                         greedy_seed: dict,
                         num_reads: int,
                         num_sweeps: int,
                         n_workers: int,
                         seed: int) -> List[Tuple[dict, float]]:
    """
    Distribute SA reads evenly across n_workers processes, each running
    its own neal.SimulatedAnnealingSampler.  Merge and sort results.
    """
    # Distribute reads evenly; give remainder to last worker
    base_reads = num_reads // n_workers
    remainder  = num_reads % n_workers

    tasks = []
    for w in range(n_workers):
        w_reads = base_reads + (1 if w < remainder else 0)
        tasks.append((Q, greedy_seed, w_reads, num_sweeps, w, seed + w))

    with multiprocessing.Pool(processes=n_workers) as pool:
        worker_results = pool.map(_neal_worker, tasks)

    # Flatten all worker results into a single list
    flat = []
    for worker_result in worker_results:
        flat.extend(worker_result)

    # Sort by energy (lowest first)
    flat.sort(key=lambda t: t[1])
    return flat


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
          num_reads:  int   = 16000,
          num_iters:  int   = 60,
          num_sweeps: int   = 8000,
          n_workers:  int   = 1,
          seed:       int   = 42,
          time_limit: float = 600.0,
          verbose:    bool  = True
          ) -> Tuple[Optional[QUBOSolution], List]:

    random.seed(seed)
    np.random.seed(seed)

    # ── Choose sampling strategy ──────────────────────────────────────────────
    use_parallel = n_workers > 1
    if verbose:
        if use_parallel:
            print(f"[Sampler] Using D-Wave Neal SA with {n_workers} parallel workers")
            print(f"          ({num_reads // n_workers}+ reads/worker × "
                  f"{num_sweeps} sweeps each)")
        else:
            print(f"[Sampler] Using D-Wave Neal SA  (single-process, "
                  f"{num_reads} reads × {num_sweeps} sweeps)")
        print(f"          Greedy seed applied to first read")

    # Create single-process sampler (used only when n_workers == 1)
    if not use_parallel:
        sampler = neal.SimulatedAnnealingSampler()

    # ── Penalty initialisation ────────────────────────────────────────────────
    obj_scale = max(
        max(data.a_ij.values()) if data.a_ij else 1,
        max(data.b_jr.values()) if data.b_jr else 1,
        max(data.c_j.values())  if data.c_j  else 1,
    )

    P1 = P2 = P3 = 10.0 * obj_scale
    P_min = 0.5  * obj_scale
    P_max = 500.0 * obj_scale

    alpha_increase = 0.25
    beta_decrease  = 0.05

    best: Optional[QUBOSolution] = None
    elite_pool: List[QUBOSolution] = []
    best_obj_iter: Optional[int] = None
    num_qubits: int = 0

    penalty_log = []
    start_time  = time.time()

    if verbose:
        print(f"\n[APL] Initial penalties: P1={P1:.1f}, P2={P2:.1f}, P3={P3:.1f}")
        print(f"[APL] Bounds: P_min={P_min:.1f}, P_max={P_max:.1f}")
        print(f"[APL] Running {num_iters} iterations, "
              f"{num_reads} total reads × {num_sweeps} sweeps each\n")

    for t in range(1, num_iters + 1):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            if verbose:
                print(f"[APL] Time limit {time_limit}s reached at iter {t}.")
            break

        Q, xmap, ymap, zmap = build_qubo_with_slacks(data, P1, P2, P3)

        # ── Count qubits once ──
        if num_qubits == 0:
            all_vars = set()
            for (u, v) in Q:
                all_vars.add(u)
                all_vars.add(v)
            num_qubits = len(all_vars)
            n_decision = len(xmap) + len(ymap) + len(zmap)
            n_slack    = num_qubits - n_decision
            if verbose:
                print(f"[QUBO] Total variables (qubits): {num_qubits}  "
                      f"(decision={n_decision}, slack={n_slack})\n")

        # ── Build greedy seed ──
        greedy_seed = build_greedy_seed(data, xmap, ymap, zmap, Q)

        # ── Sample ────────────────────────────────────────────────────────────
        if use_parallel:
            # Distribute reads across workers, each with its own Neal instance
            sample_list = parallel_neal_sample(
                Q, greedy_seed, num_reads, num_sweeps, n_workers, seed + t
            )
        else:
            # Single-process Neal
            initial_ss = dimod.SampleSet.from_samples(
                greedy_seed,
                vartype=dimod.BINARY,
                energy=0.0
            )
            sampleset = sampler.sample_qubo(
                Q,
                num_reads      = num_reads,
                num_sweeps     = num_sweeps,
                initial_states = initial_ss,
            )
            sample_list = [(dict(s.sample), float(s.energy))
                           for s in sampleset.data()]

        # ── Evaluate all samples ──────────────────────────────────────────────
        iter_best_v = (math.inf, math.inf, math.inf)
        iter_best_e = math.inf

        for sample_dict, energy in sample_list:
            x, y, z = extract_solution(sample_dict, xmap, ymap, zmap)
            v1, v2, v3 = count_violations(data, x, y, z)

            if v1 == 0 and v2 == 0 and v3 == 0:
                obj = original_objective(data, x, y, z)
                sol = QUBOSolution(x, y, z, obj, energy, feasible=True)
                elite_pool.append(sol)
                if best is None or obj < best.obj:
                    best = sol
                    best_obj_iter = t

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
            pass

    # Deduplicate elite pool
    elite_pool = sorted(
        {sol.obj: sol for sol in elite_pool if sol.feasible}.values(),
        key=lambda s: s.obj
    )

    return best, penalty_log, elite_pool, num_qubits, best_obj_iter


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
                  used_ppln: bool,
                  num_qubits: int,
                  best_obj_iter: Optional[int],
                  n_workers: int):

    print("\n" + "="*65)
    print("  TLFLP QUBO SOLVER – RESULTS  (Parallel Edition)")
    print("="*65)

    print(f"\n Instance : |I|={len(data_used.I)}, "
          f"|C|={len(data_used.C)}, |R|={len(data_used.R)}")
    if used_ppln:
        print(f" PPLN     : |C| reduced from {len(data_orig.C)} → {len(data_used.C)}")
    print(f" Qubits   : {num_qubits}  (total QUBO variables used by the sampler)")
    print(f" Workers  : {n_workers}")
    print(f" Feasible : {best is not None}")

    if best:
        print(f"\n Objective (best feasible) : {best.obj:.2f}")
        print(f" QUBO Energy               : {best.energy:.2f}")
        print(f" Solution found at iter    : {best_obj_iter}")
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
        description="High-Performance TLFLP QUBO solver — PPLN + Parallel Neal SA + APL")
    ap.add_argument("instance",
                    help="Path to instance .txt file")
    ap.add_argument("--reads",      type=int,   default=16000,
                    help="Total SA reads per APL iteration (default 16000)")
    ap.add_argument("--iters",      type=int,   default=60,
                    help="APL outer iterations (default 60)")
    ap.add_argument("--sweeps",     type=int,   default=8000,
                    help="SA sweeps per read (default 8000)")
    ap.add_argument("--workers",    type=int,   default=0,
                    help="Number of parallel worker processes "
                         "(default: auto = CPU count)")
    ap.add_argument("--seed",       type=int,   default=42,
                    help="Random seed (default 42)")
    ap.add_argument("--time_limit", type=float, default=600.0,
                    help="Wall-clock time limit in seconds (default 600)")
    ap.add_argument("--no_ppln",    action="store_true",
                    help="Disable PPLN preprocessing")
    args = ap.parse_args()

    # Auto-detect worker count
    if args.workers <= 0:
        args.workers = max(1, os.cpu_count() or 1)

    print("="*65)
    print("  TLFLP High-Performance QUBO Solver  (Parallel Edition)")
    print("  Ciacco et al. (2026) + PPLN + Greedy-Seeded APL SA")
    print("="*65)
    print(f"\n[Config] Workers: {args.workers} | Reads: {args.reads} | "
          f"Sweeps: {args.sweeps} | Iters: {args.iters}")
    print(f"[Config] Time limit: {args.time_limit}s | Seed: {args.seed}")

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
    best, penalty_log, elite_pool, num_qubits, best_obj_iter = solve(
        data_used,
        num_reads  = args.reads,
        num_iters  = args.iters,
        num_sweeps = args.sweeps,
        n_workers  = args.workers,
        seed       = args.seed,
        time_limit = args.time_limit,
        verbose    = True,
    )
    solve_time = time.time() - t0

    # Report
    print_results(data_orig, data_used, best, elite_pool,
                  penalty_log, solve_time, use_ppln,
                  num_qubits, best_obj_iter, args.workers)


if __name__ == "__main__":
    main()
