#!/usr/bin/env python3
"""
test2.py

TLFLP QUBO solver with:
- PPLN preprocessing
- Adaptive Penalty Learning (APL)
- Neal Simulated Annealing

Self-contained file.
"""

import argparse
import itertools
import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import neal
import dimod
from parser import InstanceParser, InstanceData
@dataclass
class QUBOSolution:
    x: Dict[Tuple[int, int], int]
    y: Dict[Tuple[int, int], int]
    z: Dict[int, int]
    obj: float
    energy: float


# =========================================================
# PPLN preprocessing (UNCHANGED from you)
# =========================================================

def ppln_preprocess(data, verbose=True):
    if verbose:
        print("\n--- Running PPLN preprocessing ---")

    total_demand = sum(data.d_r.values())
    max_q = max(data.q_j.values())
    beta = max(1, math.ceil(total_demand / max_q))

    for k in range(beta, len(data.C) + 1):
        for comb in itertools.combinations(data.C, k):
            cap = sum(data.q_j[j] for j in comb)
            if cap >= total_demand:
                if verbose:
                    print(f"  Found feasible CDC combination: {comb}")
                return list(comb)

    return data.C

# =========================================================
# Objective and violations
# =========================================================

def original_objective(data, x, y, z):
    return (
        sum(data.a_ij[(i, j)] * v for (i, j), v in x.items()) +
        sum(data.b_jr[(j, r)] * v for (j, r), v in y.items()) +
        sum(data.c_j[j] * v for j, v in z.items())
    )

def count_violations(data, x, y, z):
    v1 = v2 = v3 = 0.0

    # (13) each RDC served exactly once
    for r in data.R:
        served = sum(y.get((j, r), 0) for j in data.T_r[r])
        v1 += (served - 1) ** 2

    # (14) demand <= supply
    for j in data.C:
        demand = sum(data.d_r[r] * y.get((j, r), 0) for r in data.R)
        supply = sum(data.p_i[i] * x.get((i, j), 0) for i in data.I)
        if demand > supply:
            v2 += (demand - supply) ** 2

    # (15) supply <= capacity
    for j in data.C:
        supply = sum(data.p_i[i] * x.get((i, j), 0) for i in data.I)
        cap = data.q_j[j] * z.get(j, 0)
        if supply > cap:
            v3 += (supply - cap) ** 2

    return v1, v2, v3

# =========================================================
# QUBO builder (normalized)
# =========================================================

def build_qubo(data, P1, P2, P3, active_C):
    Q = {}
    x = {(i, j): f"x_{i}_{j}" for i in data.I for j in active_C}
    y = {(j, r): f"y_{j}_{r}" for j in active_C for r in data.R}
    z = {j: f"z_{j}" for j in active_C}

    d_norm = max(data.d_r.values())
    p_norm = max(data.p_i.values())
    q_norm = max(data.q_j.values())

    for (i, j), n in x.items():
        Q[(n, n)] = data.a_ij[(i, j)]

    for (j, r), n in y.items():
        Q[(n, n)] = data.b_jr[(j, r)]

    for j, n in z.items():
        Q[(n, n)] = data.c_j[j]

    for r in data.R:
        vars_r = [y[(j, r)] for j in active_C]
        for u, v in itertools.product(vars_r, repeat=2):
            Q[(u, v)] = Q.get((u, v), 0) + P1
        for u in vars_r:
            Q[(u, u)] -= 2 * P1

    for j in active_C:
        terms = []
        for r in data.R:
            terms.append((y[(j, r)], data.d_r[r] / d_norm))
        for i in data.I:
            terms.append((x[(i, j)], -data.p_i[i] / p_norm))

        for (v1, w1), (v2, w2) in itertools.product(terms, repeat=2):
            Q[(v1, v2)] = Q.get((v1, v2), 0) + P2 * w1 * w2

    for j in active_C:
        terms = [(x[(i, j)], data.p_i[i] / p_norm) for i in data.I]
        terms.append((z[j], -data.q_j[j] / q_norm))
        for (v1, w1), (v2, w2) in itertools.product(terms, repeat=2):
            Q[(v1, v2)] = Q.get((v1, v2), 0) + P3 * w1 * w2

    return Q, x, y, z

def print_assignments(data: InstanceData, sol: QUBOSolution):
    print("\n=== ASSIGNMENTS ===")

    # Activated CDCs
    print("\nActivated CDCs (z_j = 1):")
    active = False
    for j, v in sol.z.items():
        if v == 1:
            print(f"  CDC {j}")
            active = True
    if not active:
        print("  None")

    # Factory → CDC
    print("\nFactory → CDC assignments (x_ij = 1):")
    found = False
    for (i, j), v in sol.x.items():
        if v == 1:
            supply = data.p_i[i]
            print(f"  Factory {i} → CDC {j}  (supply={supply})")
            found = True
    if not found:
        print("  None")

    # CDC → RDC
    print("\nCDC → Zone assignments (y_jr = 1):")
    found = False
    for (j, r), v in sol.y.items():
        if v == 1:
            demand = data.d_r[r]
            print(f"  CDC {j} → Zone {r}  (demand={demand})")
            found = True
    if not found:
        print("  None")

# =========================================================
# Main
# =========================================================


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--reads", type=int, default=15000)
    ap.add_argument("--iters", type=int, default=25)
    args = ap.parse_args()

    data = InstanceParser().parse_file(args.instance)
    # active_C = ppln_preprocess(data)

    sampler = neal.SimulatedAnnealingSampler()

    scale = max(max(data.a_ij.values()), max(data.b_jr.values()), max(data.c_j.values()))
    P1 = P2 = P3 = scale
    P_min = 0.1 * scale
    P_max = 50 * scale
    alpha, beta = 0.2, 0.05

    best = None
    penalty_log = []

    start = time.time()

    for t in range(1, args.iters + 1):
        Q, xmap, ymap, zmap = build_qubo(data, P1, P2, P3, active_C)
        sampleset = sampler.sample_qubo(Q, num_reads=args.reads, num_sweeps=3000)

        for s in sampleset.data():
            sample = s.sample
            x = {(i, j): sample[xmap[(i, j)]] for (i, j) in xmap}
            y = {(j, r): sample[ymap[(j, r)]] for (j, r) in ymap}
            z = {j: sample[zmap[j]] for j in zmap}

            v1, v2, v3 = count_violations(data, x, y, z)

            if v1 == v2 == v3 == 0:
                obj = original_objective(data, x, y, z)
                if best is None or obj < best[1]:
                    best = (x, y, z, obj, s.energy)

        penalty_log.append((t, P1, P2, P3, v1, v2, v3))

        P1 = min(P_max, P1 * (1 + alpha * v1)) if v1 > 0 else max(P_min, P1 * (1 - beta))
        P2 = min(P_max, P2 * (1 + alpha * v2)) if v2 > 0 else max(P_min, P2 * (1 - beta))
        P3 = min(P_max, P3 * (1 + alpha * v3)) if v3 > 0 else max(P_min, P3 * (1 - beta))

    solve_time = time.time() - start

    print("\n=== QUBO RESULT ===")
    print(f" Feasible: {best is not None}")
    if best:
        print(f" Original Objective: {best[3]:.2f}")
        print(f" QUBO Energy: {best[4]:.2f}")
        print_assignments(data, best)
    print(f" Solve Time: {solve_time:.2f}s")

    print("\n=== PENALTY EVOLUTION ===")
    print("Iter |    P1    |    P2    |    P3    | v1 | v2 | v3")
    print("-" * 55)
    for r in penalty_log:
        print(f"{r[0]:>4} | {r[1]:>8.2f} | {r[2]:>8.2f} | {r[3]:>8.2f} | {r[4]:>2.0f} | {r[5]:>2.0f} | {r[6]:>2.0f}")

if __name__ == "__main__":
    main()
