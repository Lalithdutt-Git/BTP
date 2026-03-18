#!/usr/bin/env python3
"""
qubo_neal.py

TLFLP QUBO solver with:
- Paper-exact QUBO formulation
- Slack variables
- PPLN preprocessing
- Neal simulated annealing
- TRUE Adaptive Penalty Learning (APL)

Usage:
    python qubo_neal.py instance.txt --reads 8000 --use-ppln --max-iters 10
"""

import argparse
import itertools
import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List

import dimod
import neal

# ---------------------------
# Parser import
# ---------------------------
from parser import InstanceParser, InstanceData

# ---------------------------
# Optional Gurobi
# ---------------------------
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except Exception:
    HAS_GUROBI = False


# ============================================================
# Data structures
# ============================================================

@dataclass
class QUBOSolution:
    x_sol: Dict[Tuple[int, int], int]
    y_sol: Dict[Tuple[int, int], int]
    z_sol: Dict[int, int]
    original_obj: float
    qubo_energy: float
    offset: float
    feasible: bool
    violations: List[str]
    solve_time: float
    reads: int


# ============================================================
# Utilities
# ============================================================

def bits_needed(u: int) -> int:
    if u <= 0:
        return 1
    return max(1, math.ceil(math.log2(u + 1)))


def compute_original_objective(data: InstanceData, x_sol, y_sol, z_sol) -> float:
    obj = 0.0
    for (i, j), v in x_sol.items():
        obj += data.a_ij.get((i, j), 0) * v
    for (j, r), v in y_sol.items():
        obj += data.b_jr.get((j, r), 0) * v
    for j, v in z_sol.items():
        obj += data.c_j.get(j, 0) * v
    return float(obj)


# ============================================================
# PPLN preprocessing (unchanged)
# ============================================================

def ppln_preprocess(data: InstanceData, verbose=True):
    I, C, R = data.I, data.C, data.R
    total_demand = sum(data.d_r.values())
    max_q = max(data.q_j.values()) if data.q_j else 1
    beta = max(1, math.ceil(total_demand / max_q))

    A_j = {j: set() for j in C}
    for r in R:
        for j in data.T_r.get(r, []):
            A_j[j].add(r)

    for k in range(beta, len(C) + 1):
        for comb in itertools.combinations(C, k):
            covered = set()
            for j in comb:
                covered |= A_j[j]
            if len(covered) < len(R):
                continue
            if sum(data.q_j[j] for j in comb) < total_demand:
                continue
            return list(comb)

    return C


# ============================================================
# Build paper-exact QUBO
# ============================================================

def build_paper_qubo_bqm(data, CDC_list, P1, P2, P3):
    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)

    total_demand = sum(data.d_r.values())
    total_supply = sum(data.p_i.values())

    U1 = {j: total_demand for j in CDC_list}
    U2 = {j: max(total_supply, data.q_j[j]) for j in CDC_list}

    k1 = {j: bits_needed(U1[j]) for j in CDC_list}
    k2 = {j: bits_needed(U2[j]) for j in CDC_list}

    x, y, z, s1, s2 = {}, {}, {}, {}, {}

    def add_lin(v, b):
        bqm.add_variable(v, bqm.linear.get(v, 0.0) + b)

    def add_quad(u, v, b):
        bqm.add_interaction(u, v, b)

    for j in CDC_list:
        z[j] = f"z_{j}"
        for i in data.N_c.get(j, []):
            x[i, j] = f"x_{i}_{j}"
        for r in data.R:
            if j in data.T_r.get(r, []):
                y[j, r] = f"y_{j}_{r}"
        for l in range(k1[j]):
            s1[j, l] = f"s1_{j}_{l}"
        for l in range(k2[j]):
            s2[j, l] = f"s2_{j}_{l}"

    # Objective
    for (i, j), c in data.a_ij.items():
        if (i, j) in x:
            add_lin(x[i, j], c)
    for (j, r), c in data.b_jr.items():
        if (j, r) in y:
            add_lin(y[j, r], c)
    for j, c in data.c_j.items():
        add_lin(z[j], c)

    # Constraint (13)
    for r in data.R:
        ys = [y[j, r] for j in data.T_r.get(r, []) if (j, r) in y]
        for u in ys:
            add_lin(u, -2 * P1 + P1)
        for i in range(len(ys)):
            for j in range(i + 1, len(ys)):
                add_quad(ys[i], ys[j], 2 * P1)
        bqm.offset += P1

    # Constraint (14)
    for j in CDC_list:
        terms, coeffs = [], []
        for r in data.R:
            if (j, r) in y:
                terms.append(y[j, r])
                coeffs.append(data.d_r[r])
        for i in data.N_c.get(j, []):
            if (i, j) in x:
                terms.append(x[i, j])
                coeffs.append(-data.p_i[i])
        for l in range(k1[j]):
            terms.append(s1[j, l])
            coeffs.append(2 ** l)
        for t in range(len(terms)):
            add_lin(terms[t], P2 * coeffs[t] ** 2)
            for u in range(t + 1, len(terms)):
                add_quad(terms[t], terms[u], 2 * P2 * coeffs[t] * coeffs[u])

    # Constraint (15)
    for j in CDC_list:
        terms, coeffs = [], []
        for i in data.N_c.get(j, []):
            if (i, j) in x:
                terms.append(x[i, j])
                coeffs.append(data.p_i[i])
        terms.append(z[j])
        coeffs.append(-data.q_j[j])
        for l in range(k2[j]):
            terms.append(s2[j, l])
            coeffs.append(2 ** l)
        for t in range(len(terms)):
            add_lin(terms[t], P3 * coeffs[t] ** 2)
            for u in range(t + 1, len(terms)):
                add_quad(terms[t], terms[u], 2 * P3 * coeffs[t] * coeffs[u])

    var_maps = {'x': x, 'y': y, 'z': z}
    return bqm, var_maps


# ============================================================
# Decode and validate
# ============================================================

def decode_validate(data, var_maps, sample):
    violations = []
    x_sol = {(i, j): sample.get(v, 0) for (i, j), v in var_maps['x'].items()}
    y_sol = {(j, r): sample.get(v, 0) for (j, r), v in var_maps['y'].items()}
    z_sol = {j: sample.get(v, 0) for j, v in var_maps['z'].items()}

    for r in data.R:
        if sum(y_sol.get((j, r), 0) for j in data.T_r.get(r, [])) != 1:
            violations.append("assign")

    for j in data.C:
        d = sum(data.d_r[r] * y_sol.get((j, r), 0) for r in data.R)
        s = sum(data.p_i[i] * x_sol.get((i, j), 0) for i in data.N_c.get(j, []))
        if d > s:
            violations.append("flow")

    for j in data.C:
        s = sum(data.p_i[i] * x_sol.get((i, j), 0) for i in data.N_c.get(j, []))
        if s > data.q_j[j] * z_sol.get(j, 0):
            violations.append("capacity")

    return len(violations) == 0, violations, x_sol, y_sol, z_sol


def print_assignments(data, sol: QUBOSolution):
    print("\n=== ASSIGNMENTS ===")

    # Activated CDCs
    active_cdcs = [j for j, v in sol.z_sol.items() if v == 1]
    print("\nActivated CDCs (z_j = 1):")
    if active_cdcs:
        for j in active_cdcs:
            print(f"  CDC {j}")
    else:
        print("  None")

    # Factory → CDC
    print("\nFactory → CDC assignments (x_ij = 1):")
    found = False
    for (i, j), v in sol.x_sol.items():
        if v == 1:
            print(f"  Factory {i} → CDC {j}  (supply={data.p_i.get(i, 0)})")
            found = True
    if not found:
        print("  None")

    # CDC → Zone
    print("\nCDC → Zone assignments (y_jr = 1):")
    found = False
    for (j, r), v in sol.y_sol.items():
        if v == 1:
            print(f"  CDC {j} → Zone {r}  (demand={data.d_r.get(r, 0)})")
            found = True
    if not found:
        print("  None")


# ============================================================
# MAIN (APL CORE)
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--reads", type=int, default=8000)
    ap.add_argument("--use-ppln", action="store_true")
    ap.add_argument("--max-iters", type=int, default=10)
    args = ap.parse_args()

    data = InstanceParser().parse_file(args.instance)

    CDC_list = ppln_preprocess(data) if args.use_ppln else data.C

    max_bias = max(
        max(data.a_ij.values(), default=1),
        max(data.b_jr.values(), default=1),
        max(data.c_j.values(), default=1),
    )
    baseP = 10 * max_bias

    # APL parameters
    alpha = 0.25
    beta = 0.05

    P1, P2, P3 = 3 * baseP, baseP, baseP

    sampler = neal.SimulatedAnnealingSampler()
    best_solution = None

    for t in range(1, args.max_iters + 1):
        print(f"\nAPL Iter {t} → P1={P1:.2e}, P2={P2:.2e}, P3={P3:.2e}")

        bqm, var_maps = build_paper_qubo_bqm(data, CDC_list, P1, P2, P3)
        bqm.normalize()

        start = time.time()
        sampleset = sampler.sample(bqm, num_reads=args.reads)
        elapsed = time.time() - start

        v_assign = v_flow = v_cap = 0

        for s in sampleset:
            feasible, viol, x, y, z = decode_validate(data, var_maps, s)
            if feasible:
                obj = compute_original_objective(data, x, y, z)
                best_solution = QUBOSolution(
                    x, y, z, obj, sampleset.first.energy,
                    bqm.offset, True, [], elapsed, args.reads
                )
                print("✅ Feasible solution found")
                print("\n=== QUBO RESULT ===")
                print(f" Feasible: {best_solution.feasible}")
                print(f" Original Objective: {best_solution.original_obj:.2f}")
                print(f" QUBO Energy: {best_solution.qubo_energy:.2f}")
                print(f" Solve Time: {best_solution.solve_time:.2f}s")

                print_assignments(data, best_solution)

                return

            for v in viol:
                if v == "assign":
                    v_assign += 1
                elif v == "flow":
                    v_flow += 1
                elif v == "capacity":
                    v_cap += 1

        if v_assign > 0:
            P1 *= (1 + alpha)
        else:
            P1 *= (1 - beta)

        if v_flow > 0:
            P2 *= (1 + alpha)
        else:
            P2 *= (1 - beta)

        if v_cap > 0:
            P3 *= (1 + alpha)
        else:
            P3 *= (1 - beta)

        print(f"Violations → assign={v_assign}, flow={v_flow}, cap={v_cap}")

    print("❌ No feasible solution found.")


if __name__ == "__main__":
    main()
