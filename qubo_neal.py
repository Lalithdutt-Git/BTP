#!/usr/bin/env python3
"""
qubo_neal.py

TLFLP QUBO solver with PPLN preprocessing + Neal (local simulated annealer).
Also supports optional Gurobi reference and warm-start.

Usage:
    python test.py instance_2I_4C_6R.txt --reads 8000 --use-ppln --gurobi-time 5 --warmstart

Requirements:
    pip install neal dimod numpy
    (optional) gurobipy for Gurobi reference/warmstart
"""

import argparse
import itertools
import math
import time
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, List

import dimod
import neal

# Import your parser (make sure parser.py is present)
try:
    from parser import InstanceParser, InstanceData
except Exception as e:
    raise ImportError("Cannot import parser.py. Ensure parser.py is in the same folder and provides InstanceParser.") from e

# Optional Gurobi
try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS_GUROBI = True
except Exception:
    HAS_GUROBI = False


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


# ---------------------------
# Utilities
# ---------------------------
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


# ---------------------------
# PPLN preprocessing
# ---------------------------
def ppln_preprocess(data: InstanceData, verbose: bool = True):
    """
    Conservative PPLN implementation:
      - finds a small feasible subset of CDCs (if any) by combinatorial search
      - search size increases from lower bound beta to |C|
    Returns: reduced_C (list of CDC indices). If no reduction possible, returns original C.
    """
    if verbose:
        print("\n--- Running PPLN preprocessing ---")

    I = data.I
    C = data.C
    R = data.R
    p_i = data.p_i
    q_j = data.q_j
    d_r = data.d_r
    N_c = data.N_c
    T_r = data.T_r

    total_demand = sum(d_r.values())
    total_supply = sum(p_i.values())

    # conservative lower bound (beta): minimum number of CDCs to cover total demand by max capacity
    max_q = max(q_j.values()) if q_j else 0
    if max_q <= 0:
        beta = 1
    else:
        beta = math.ceil(total_demand / max_q)
    if beta < 1:
        beta = 1
    beta = min(beta, len(C))

    # upper bound we allow is full set but we'll search increasing sizes
    if verbose:
        print(f" total_demand={total_demand}, total_supply={total_supply}, beta(lower)={beta}")

    # Precompute A_j: which RDCs each CDC j can serve (i.e., set of r with j in T_r[r])
    A_j = {j: set() for j in C}
    for r in R:
        for j in T_r.get(r, []):
            if j in A_j:
                A_j[j].add(r)

    # try combinations from size beta upward
    for k in range(beta, len(C) + 1):
        if verbose:
            print(f" Trying combinations with {k} CDC(s)...")
        for comb in itertools.combinations(C, k):
            # coverage check
            covered = set()
            for j in comb:
                covered |= A_j.get(j, set())
            if len(covered) < len(R):
                continue
            # capacity check (total capacity of chosen CDCs must >= total demand)
            cap_sum = sum(q_j.get(j, 0) for j in comb)
            if cap_sum < total_demand:
                continue
            # supply check: all required production must be <= total supply
            if total_supply < total_demand:
                continue
            # if reaches here, comb is a feasible candidate w.r.t coverage and capacities
            if verbose:
                print(f"  Found feasible CDC combination: {comb} (cap_sum={cap_sum})")
            return list(comb)

    if verbose:
        print(" PPLN found no strict reduction; using full CDC set.")
    return C


# ---------------------------
# Gurobi reference (optional)
# ---------------------------
def gurobi_reference(data: InstanceData, timelimit: float = 5.0):
    if not HAS_GUROBI:
        return None
    model = gp.Model("TLFLP_ref")
    model.setParam('OutputFlag', 0)
    if timelimit > 0:
        model.setParam('TimeLimit', timelimit)

    x = {}
    for j in data.C:
        for i in data.N_c.get(j, []):
            x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    y = {}
    for r in data.R:
        for j in data.T_r.get(r, []):
            y[j, r] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}_{r}")

    z = {}
    for j in data.C:
        z[j] = model.addVar(vtype=GRB.BINARY, name=f"z_{j}")

    model.update()

    # objective
    model.setObjective(
        gp.quicksum(data.a_ij.get((i, j), 0) * x[i, j] for (i, j) in x)
        + gp.quicksum(data.b_jr.get((j, r), 0) * y[j, r] for (j, r) in y)
        + gp.quicksum(data.c_j.get(j, 0) * z[j] for j in z),
        GRB.MINIMIZE
    )

    # constraints
    for r in data.R:
        model.addConstr(gp.quicksum(y[j, r] for j in data.T_r.get(r, [])) == 1, name=f"assign_r_{r}")

    for j in data.C:
        lhs = gp.quicksum(data.d_r.get(r, 0) * y[j, r] for r in data.R if (j, r) in y)
        rhs = gp.quicksum(data.p_i.get(i, 0) * x[i, j] for i in data.N_c.get(j, []) if (i, j) in x)
        model.addConstr(lhs <= rhs, name=f"demand_supply_{j}")

    for j in data.C:
        lhs = gp.quicksum(data.p_i.get(i, 0) * x[i, j] for i in data.N_c.get(j, []) if (i, j) in x)
        model.addConstr(lhs <= data.q_j.get(j, 0) * z[j], name=f"capacity_{j}")

    model.optimize()
    if model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
        sol_x = {(i, j): int(x[i, j].X + 0.5) for (i, j) in x}
        sol_y = {(j, r): int(y[j, r].X + 0.5) for (j, r) in y}
        sol_z = {j: int(z[j].X + 0.5) for j in z}
        return {'x': sol_x, 'y': sol_y, 'z': sol_z, 'objval': float(model.ObjVal)}
    return None


# ---------------------------
# Build QUBO (paper exact)
# ---------------------------
def build_paper_qubo_bqm(data: InstanceData, CDC_list: List[int], P1: float, P2: float, P3: float):
    """
    Build paper-exact QUBO BQM for CDCs restricted to CDC_list.
    Returns: bqm, var_maps
    var_maps = { 'x': {(i,j): name}, 'y': {(j,r): name}, 'z': {j: name}, 's1': {...}, 's2': {...}, 'k1': {...}, 'k2': {...} }
    """
    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, vartype=dimod.BINARY)

    # slack bounds
    total_demand = sum(data.d_r.values())
    total_supply = sum(data.p_i.values())
    U1 = {j: total_demand for j in CDC_list}
    U2 = {j: max(total_supply, data.q_j.get(j, 0)) for j in CDC_list}
    k1 = {j: bits_needed(U1[j]) for j in CDC_list}
    k2 = {j: bits_needed(U2[j]) for j in CDC_list}

    # variable maps
    x_vars = {}
    y_vars = {}
    z_vars = {}
    s1_vars = {}
    s2_vars = {}

    for j in CDC_list:
        z_vars[j] = f"z_{j}"
        # x: only for plants allowed in N_c[j]
        for i in data.N_c.get(j, []):
            x_vars[(i, j)] = f"x_{i}_{j}"
        # y: only for r where j in T_r[r]
        for r in data.R:
            if j in data.T_r.get(r, []):
                y_vars[(j, r)] = f"y_{j}_{r}"
        # slack bits
        for l in range(k1[j]):
            s1_vars[(j, l)] = f"s1_{j}_{l}"
        for l in range(k2[j]):
            s2_vars[(j, l)] = f"s2_{j}_{l}"

    # helpers
    def add_lin(var, bias):
        # bqm.add_variable exists in many dimod versions
        if var in bqm.linear:
            bqm.linear[var] += bias
        else:
            bqm.add_variable(var, bias)

    def add_quad(u, v, bias):
        try:
            bqm.add_interaction(u, v, bias)
        except Exception:
            # fallback: update quadratic dict
            bqm.quadratic[(u, v)] = bqm.quadratic.get((u, v), 0.0) + bias

    # Objective: a_ij x + b_jr y + c_j z
    for (i, j), cost in data.a_ij.items():
        if (i, j) in x_vars:
            add_lin(x_vars[(i, j)], float(cost))
    for (j, r), cost in data.b_jr.items():
        if (j, r) in y_vars:
            add_lin(y_vars[(j, r)], float(cost))
    for j, cost in data.c_j.items():
        if j in z_vars:
            add_lin(z_vars[j], float(cost))

    # Penalty (13): each r served exactly once
    for r in data.R:
        ys = [y_vars[(j, r)] for j in data.T_r.get(r, []) if (j, r) in y_vars]
        if not ys:
            # region has no CDCs in reduced set -> infeasible instance (we'll rely on penalties)
            continue
        # (sum y)^2 -> linear + pairwise quad
        for u in ys:
            add_lin(u, P1)          # y^2 coefficient
            add_lin(u, -2.0 * P1)   # -2*y term
        # pairwise terms
        for i in range(len(ys)):
            for jdx in range(i + 1, len(ys)):
                add_quad(ys[i], ys[jdx], 2.0 * P1)
        bqm.offset += P1

    # Penalty (14): for each j: (sum_r d_r y_jr - sum_i p_i x_ij + S1_j)^2
    for j in CDC_list:
        terms = []
        coeffs = []
        # demand terms
        for r in data.R:
            if (j, r) in y_vars:
                terms.append(y_vars[(j, r)])
                coeffs.append(float(data.d_r.get(r, 0)))
        # supply terms (negative)
        for i in data.N_c.get(j, []):
            if (i, j) in x_vars:
                terms.append(x_vars[(i, j)])
                coeffs.append(float(-data.p_i.get(i, 0)))
        # slack S1 bits
        for l in range(k1[j]):
            terms.append(s1_vars[(j, l)])
            coeffs.append(float(2 ** l))
        # expand square
        for t in range(len(terms)):
            add_lin(terms[t], P2 * coeffs[t] * coeffs[t])
            for u in range(t + 1, len(terms)):
                add_quad(terms[t], terms[u], 2.0 * P2 * coeffs[t] * coeffs[u])

    # Penalty (15): for each j: (sum_i p_i x_ij - q_j z_j + S2_j)^2
    for j in CDC_list:
        terms = []
        coeffs = []
        # supply terms
        for i in data.N_c.get(j, []):
            if (i, j) in x_vars:
                terms.append(x_vars[(i, j)])
                coeffs.append(float(data.p_i.get(i, 0)))
        # - q_j * z_j
        if j in z_vars:
            terms.append(z_vars[j])
            coeffs.append(float(-data.q_j.get(j, 0)))
        # slack S2 bits
        for l in range(k2[j]):
            terms.append(s2_vars[(j, l)])
            coeffs.append(float(2 ** l))
        # expand square
        for t in range(len(terms)):
            add_lin(terms[t], P3 * coeffs[t] * coeffs[t])
            for u in range(t + 1, len(terms)):
                add_quad(terms[t], terms[u], 2.0 * P3 * coeffs[t] * coeffs[u])

    var_maps = {'x': x_vars, 'y': y_vars, 'z': z_vars, 's1': s1_vars, 's2': s2_vars, 'k1': k1, 'k2': k2}
    return bqm, var_maps


# ---------------------------
# Decode and validate
# ---------------------------
def decode_validate(data: InstanceData, var_maps, sample: Dict[str, int]):
    violations = []
    tol = 1e-6
    # extract
    x_sol = {}
    for (i, j), name in var_maps['x'].items():
        x_sol[(i, j)] = int(bool(sample.get(name, 0)))
    y_sol = {}
    for (j, r), name in var_maps['y'].items():
        y_sol[(j, r)] = int(bool(sample.get(name, 0)))
    z_sol = {}
    for j, name in var_maps['z'].items():
        z_sol[j] = int(bool(sample.get(name, 0)))

    # (2) each RDC served once
    for r in data.R:
        cnt = sum(y_sol.get((j, r), 0) for j in data.T_r.get(r, []))
        if abs(cnt - 1) > tol:
            violations.append(f"RDC {r}: served {cnt}")

    # (3) demand <= supply (per CDC j)
    for j in data.C:
        # if j not in reduced CDC set, skip (no variables)
        if j not in var_maps['z']:
            continue
        demand = sum(data.d_r.get(r, 0) * y_sol.get((j, r), 0) for r in data.R)
        supply = sum(data.p_i.get(i, 0) * x_sol.get((i, j), 0) for i in data.N_c.get(j, []))
        if demand > supply + tol:
            violations.append(f"CDC {j}: demand {demand} > supply {supply}")

    # (4) supply <= q_j * z_j
    for j in data.C:
        if j not in var_maps['z']:
            continue
        supply = sum(data.p_i.get(i, 0) * x_sol.get((i, j), 0) for i in data.N_c.get(j, []))
        cap = data.q_j.get(j, 0) * z_sol.get(j, 0)
        if supply > cap + tol:
            violations.append(f"CDC {j}: supply {supply} > capacity {cap}")

    feasible = len(violations) == 0
    return feasible, violations, x_sol, y_sol, z_sol


# ---------------------------
# Main orchestration
# ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Two-Stage Logistics Network Design (TSLND) QUBO with Adaptive Penalties and PPLN Preprocessing")
    ap.add_argument("instance", help="path to TSLND instance file")
    ap.add_argument("--reads", type=int, default=8000, help="neal num_reads")
    ap.add_argument("--use-ppln", action="store_true", help="apply PPLN preprocessing before QUBO")
    ap.add_argument("--max-retries", type=int, default=4, help="adaptive penalty retries")
    ap.add_argument("--gurobi-time", type=float, default=5.0, help="run gurobi ref for this many seconds (0 disables)")
    args = ap.parse_args()

    # Parse instance
    parser = InstanceParser()
    data: InstanceData = parser.parse_file(args.instance)
    print("\nInstance Summary:")
    print(f" Factories: {len(data.I)}, Hubs: {len(data.C)}, Zones: {len(data.R)}")
    print(f" Total supply: {sum(data.p_i.values()):.2f}, Total demand: {sum(data.d_r.values()):.2f}\n")

    # Run short Gurobi reference if available
    gurobi_ref = None
    if HAS_GUROBI and args.gurobi_time > 0:
        print("Running Gurobi reference (short)...")
        gurobi_ref = gurobi_reference(data, timelimit=args.gurobi_time)
        if gurobi_ref:
            print(f" Reference Gurobi Objective: {gurobi_ref['objval']:.2f}")
        else:
            print(" Gurobi reference not available or timed out.\n")

    # Apply PPLN if requested
    CDC_list = list(data.C)
    if args.use_ppln:
        reduced = ppln_preprocess(data, verbose=True)
        CDC_list = reduced
        print(f"PPLN reduced active Hubs → {CDC_list}\n")

    # Base penalty setup
    max_bias = max(
        max(abs(v) for v in data.a_ij.values()) if data.a_ij else [1],
        max(abs(v) for v in data.b_jr.values()) if data.b_jr else [1],
        max(abs(v) for v in data.c_j.values()) if data.c_j else [1]
    )
    baseP = 10.0 * max_bias

    print(f"Initial penalty base (per constraint group): {baseP:.2e}\n")

    # Adaptive penalty loop
    best_solution = None
    for attempt in range(1, args.max_retries + 1):
        scale = 4 ** (attempt - 1)
        P1 = 3.0 * baseP * scale
        P2 = 1.0 * baseP * scale
        P3 = 1.0 * baseP * scale
        print(f"QUBO build attempt {attempt}: P1={P1:.2e}, P2={P2:.2e}, P3={P3:.2e}")

        # Build QUBO
        bqm, var_maps = build_paper_qubo_bqm(data, CDC_list, P1=P1, P2=P2, P3=P3)
        print(f" BQM linear terms: {len(bqm.linear)}, quadratic terms: {len(bqm.quadratic)}, offset: {bqm.offset:.2f}")

        # Simulated annealing (Neal)
        sampler = neal.SimulatedAnnealingSampler()
        start_time = time.time()
        sampleset = sampler.sample(bqm, num_reads=args.reads)
        solve_time = time.time() - start_time

        # Evaluate samples
        best_feasible = None
        best_obj = float("inf")
        for datum in sampleset.data():
            sample = datum.sample
            feasible, violations, x_sol, y_sol, z_sol = decode_validate(data, var_maps, sample)
            orig_obj = compute_original_objective(data, x_sol, y_sol, z_sol)
            if feasible and orig_obj < best_obj:
                best_obj = orig_obj
                best_feasible = (x_sol, y_sol, z_sol, orig_obj, datum.energy)

        if best_feasible:
            x_sol, y_sol, z_sol, obj, energy = best_feasible
            best_solution = QUBOSolution(
                x_sol, y_sol, z_sol,
                original_obj=obj, qubo_energy=energy,
                offset=bqm.offset, feasible=True,
                violations=[], solve_time=solve_time,
                reads=args.reads
            )
            print(f"  Feasible solution found at attempt {attempt}: Objective={obj:.2f}, Energy={energy:.2f}")
            break
        else:
            print(f"  No feasible solution found at attempt {attempt}, increasing penalties...")
            continue

    # If nothing feasible
    if not best_solution:
        print("No feasible QUBO solution found after all retries.")
        return

    # ----------- OUTPUT -----------
    print("\n=== QUBO RESULT (Two-Stage Logistics Network Design) ===")
    print(f" Feasible: {best_solution.feasible}")
    print(f" Original Objective (no penalties): {best_solution.original_obj:.2f}")
    print(f" QUBO Energy: {best_solution.qubo_energy:.2f}, Offset: {best_solution.offset:.2f}")
    print(f" Solve Time: {best_solution.solve_time:.2f}s (reads={best_solution.reads})")

    print("\nActivated Hubs (ω=1):", [j for j, v in best_solution.z_sol.items() if v == 1])
    print("Factory→Hub Assignments (ξ=1):")
    for (i, j), v in best_solution.x_sol.items():
        if v == 1:
            print(f"  Factory {i} → Hub {j}")
    print("Hub→Zone Assignments (ψ=1):")
    for (j, r), v in best_solution.y_sol.items():
        if v == 1:
            print(f"  Hub {j} → Zone {r}")

    # # Export
    # outfn = "tslnd_qubo_output.txt"
    # with open(outfn, "w") as f:
    #     f.write("Two-Stage Logistics Network Design (Adaptive Penalty QUBO)\n")
    #     f.write(f"Feasible: {best_solution.feasible}\n")
    #     f.write(f"Original Objective: {best_solution.original_obj:.2f}\n")
    #     f.write(f"QUBO Energy: {best_solution.qubo_energy:.2f}, Offset: {best_solution.offset:.2f}\n\n")

    #     f.write("Activated Hubs (ω=1):\n")
    #     for j, v in best_solution.z_sol.items():
    #         if v == 1:
    #             f.write(f" ω[{j}]\n")

    #     f.write("Factory→Hub (ξ=1):\n")
    #     for (i, j), v in best_solution.x_sol.items():
    #         if v == 1:
    #             f.write(f" ξ[{i},{j}]\n")

    #     f.write("Hub→Zone (ψ=1):\n")
    #     for (j, r), v in best_solution.y_sol.items():
    #         if v == 1:
    #             f.write(f" ψ[{j},{r}]\n")

    # print(f"\nSolution exported to {outfn}")



if __name__ == "__main__":
    main()
