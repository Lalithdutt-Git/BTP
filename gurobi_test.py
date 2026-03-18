#!/usr/bin/env python3
# gurobi_test.py
# TLFLP solved using Gurobi with robust parser integration

import argparse
import gurobipy as gp
from gurobipy import GRB

from parser import InstanceParser


# ----------------------------
# Build + Solve Gurobi MILP
# ----------------------------
def build_and_solve(filename):
    print(f"\nLoading instance: {filename}")

    # ----------------------------
    # Use robust parser
    # ---------------------------- 
    parser = InstanceParser()
    data_obj = parser.parse_file(filename)

    # Validation (important for debugging)
    issues = parser.validate_data()
    if issues:
        print("\n Data Issues Found:")
        for issue in issues:
            print(" -", issue)

    print("\n Instance Summary:")
    print(data_obj.summary())

    # Extract data
    I = data_obj.I
    C = data_obj.C
    R = data_obj.R

    a = data_obj.a_ij
    b = data_obj.b_jr
    p = data_obj.p_i
    q = data_obj.q_j
    c = data_obj.c_j
    d = data_obj.d_r

    N_c = data_obj.N_c
    T_r = data_obj.T_r

    print(f"\nParsed: |I|={len(I)}, |C|={len(C)}, |R|={len(R)}")

    # ----------------------------
    # Validation (strict)
    # ----------------------------
    missing_p = [i for i in I if i not in p]
    if missing_p:
        raise ValueError(f"Missing plant capacities for plants: {missing_p}")

    missing_q = [j for j in C if j not in q]
    if missing_q:
        raise ValueError(f"Missing DC capacities for DCs: {missing_q}")

    missing_d = [r for r in R if r not in d]
    if missing_d:
        raise ValueError(f"Missing demand entries for regions: {missing_d}")

    # ----------------------------
    # Build Model
    # ----------------------------
    m = gp.Model("TLFLP")
    m.setParam("OutputFlag", 1)

    # Variable keys (respect feasibility)
    x_keys = [(i, j) for j in C for i in N_c.get(j, I)]
    y_keys = [(j, r) for r in R for j in T_r.get(r, C)]

    # Variables
    x = m.addVars(x_keys, vtype=GRB.BINARY, name="x")
    y = m.addVars(y_keys, vtype=GRB.BINARY, name="y")
    z = m.addVars(C, vtype=GRB.BINARY, name="z")

    # ----------------------------
    # Objective
    # ----------------------------
    m.setObjective(
        gp.quicksum(a.get((i, j), 0) * x[i, j] for (i, j) in x_keys)
        + gp.quicksum(b.get((j, r), 0) * y[j, r] for (j, r) in y_keys)
        + gp.quicksum(c[j] * z[j] for j in C),
        GRB.MINIMIZE
    )

    # ----------------------------
    # Constraints
    # ----------------------------

    # (1) Each RDC served by exactly one CDC
    for r in R:
        eligible = [j for j in T_r.get(r, C) if (j, r) in y_keys]
        if not eligible:
            eligible = [j for j in C if (j, r) in y_keys]

        m.addConstr(
            gp.quicksum(y[j, r] for j in eligible) == 1,
            name=f"assign_r{r}"
        )

    # (2) Demand assigned <= supply sent
    for j in C:
        lhs = gp.quicksum(d[r] * y[j, r] for r in R if (j, r) in y_keys)
        rhs = gp.quicksum(p[i] * x[i, j] for i in I if (i, j) in x_keys)

        m.addConstr(lhs <= rhs, name=f"demand_supply_{j}")

    # (3) Supply sent <= capacity * activation
    for j in C:
        sent = gp.quicksum(p[i] * x[i, j] for i in I if (i, j) in x_keys)

        m.addConstr(sent <= q[j] * z[j], name=f"capacity_{j}")

    # ----------------------------
    # Solve
    # ----------------------------
    m.optimize()

    # ----------------------------
    # Results
    # ----------------------------
    if m.status == GRB.OPTIMAL:
        print(f"\n Optimal objective value: {m.objVal:.2f}\n")

        print("Activated CDCs:")
        for j in C:
            if z[j].X > 0.5:
                print(f"  CDC {j}")

        print("\nPlant -> CDC assignments:")
        for (i, j) in x_keys:
            if x[i, j].X > 0.5:
                print(f"  Plant {i} -> CDC {j}")

        print("\nCDC -> RDC assignments:")
        for (j, r) in y_keys:
            if y[j, r].X > 0.5:
                print(f"  CDC {j} -> RDC {r}")

    elif m.status == GRB.INFEASIBLE:
        print("\n Model is infeasible. Computing IIS...")
        m.computeIIS()
        m.write("model.ilp")
        print("IIS written to model.ilp")

    else:
        print("\n No optimal solution found. Status:", m.status)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Solve TLFLP instance using Gurobi")
    ap.add_argument("instance", help="Path to instance file")
    args = ap.parse_args()

    build_and_solve(args.instance)