# Gurobi MILP for Two-Level Facility Location Problem
# Using the test instance instance_2I_4C_6R.txt

import gurobipy as gp
from gurobipy import GRB

# ----------------------------
# Sets
# ----------------------------
I = [0, 1]                      # production plants
C = [0, 1, 2, 3]                # potential CDCs
R = [0, 1, 2, 3, 4, 5]          # regional centers

# ----------------------------
# Parameters
# ----------------------------
# Transportation cost from plants to CDCs (a_ij)
a = {
    (0,0):89, (0,1):72, (0,2):87, (0,3):53,
    (1,0):56, (1,1):35, (1,2):49, (1,3):29
}

# Transportation cost from CDCs to RDCs (b_jr)
b = {
    (0,0):69, (0,1):51, (0,2):33, (0,3):99, (0,4):5,  (0,5):151,
    (1,0):62, (1,1):34, (1,2):32, (1,3):33, (1,4):60, (1,5):85,
    (2,0):76, (2,1):48, (2,2):34, (2,3):82, (2,4):12, (2,5):134,
    (3,0):42, (3,1):34, (3,2):52, (3,3):13, (3,4):80, (3,5):65
}

# Production capacity (p_i)
p = {0:80, 1:89}

# CDC capacity (q_j)
q = {0:320, 1:393, 2:344, 3:235}

# Activation cost of CDCs (c_j)
c = {0:1160, 1:1196, 2:1172, 3:1117}

# RDC demand (d_r)
d = {0:35, 1:7, 2:24, 3:26, 4:38, 5:35}

# Allowed plant-to-CDC links
N_c = {0:[0,1], 1:[0,1], 2:[0,1], 3:[0,1]}

# Allowed CDC-to-RDC links
T_r = {0:[0,1,2,3], 1:[0,1,2,3], 2:[0,1,2,3],
       3:[0,1,2,3], 4:[0,1,2,3], 5:[1,3]}

# ----------------------------
# Model
# ----------------------------
m = gp.Model("TLFLP")

# Decision variables
x = m.addVars([(i,j) for j in C for i in N_c[j]], vtype=GRB.BINARY, name="x")
y = m.addVars([(j,r) for r in R for j in T_r[r]], vtype=GRB.BINARY, name="y")
z = m.addVars(C, vtype=GRB.BINARY, name="z")

# Objective
m.setObjective(
    gp.quicksum(a[i,j]*x[i,j] for (i,j) in x.keys()) +
    gp.quicksum(b[j,r]*y[j,r] for (j,r) in y.keys()) +
    gp.quicksum(c[j]*z[j] for j in C),
    GRB.MINIMIZE
)

# ----------------------------
# Constraints
# ----------------------------

# Each RDC served by exactly one CDC
for r in R:
    m.addConstr(gp.quicksum(y[j,r] for j in T_r[r]) == 1, name=f"RDC_assign_{r}")

# Demand assigned to CDC ≤ supply sent to CDC
for j in C:
    lhs = gp.quicksum(d[r]*y[j,r] for r in R if (j,r) in y)
    rhs = gp.quicksum(p[i]*x[i,j] for i in I if (i,j) in x)
    m.addConstr(lhs <= rhs, name=f"demand_supply_{j}")

# Supply sent to CDC ≤ capacity × activation
for j in C:
    lhs = gp.quicksum(p[i]*x[i,j] for i in I if (i,j) in x)
    m.addConstr(lhs <= q[j]*z[j], name=f"capacity_{j}")

# Optimize
m.optimize()

# ----------------------------
# Print results
# ----------------------------
if m.status == GRB.OPTIMAL:
    print("\nOptimal Objective Value:", m.objVal)
    print("\nActivated CDCs:")
    for j in C:
        if z[j].X > 0.5:
            print(f"  CDC {j} activated")
    print("\nPlant-to-CDC Assignments:")
    for (i,j) in x:
        if x[i,j].X > 0.5:
            print(f"  Plant {i} -> CDC {j}")
    print("\nCDC-to-RDC Assignments:")
    for (j,r) in y:
        if y[j,r].X > 0.5:
            print(f"  CDC {j} -> RDC {r}")
else:
    print("No optimal solution found. Status:", m.status)
