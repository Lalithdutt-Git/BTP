# tlflp_qubo_paper_optimized.py
# Paper-exact QUBO for TLFLP with adaptive scaling for neal

from pyqubo import Array, Placeholder
from neal import SimulatedAnnealingSampler
import math, argparse, numpy as np

# ---------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------
def parse_instance(path):
    data = {'I': [], 'C': [], 'R': [],
            'a_ij': {}, 'b_jr': {},
            'p_i': {}, 'q_j': {}, 'c_j': {}, 'd_r': {},
            'N_c': {}, 'T_r': {}}
    section = None
    with open(path, 'r') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("###"):
                continue
            if line.startswith("I "): section="I"; continue
            if line.startswith("C "): section="C"; continue
            if line.startswith("R "): section="R"; continue
            if line.startswith("a_ij"): section="a_ij"; continue
            if line.startswith("b_jr"): section="b_jr"; continue
            if line.startswith("p_i"): section="p_i"; continue
            if line.startswith("q_j"): section="q_j"; continue
            if line.startswith("c_j"): section="c_j"; continue
            if line.startswith("d_r"): section="d_r"; continue
            if line.startswith("N_c"): section="N_c"; continue
            if line.startswith("T_r"): section="T_r"; continue

            if section in ["I", "C", "R"] and line.startswith("["):
                data[section] = [int(v.strip()) for v in line.strip("[]").split(",") if v.strip()]
            elif section == "a_ij":
                i, j, c = map(int, line.split()); data["a_ij"][(i,j)] = c
            elif section == "b_jr":
                j, r, c = map(int, line.split()); data["b_jr"][(j,r)] = c
            elif section in ["p_i", "q_j", "c_j", "d_r"]:
                vals = [int(v) for v in line.split()]
                for idx,v in enumerate(vals):
                    data[section][idx] = v
            elif section in ["N_c","T_r"] and ":" in line:
                key, arr = line.split(":"); key=int(key.strip())
                arr = arr.replace("[", "").replace("]", "").strip()
                vals = [int(v.strip()) for v in arr.split(",") if v.strip().isdigit()]

                data[section][key]=vals

    for g in ["p_i","q_j","c_j","d_r"]:
        if data[g]:
            vals=list(data[g].values())
            data[g]={i:vals[i] for i in range(len(vals))}
    if not data["I"]: data["I"]=list(data["p_i"].keys())
    if not data["C"]: data["C"]=list(data["q_j"].keys())
    if not data["R"]: data["R"]=list(data["d_r"].keys())

    print(f"Plants: {len(data['I'])}, DCs: {len(data['C'])}, RCs: {len(data['R'])}")
    return data

# ---------------------------------------------------------------------
# Slack Bounds
# ---------------------------------------------------------------------
def estimate_slack_bounds(data):
    total_demand=sum(data["d_r"].values())
    total_supply=sum(data["p_i"].values())
    U1,U2={},{}
    for j in sorted(data["q_j"].keys()):
        U1[j]=total_demand
        U2[j]=max(total_supply, data["q_j"][j])
    return U1,U2

# ---------------------------------------------------------------------
# Build Paper QUBO with scaling
# ---------------------------------------------------------------------
def build_paper_qubo(data, P_scale=None):
    I,C,R=sorted(data["p_i"].keys()),sorted(data["q_j"].keys()),sorted(data["d_r"].keys())
    x=Array.create("x",(len(I),len(C)),"BINARY")
    y=Array.create("y",(len(C),len(R)),"BINARY")
    z=Array.create("z",len(C),"BINARY")

    U1,U2=estimate_slack_bounds(data)
    s1,s2,k1,k2={}, {}, {}, {}
    for j_idx,j in enumerate(C):
        k1[j_idx]=max(1, math.ceil(math.log2(U1[j]+1)))
        k2[j_idx]=max(1, math.ceil(math.log2(U2[j]+1)))
        s1[j_idx]=Array.create(f"s1_{j_idx}",(k1[j_idx]),"BINARY")
        s2[j_idx]=Array.create(f"s2_{j_idx}",(k2[j_idx]),"BINARY")

    # --- Objective normalization ---
    all_costs = list(data["a_ij"].values()) + list(data["b_jr"].values()) + list(data["c_j"].values())
    norm_factor = max(abs(max(all_costs, default=1)),1)
    cost = 0
    for i_idx,i in enumerate(I):
        for j_idx,j in enumerate(C):
            cost += (data["a_ij"].get((i,j),0)/norm_factor)*x[i_idx][j_idx]
    for j_idx,j in enumerate(C):
        for r_idx,r in enumerate(R):
            cost += (data["b_jr"].get((j,r),0)/norm_factor)*y[j_idx][r_idx]
    for j_idx,j in enumerate(C):
        cost += (data["c_j"][j]/norm_factor)*z[j_idx]

    # --- Penalties ---
    P1,P2,P3=Placeholder("P1"),Placeholder("P2"),Placeholder("P3")

    # Eq (13)
    pen1=0
    for r_idx,r in enumerate(R):
        T_r=data["T_r"].get(r,C)
        indices=[C.index(j) for j in T_r if j in C]
        pen1+=(sum(y[j_idx][r_idx] for j_idx in indices)-1)**2

    # Eq (14)
    pen2=0
    for j_idx,j in enumerate(C):
        N_j=data["N_c"].get(j,I)
        expr_dy=sum(data["d_r"][r]*y[j_idx][r_idx] for r_idx,r in enumerate(R))
        expr_px=sum(data["p_i"][i]*x[i_idx][j_idx] for i_idx,i in enumerate(I) if i in N_j)
        S1=sum((2**b)*s1[j_idx][b] for b in range(k1[j_idx]))
        pen2+=(expr_dy-expr_px+S1)**2

    # Eq (15)
    pen3=0
    for j_idx,j in enumerate(C):
        N_j=data["N_c"].get(j,I)
        expr_px=sum(data["p_i"][i]*x[i_idx][j_idx] for i_idx,i in enumerate(I) if i in N_j)
        S2=sum((2**b)*s2[j_idx][b] for b in range(k2[j_idx]))
        pen3+=(expr_px-data["q_j"][j]*z[j_idx]+S2)**2

    H = cost + P1*pen1 + P2*pen2 + P3*pen3
    model=H.compile()

    baseP = P_scale if P_scale else 5 * np.sqrt(norm_factor)
    feed={"P1": baseP, "P2": baseP, "P3": baseP}
    return model,feed,norm_factor

# ---------------------------------------------------------------------
def solve_locally(model, feed, num_reads=2000):
    qubo, offset = model.to_qubo(feed_dict=feed)
    sampler=SimulatedAnnealingSampler()
    resp=sampler.sample_qubo(qubo, num_reads=num_reads, beta_range=[0.1,10.0])
    best=resp.first.sample
    decoded=model.decode_sample(best,vartype="BINARY",feed_dict=feed)
    return decoded

# ---------------------------------------------------------------------
def print_solution(decoded):
    print(f"\nEnergy: {decoded.energy:.3f}")
    if decoded.constraints(only_broken=True):
        print("⚠️ Broken constraints:", decoded.constraints(only_broken=True))
    else:
        print("✅ All constraints satisfied")
    s=decoded.sample
    print("\nActivated CDCs (z=1):")
    for k,v in s.items():
        if k.startswith("z[") and v==1: print(" ",k)

    print("\nAssignments:")
    for k,v in s.items():
        if v==1 and (k.startswith("x[") or k.startswith("y[")):
            print(" ",k)

# ---------------------------------------------------------------------
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("instance")
    ap.add_argument("--reads",type=int,default=2000)
    args=ap.parse_args()
    data=parse_instance(args.instance)
    model,feed,nf=build_paper_qubo(data)
    print("Solving (neal, scaled QUBO)...")
    decoded=solve_locally(model,feed,num_reads=args.reads)
    print_solution(decoded)
