import gurobipy as gp
from gurobipy import GRB
from parser import InstanceParser, InstanceData
from typing import Dict, Tuple, Optional
import argparse
from dataclasses import dataclass
import time

@dataclass
class Solution:
    """Stores the solution to the TLFLP problem"""
    objective_value: float
    solve_time: float
    status: str
    gap: float
    
    # Decision variables
    x_ij: Dict[Tuple[int, int], int]  # PF i serves CDC j
    y_jr: Dict[Tuple[int, int], int]  # CDC j serves RDC r
    z_j: Dict[int, int]  # CDC j is activated
    
    # Solution statistics
    num_activated_cdcs: int
    activated_cdcs: list
    
    def __str__(self) -> str:
        """Returns a formatted summary of the solution"""
        lines = []
        lines.append("=" * 60)
        lines.append("SOLUTION SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Status: {self.status}")
        lines.append(f"Objective Value: {self.objective_value:.2f}")
        lines.append(f"Solution Time: {self.solve_time:.2f} seconds")
        lines.append(f"Optimality Gap: {self.gap:.4f}%")
        lines.append(f"\nNumber of Activated HUBSs: {self.num_activated_cdcs}")
        lines.append(f"Activated HUBS IDs: {self.activated_cdcs}")
        
        lines.append("\n" + "-" * 60)
        lines.append("FACTORIES TO HUB ASSIGNMENTS:")
        lines.append("-" * 60)
        for (i, j), value in sorted(self.x_ij.items()):
            if value > 0.5:  # Binary variable
                lines.append(f"  FACTORY {i} -> HUB {j}")
        
        lines.append("\n" + "-" * 60)
        lines.append("HUB TO ZONE ASSIGNMENTS:")
        lines.append("-" * 60)
        for (j, r), value in sorted(self.y_jr.items()):
            if value > 0.5:  # Binary variable
                lines.append(f"  HUB {j} -> ZONES {r}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class TLFLPSolver:
    """
    Solver for the Two-Level Facility Location Problem using Gurobi.
    
    Mathematical formulation from the paper:
    
    min ∑∑ a_ij * x_ij + ∑∑ b_jr * y_jr + ∑ c_j * z_j
    
    s.t.
    (2) ∑_{j∈T_r} y_jr = 1                    ∀r ∈ R
    (3) ∑_{r∈R} d_r * y_jr ≤ ∑_{i∈N_j} p_i * x_ij    ∀j ∈ C
    (4) ∑_{i∈N_j} p_i * x_ij ≤ q_j * z_j      ∀j ∈ C
    (5) x_ij, y_jr, z_j ∈ {0,1}               ∀i,j,r
    """
    
    def __init__(self, data: InstanceData, time_limit: Optional[float] = None,
                 mip_gap: float = 1e-4, verbose: bool = True):
        """
        Initialize the TLFLP solver.
        
        Args:
            data: Instance data parsed from file
            time_limit: Maximum time in seconds (None for no limit)
            mip_gap: MIP optimality gap tolerance
            verbose: Whether to print Gurobi output
        """
        self.data = data
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.verbose = verbose
        self.model = None
        self.solution = None
        
    def build_model(self) -> gp.Model:
        """Build the Gurobi optimization model"""
        
        # Create model
        model = gp.Model("TLFLP")
        
        # Set parameters
        if not self.verbose:
            model.setParam('OutputFlag', 0)
        if self.time_limit:
            model.setParam('TimeLimit', self.time_limit)
        model.setParam('MIPGap', self.mip_gap)
        
        # Decision variables
        # x_ij: Binary variable, 1 if PF i serves CDC j
        x = {}
        for j in self.data.C:
            for i in self.data.N_c.get(j, []):
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        
        # y_jr: Binary variable, 1 if CDC j serves RDC r
        y = {}
        for r in self.data.R:
            for j in self.data.T_r.get(r, []):
                y[j, r] = model.addVar(vtype=GRB.BINARY, name=f"y_{j}_{r}")
        
        # z_j: Binary variable, 1 if CDC j is activated
        z = {}
        for j in self.data.C:
            z[j] = model.addVar(vtype=GRB.BINARY, name=f"z_{j}")
        
        model.update()
        
        # Objective function (1)
        # min ∑∑ a_ij * x_ij + ∑∑ b_jr * y_jr + ∑ c_j * z_j
        obj_expr = gp.LinExpr()
        
        # First term: transportation cost from plants to CDCs
        for (i, j), cost in self.data.a_ij.items():
            if (i, j) in x:
                obj_expr += cost * x[i, j]
        
        # Second term: transportation cost from CDCs to RDCs
        for (j, r), cost in self.data.b_jr.items():
            if (j, r) in y:
                obj_expr += cost * y[j, r]
        
        # Third term: activation cost of CDCs
        for j in self.data.C:
            obj_expr += self.data.c_j[j] * z[j]
        
        model.setObjective(obj_expr, GRB.MINIMIZE)
        
        # Constraints (2): Each RDC must be served by exactly one CDC
        # ∑_{j∈T_r} y_jr = 1  ∀r ∈ R
        for r in self.data.R:
            expr = gp.LinExpr()
            for j in self.data.T_r.get(r, []):
                if (j, r) in y:
                    expr += y[j, r]
            model.addConstr(expr == 1, name=f"rdc_coverage_{r}")
        
        # Constraints (3): Flow conservation at CDCs
        # ∑_{r∈R} d_r * y_jr ≤ ∑_{i∈N_j} p_i * x_ij  ∀j ∈ C
        for j in self.data.C:
            lhs = gp.LinExpr()  # Demand served by CDC j
            rhs = gp.LinExpr()  # Supply received by CDC j
            
            # Left side: total demand from RDCs served by CDC j
            for r in self.data.R:
                if (j, r) in y:
                    lhs += self.data.d_r[r] * y[j, r]
            
            # Right side: total supply from plants to CDC j
            for i in self.data.N_c.get(j, []):
                if (i, j) in x:
                    rhs += self.data.p_i[i] * x[i, j]
            
            model.addConstr(lhs <= rhs, name=f"flow_balance_{j}")
        
        # Constraints (4): Capacity constraints at CDCs
        # ∑_{i∈N_j} p_i * x_ij ≤ q_j * z_j  ∀j ∈ C
        for j in self.data.C:
            expr = gp.LinExpr()
            
            # Total supply to CDC j
            for i in self.data.N_c.get(j, []):
                if (i, j) in x:
                    expr += self.data.p_i[i] * x[i, j]
            
            # Must not exceed capacity if activated
            model.addConstr(expr <= self.data.q_j[j] * z[j], 
                          name=f"capacity_{j}")
        
        self.model = model
        self.x_vars = x
        self.y_vars = y
        self.z_vars = z
        
        return model
    
    def solve(self) -> Solution:
        """
        Solve the TLFLP problem.
        
        Returns:
            Solution object containing optimal values and statistics
        """
        if self.model is None:
            self.build_model()
        
        # Solve the model
        start_time = time.time()
        self.model.optimize()
        solve_time = time.time() - start_time
        
        # Extract solution
        status_map = {
            GRB.OPTIMAL: "Optimal",
            GRB.INFEASIBLE: "Infeasible",
            GRB.UNBOUNDED: "Unbounded",
            GRB.INF_OR_UNBD: "Infeasible or Unbounded",
            GRB.TIME_LIMIT: "Time Limit Reached",
            GRB.INTERRUPTED: "Interrupted",
            GRB.SUBOPTIMAL: "Suboptimal"
        }
        
        status = status_map.get(self.model.status, f"Unknown ({self.model.status})")
        
        if self.model.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT]:
            # Extract variable values
            x_sol = {k: int(round(v.X)) for k, v in self.x_vars.items()}
            y_sol = {k: int(round(v.X)) for k, v in self.y_vars.items()}
            z_sol = {k: int(round(v.X)) for k, v in self.z_vars.items()}
            
            # Calculate statistics
            activated_cdcs = [j for j in self.data.C if z_sol[j] > 0.5]
            num_activated = len(activated_cdcs)
            
            # Get optimality gap
            try:
                gap = self.model.MIPGap * 100  # Convert to percentage
            except:
                gap = 0.0
            
            self.solution = Solution(
                objective_value=self.model.ObjVal,
                solve_time=solve_time,
                status=status,
                gap=gap,
                x_ij=x_sol,
                y_jr=y_sol,
                z_j=z_sol,
                num_activated_cdcs=num_activated,
                activated_cdcs=activated_cdcs
            )
        else:
            # No feasible solution found
            self.solution = Solution(
                objective_value=float('inf'),
                solve_time=solve_time,
                status=status,
                gap=100.0,
                x_ij={},
                y_jr={},
                z_j={},
                num_activated_cdcs=0,
                activated_cdcs=[]
            )
        
        return self.solution
    
    def validate_solution(self) -> Tuple[bool, list]:
        """
        Validate the current solution against all constraints.
        
        Returns:
            (is_valid, list_of_violations)
        """
        if self.solution is None:
            return False, ["No solution available"]
        
        violations = []
        
        # Check constraint (2): Each RDC served exactly once
        for r in self.data.R:
            count = sum(self.solution.y_jr.get((j, r), 0) 
                       for j in self.data.T_r.get(r, []))
            if abs(count - 1) > 0.01:
                violations.append(f"zone {r} served {count} times (should be 1)")
        
        # Check constraint (3): Flow balance
        for j in self.data.C:
            demand = sum(self.data.d_r[r] * self.solution.y_jr.get((j, r), 0) 
                        for r in self.data.R)
            supply = sum(self.data.p_i[i] * self.solution.x_ij.get((i, j), 0) 
                        for i in self.data.N_c.get(j, []))
            
            if demand > supply + 0.01:
                violations.append(
                    f"hub {j}: demand ({demand:.2f}) > supply ({supply:.2f})")
        
        # Check constraint (4): Capacity constraints
        for j in self.data.C:
            supply = sum(self.data.p_i[i] * self.solution.x_ij.get((i, j), 0) 
                        for i in self.data.N_c.get(j, []))
            capacity = self.data.q_j[j] * self.solution.z_j[j]
            
            if supply > capacity + 0.01:
                violations.append(
                    f"hub {j}: supply ({supply:.2f}) > capacity ({capacity:.2f})")
        
        return len(violations) == 0, violations
    
    def print_detailed_solution(self):
        """Print detailed solution information"""
        if self.solution is None:
            print("No solution available. Run solve() first.")
            return
        
        print(self.solution)
        
        # Validation
        is_valid, violations = self.validate_solution()
        print("\nSOLUTION VALIDATION:")
        print("-" * 60)
        if is_valid:
            print("All constraints satisfied")
        else:
            print("Constraint violations detected:")
            for v in violations:
                print(f"  - {v}")


def main():
    """Example usage of the TLFLP solver"""
    # import sys
    
    # # Get instance file from command line or use default
    # if len(sys.argv) > 1:
    #     instance_file = sys.argv[1]
    # else:
    #     instance_file = 'instance_2I_4C_6R.txt'
    
    # Parse instance file
    parser = InstanceParser()
    ap = argparse.ArgumentParser(description="Solve TLFLP instance using Gurobi (robust parser).")
    ap.add_argument("instance", help="Path to TLFLP instance file")
    args = ap.parse_args()
    data = parser.parse_file(args.instance)
    
    # print("Instance Summary:")
    # print(data.summary())
    # print("\n")
    
    # Validate parsed data
    # issues = parser.validate_data()
    # if issues:
    #     print("Data validation issues:")
    #     for issue in issues:
    #         print(f"  - {issue}")
    #     print()
    
    # Create and solve the problem
    solver = TLFLPSolver(data, time_limit=300, verbose=True)
    
    print("Building and solving MILP model...")
    print("=" * 60)
    
    solution = solver.solve()
    
    # Print results
    solver.print_detailed_solution()
    
    # Export solution to file (with UTF-8 encoding to handle any special characters)
    try:
        with open('solution_output.txt', 'w', encoding='utf-8') as f:
            f.write(str(solution))
            f.write("\n\n")
            f.write("Instance Summary:\n")
            f.write(data.summary())
        
        print("\nSolution exported to 'solution_output.txt'")
    except Exception as e:
        print(f"\nWarning: Could not export solution to file: {e}")
        # Try ASCII-only fallback
        try:
            with open('solution_output.txt', 'w', encoding='ascii', errors='replace') as f:
                f.write(str(solution))
                f.write("\n\n")
                f.write("Instance Summary:\n")
                f.write(data.summary())
            print("Solution exported with ASCII encoding to 'solution_output.txt'")
        except Exception as e2:
            print(f"Failed to export solution: {e2}")


if __name__ == "__main__":
    main()
