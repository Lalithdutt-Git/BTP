import re
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

@dataclass
class InstanceData:
    """Stores parsed instance data for the ILP problem"""
    I: List[int] = field(default_factory=list)  # Production plants
    C: List[int] = field(default_factory=list)  # Potential distribution centers
    R: List[int] = field(default_factory=list)  # Regional distribution centers
    
    a_ij: Dict[Tuple[int, int], float] = field(default_factory=dict)  # Transport cost I to C
    b_jr: Dict[Tuple[int, int], float] = field(default_factory=dict)  # Transport cost C to R
    
    p_i: Dict[int, float] = field(default_factory=dict)  # Production capacity
    q_j: Dict[int, float] = field(default_factory=dict)  # Supply capacity
    c_j: Dict[int, float] = field(default_factory=dict)  # Activation cost
    d_r: Dict[int, float] = field(default_factory=dict)  # Demand
    
    N_c: Dict[int, List[int]] = field(default_factory=dict)  # Plants that can serve CDC j
    T_r: Dict[int, List[int]] = field(default_factory=dict)  # CDCs that can serve RDC r
    
    def summary(self) -> str:
        """Returns a formatted summary of the instance data"""
        summary = []
        summary.append(f"Production Plants (I): {len(self.I)} plants")
        summary.append(f"Distribution Centers (C): {len(self.C)} centers")
        summary.append(f"Regional Centers (R): {len(self.R)} centers")
        summary.append(f"\nTransportation costs (a_ij): {len(self.a_ij)} entries")
        summary.append(f"Transportation costs (b_jr): {len(self.b_jr)} entries")
        summary.append(f"\nTotal production capacity: {sum(self.p_i.values()):.2f}")
        summary.append(f"Total demand: {sum(self.d_r.values()):.2f}")
        summary.append(f"Total activation cost: {sum(self.c_j.values()):.2f}")
        return "\n".join(summary)


class InstanceParser:
    """Robust parser for instance data files"""
    
    def __init__(self):
        self.data = InstanceData()
        
    def parse_list(self, line: str) -> List[int]:
        """Parse a list in format [0, 1, 2] or [0, 1]"""
        match = re.search(r'\[(.*?)\]', line)
        if match:
            content = match.group(1).strip()
            if content:
                return [int(x.strip()) for x in content.split(',')]
        return []
    
    def parse_matrix_line(self, line: str) -> Tuple[int, int, float]:
        """Parse a line like '0 1 72' into (i, j, value)"""
        parts = line.strip().split()
        if len(parts) == 3:
            return int(parts[0]), int(parts[1]), float(parts[2])
        raise ValueError(f"Invalid matrix line format: {line}")
    
    def parse_dict_line(self, line: str, section_name: str = None) -> List[Tuple[int, float]]:
        """Parse a line like '0 320' or '80 89' into [(key, value), ...]
        
        For sections like p_i, if we see multiple values on one line (e.g., "80 89"),
        we treat them as sequential values without explicit indices.
        For sections like q_j, c_j, d_r that have explicit indices, we parse as key-value pairs.
        """
        parts = line.strip().split()
        results = []
        
        if len(parts) == 0:
            return results
        
        # If only one value, treat as (0, value)
        if len(parts) == 1:
            try:
                results.append((0, float(parts[0])))
            except ValueError:
                pass
            return results
        
        # For p_i section specifically, check if this looks like space-separated values
        # without indices (e.g., "80 89" instead of "0 80" and "1 89")
        if section_name == 'p_i':
            # Try to parse all parts as floats (no indices)
            try:
                values = [float(p) for p in parts]
                # If all parse successfully, assign sequential indices
                results = [(idx, val) for idx, val in enumerate(values)]
                return results
            except ValueError:
                # If parsing as all floats fails, fall through to key-value parsing
                pass
        
        # Default: parse as key-value pairs (0 320, 1 393, etc.)
        if len(parts) == 2:
            try:
                results.append((int(parts[0]), float(parts[1])))
            except ValueError:
                # If key-value parsing fails, try as sequential values
                try:
                    results = [(idx, float(p)) for idx, p in enumerate(parts)]
                except ValueError:
                    pass
        
        return results
    
    def parse_mapping_line(self, line: str) -> Tuple[int, List[int]]:
        """Parse a line like '0: [0, 1]' into (key, [values])"""
        if ':' in line:
            key_part, value_part = line.split(':', 1)
            key = int(key_part.strip())
            values = self.parse_list(value_part)
            return key, values
        raise ValueError(f"Invalid mapping line format: {line}")
    
    def parse_file(self, filepath: str) -> InstanceData:
        """Parse an instance data file"""
        with open(filepath, 'r') as f:
            content = f.read()
        return self.parse_content(content)
    
    def parse_content(self, content: str) -> InstanceData:
        """Parse instance data from string content"""
        lines = content.strip().split('\n')
        current_section = None
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and header
            if not line or line.startswith('###'):
                continue
            
            # Detect section headers
            if line.startswith('I (') or line.startswith('I:'):
                current_section = 'I'
                continue
            elif line.startswith('C (') or line.startswith('C:'):
                current_section = 'C'
                continue
            elif line.startswith('R (') or line.startswith('R:'):
                current_section = 'R'
                continue
            elif line.startswith('a_ij'):
                current_section = 'a_ij'
                continue
            elif line.startswith('b_jr'):
                current_section = 'b_jr'
                continue
            elif line.startswith('p_i'):
                current_section = 'p_i'
                continue
            elif line.startswith('q_j'):
                current_section = 'q_j'
                continue
            elif line.startswith('c_j'):
                current_section = 'c_j'
                continue
            elif line.startswith('d_r'):
                current_section = 'd_r'
                continue
            elif line.startswith('N_c'):
                current_section = 'N_c'
                continue
            elif line.startswith('T_r'):
                current_section = 'T_r'
                continue
            
            # Parse data based on current section
            try:
                if current_section == 'I':
                    self.data.I = self.parse_list(line)
                    
                elif current_section == 'C':
                    self.data.C = self.parse_list(line)
                    
                elif current_section == 'R':
                    self.data.R = self.parse_list(line)
                    
                elif current_section == 'a_ij':
                    i, j, cost = self.parse_matrix_line(line)
                    self.data.a_ij[(i, j)] = cost
                    
                elif current_section == 'b_jr':
                    j, r, cost = self.parse_matrix_line(line)
                    self.data.b_jr[(j, r)] = cost
                    
                elif current_section == 'p_i':
                    entries = self.parse_dict_line(line, 'p_i')
                    for i, capacity in entries:
                        self.data.p_i[i] = capacity
                    
                elif current_section == 'q_j':
                    entries = self.parse_dict_line(line, 'q_j')
                    for j, capacity in entries:
                        self.data.q_j[j] = capacity
                    
                elif current_section == 'c_j':
                    entries = self.parse_dict_line(line, 'c_j')
                    for j, cost in entries:
                        self.data.c_j[j] = cost
                    
                elif current_section == 'd_r':
                    entries = self.parse_dict_line(line, 'd_r')
                    for r, demand in entries:
                        self.data.d_r[r] = demand
                    
                elif current_section == 'N_c':
                    j, plants = self.parse_mapping_line(line)
                    self.data.N_c[j] = plants
                    
                elif current_section == 'T_r':
                    r, centers = self.parse_mapping_line(line)
                    self.data.T_r[r] = centers
                    
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse line in section {current_section}: {line}")
                print(f"Error: {e}")
                continue
        
        return self.data
    
    def validate_data(self) -> List[str]:
        """Validate the parsed data and return list of issues"""
        issues = []
        
        # Check if basic sets are populated
        if not self.data.I:
            issues.append("No production plants (I) found")
        if not self.data.C:
            issues.append("No distribution centers (C) found")
        if not self.data.R:
            issues.append("No regional centers (R) found")
        
        # Check cost matrices
        expected_a_ij = len(self.data.I) * len(self.data.C)
        if len(self.data.a_ij) != expected_a_ij:
            issues.append(f"Expected {expected_a_ij} a_ij entries, found {len(self.data.a_ij)}")
        
        expected_b_jr = len(self.data.C) * len(self.data.R)
        if len(self.data.b_jr) < expected_b_jr:  # Less than because of T_r constraints
            issues.append(f"Warning: Found {len(self.data.b_jr)} b_jr entries (max {expected_b_jr})")
        
        # Check capacities and demands
        if len(self.data.p_i) != len(self.data.I):
            issues.append(f"Expected {len(self.data.I)} p_i entries, found {len(self.data.p_i)}")
        if len(self.data.d_r) != len(self.data.R):
            issues.append(f"Expected {len(self.data.R)} d_r entries, found {len(self.data.d_r)}")
        
        # Check if supply can meet demand
        total_supply = sum(self.data.p_i.values())
        total_demand = sum(self.data.d_r.values())
        if total_supply < total_demand:
            issues.append(f"Total supply ({total_supply}) < Total demand ({total_demand})")
        
        return issues

