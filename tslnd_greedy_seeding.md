
The Population SA Sampler with Greedy Seeding
---------------------------------------------

The PopulationSASampler is the fallback sampler used when neal/dimod are unavailable. It replaces the single random-start SA chain of a standard solver with a structured multi-chain architecture. The greedy seed is the cornerstone of this architecture.

### What Problem the Seed Solves

A standard SA sampler initialises each chain from a uniformly random binary vector. For a TLFLP instance, the vast majority of random binary assignments are deeply infeasible — they violate RDC coverage, capacity, and supply constraints simultaneously. An SA chain that starts in this region of the search space must spend most of its sweep budget climbing out of infeasibility before it can even begin exploring the feasible region. With a limited sweep budget, many chains never reach feasibility at all.

The greedy seed constructs a binary starting point that is structurally close to feasibility from the outset, so the SA chain starting from it begins its temperature descent already near the feasible region.

### How the Greedy Seed is Constructed

The seed is built in three sequential steps, each operating on a different subset of the binary decision variables.

**Step 1 — RDC-to-CDC assignment (y variables)**

All y, x, and z variables are initialised to 0. Then, for each RDC r in data.R, the algorithm finds the CDC j that minimises the transport cost b\_jr among all CDCs in data.T\_r\[r\] (the set of CDCs capable of serving r):

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   best_j = min(data.T_r[r], key=lambda j: data.b_jr.get((j,r), 1e9))  sample[ymap[(best_j, r)]] = 1   `

This is a greedy assignment by cheapest arc. Every RDC is assigned to exactly one CDC, which means constraint (13) — each RDC served exactly once — is satisfied by construction. The 1e9 default in the key function handles missing arcs gracefully by making them effectively infinite cost.

**Step 2 — CDC activation (z variables)**

The set of CDCs that received at least one RDC assignment in Step 1 is collected:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   active_cdcs = set()  for (j,r), nm in ymap.items():      if sample[nm] == 1:          active_cdcs.add(j)  for j in active_cdcs:      sample[zmap[j]] = 1   `

Only CDCs that are actually used are activated. This is important because activating an unused CDC incurs its fixed opening cost c\_j in the objective with no benefit, and also introduces potential violations in constraint (15) since any supply routed through an inactive CDC would violate supply ≤ q\_j · z\_j. By activating exactly the used CDCs, the seed keeps z variables consistent with the y assignments.

**Step 3 — Plant-to-CDC assignment (x variables)**

For each active CDC j, the plant i in data.N\_c\[j\] with the lowest transportation cost a\_ij is assigned:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   best_i = min(data.N_c[j], key=lambda i: data.a_ij.get((i,j), 1e9))  sample[xmap[(best_i, j)]] = 1   `

Only one plant is assigned per CDC. This is a greedy cheapest-arc decision on the supply side. It does not guarantee that capacity and demand constraints (14) and (15) are fully satisfied — a single plant may not supply enough to meet all demand routed to a CDC — but it produces a structurally coherent starting point where the active variables are at least logically consistent with each other.

### Converting the Seed to a Binary Array

The seed is a dictionary mapping QUBO variable name strings (e.g. "y\_2\_5", "z\_3") to integer 0/1 values. Before it can be used as an SA starting state, it must be converted to a dense NumPy array indexed by the QUBO variable ordering:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   def _seed_to_array(self, seed_dict, var_list, var_idx):      x = np.zeros(n, dtype=np.float64)      for nm, v in seed_dict.items():          if nm in var_idx:              x[var_idx[nm]] = float(v)      return x   `

Variables in the QUBO that do not appear in the seed dictionary (notably the binary slack variables s1 and s2 introduced for constraints 14 and 15) are left at 0. This is a reasonable default since zero slack means the constraint expression evaluates to whatever the decision variables produce — the SA will adjust slack bits early in its descent.

### How the Seed Fits into the Population

The population of pop\_size chains is assembled as follows:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   population = [x_gs]  # greedy seed chain  for _ in range(pop_size - 1):      population.append(rng.integers(0, 2, size=n))  # random chains   `

One chain starts from the greedy seed; all others start from independent uniform random binary vectors. The total read budget num\_reads is divided equally across all chains: reads\_per\_chain = num\_reads // pop\_size. This means the greedy seed chain receives the same number of SA runs as each random chain, but begins each of those runs from a structurally informed starting point.

### The Soft Restart Mechanism

Within each chain, reads are not independent restarts from the same initial point. After each read completes, the next read starts from a **perturbed version of the solution just found**, not from the original chain initialisation:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   flip_n = max(1, int(0.05 * n))  idxs = rng.choice(n, flip_n, replace=False)  chain_init = x.copy()  chain_init[idxs] = 1 - chain_init[idxs]   `

Exactly 5% of the bits in the current solution are randomly flipped to produce the next starting point. This is a **basin-hopping** strategy: the SA has annealed to a local minimum, and the perturbation kicks it over a small energy barrier into a neighbouring basin, from which the next SA run descends again. For the greedy seed chain in particular, this means subsequent reads explore the neighbourhood of the greedy solution progressively — starting near feasibility, finding a local minimum, then probing adjacent basins — rather than jumping back to a fully random state.

### Why This Matters for Solution Quality

The combination of these mechanisms addresses the three main failure modes of naive random-start SA on QUBO problems with hard constraints:

The greedy seed ensures that at least one chain in every APL iteration begins from a point where constraint (13) is already satisfied by construction, giving that chain's energy descent a head start toward the feasible region. The population of random chains provides diversity, exploring distant parts of the search space in parallel. The soft restart keeps each chain's reads locally correlated, allowing it to refine solutions incrementally rather than re-discovering the same basin from scratch on every read. Together, these properties increase the probability that at least one sample per iteration has zero violations — which is the condition required for a solution to be accepted and evaluated against the best known objective.