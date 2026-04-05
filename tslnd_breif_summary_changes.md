Differences Between test2\_updated.py (Baseline) and tslnd\_solver.py (Improved)
--------------------------------------------------------------------------------

### 1\. PPLN Preprocessing (entirely new in improved version)

The improved solver introduces a **Preprocessing Procedure of Logistic Network (PPLN)**, attributed to Ciacco et al. (2026), which does not exist in the baseline at all. Before the QUBO is built, PPLN computes a lower bound β and upper bound γ on the minimum number of CDCs needed to feasibly serve all RDCs. It then enumerates combinations of CDCs of increasing size from β to γ, filtering by coverage of all RDCs and capacity feasibility, and selects the lowest heuristic-cost valid combination. The result is a reduced instance with a smaller set |C|, which directly reduces the number of decision variables fed into the QUBO — and therefore the number of qubits used by the annealer. In the baseline, the full unfiltered set C is always used.

### 2\. QUBO Formulation — Slack Variables for Constraints (14) and (15)

This is one of the most significant structural differences affecting solution quality.

In test2\_updated.py, constraints (14) and (15) — demand-supply balance and supply-capacity — are encoded as **inequality penalties** directly in the QUBO. The penalty terms are of the form P·(∑ wᵢxᵢ)², which penalises any nonzero value of the expression but does not enforce that it equals zero. This means the QUBO landscape has no hard zero-point for these constraints; the annealer can find low-energy states that still violate them slightly, because the unconstrained quadratic form does not distinguish between "slightly violated" and "exactly satisfied."

In tslnd\_solver.py, constraints (14) and (15) are re-encoded as **equality constraints** using binary-encoded slack variables. For each CDC j, a slack variable S1ⱼ with ⌈log₂(U1ⱼ + 1)⌉ binary bits is introduced such that:

> ∑ d\_r · y\_{j,r} − ∑ p\_i · x\_{i,j} + S1ⱼ = 0

and similarly S2ⱼ for constraint (15). The QUBO then penalises the square of this full equality expression. This is the standard QUBO equality encoding. The consequence is that a feasible solution — one where the slack absorbs the difference — corresponds to an exact energy minimum for those penalty terms, rather than an approximate one. This makes the energy landscape much better aligned with true feasibility.

### 3\. Penalty Initialisation and Bounds

In the baseline, penalties are initialised at scale (the maximum single cost coefficient) with a floor of 0.1 \* scale and no upper bound. There is no ceiling — penalties can grow without limit.

In the improved solver, penalties start at 10 \* obj\_scale, with a defined floor of 0.5 \* obj\_scale and a ceiling of 500 \* obj\_scale. The higher initial value and the explicit maximum bound serve two purposes: the higher start gives constraints immediate weight relative to the objective from iteration 1, and the ceiling prevents numerical instability in later iterations where runaway penalty growth would cause the QUBO matrix to become ill-conditioned and the SA landscape to flatten.

### 4\. APL Update Rule

Both implementations use a multiplicative APL rule, but they differ in the growth and decay formulation.

The baseline applies: P ← P \* (1 + α \* v) when violated, and P ← max(P\_min, P \* (1 − β)) when satisfied, with α = 0.2, β = 0.05.

The improved solver uses the same structure but with α = 0.25 and β = 0.05, and crucially applies the ceiling clamp min(P\_max, ...) on growth. Additionally, the penalty update in the improved solver is driven by the **least-violated sample across all reads in the iteration**, not by whichever sample happened to be evaluated last. In the baseline, v1, v2, v3 used for the APL update are simply the values from the last sample processed in the loop — which is arbitrary and not the best information available. In the improved solver, the iteration explicitly tracks iter\_best\_v as the minimum total violation seen across all samples, giving the APL update a more informative and stable signal.

### 5\. Feasibility Acceptance and Solution Selection

In test2\_updated.py, the inner loop over samples breaks on the **first** sample with zero violations:

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   if v1 == v2 == v3 == 0:      obj = original_objective(...)      best = QUBOSolution(...)      break   `

It then breaks out of the outer APL loop entirely as well (if best: break). This means the solver stops at the first feasible solution it encounters regardless of its objective value — it makes no attempt to search for a better one.

In tslnd\_solver.py, the inner loop does **not break** on finding a feasible solution. It evaluates all samples in every iteration and maintains a running best defined as the feasible solution (zero violations) with the lowest objective value seen so far across all iterations. The APL loop continues to run until the iteration limit or time limit is reached, continuously attempting to improve the best known objective. An elite pool of all distinct feasible solutions found is also maintained.

This is a fundamental change in search philosophy: the baseline performs **feasibility-first, single-accept** search, while the improved solver performs **exhaustive-feasible, objective-optimising** search across the full iteration budget.

### 6\. Sampler Architecture — Population SA vs. Single Neal Call

The baseline makes a single neal.SimulatedAnnealingSampler().sample\_qubo() call per iteration with num\_sweeps=3000.

The improved solver introduces a fallback PopulationSASampler for when neal is unavailable. This sampler runs multiple independent SA chains in parallel, one of which is warm-started using a **greedy feasibility seed** — a deterministic construction heuristic that assigns each RDC to its cheapest CDC, activates those CDCs, and assigns the cheapest plant to each. The remaining chains start from random binary configurations. Between reads within a chain, the next starting point is the current solution with 5% of bits randomly flipped, implementing a soft restart strategy. When neal is available, the standard sampler is used directly, but the QUBO it receives is the improved slack-variable formulation described in point 2.

### 7\. Reproducibility and Seeding

The baseline has no random seed control. The improved solver seeds both Python's random module and NumPy's random state at the start of solve() via --seed (default 42), making all runs fully reproducible.

### Summary Table

Aspecttest2\_updated.pytslnd\_solver.pyPreprocessingNonePPLN (reduces |C|)Constraints (14),(15)Inequality penaltyEquality with binary slack variablesPenalty initialisation1×scale, no ceiling10×scale, ceiling at 500×scaleAPL violation signalLast sample in loopBest (min violation) sample across all readsFeasibility acceptanceFirst zero-violation sample, then stopAll zero-violation samples, keep best objectiveSearch terminationStops at first feasibleRuns full iteration/time budgetSamplerSingle Neal chainNeal (if available) or Population SA with greedy seedReproducibilityNoneSeeded (default 42)