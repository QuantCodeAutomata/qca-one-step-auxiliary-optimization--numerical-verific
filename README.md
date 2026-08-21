# One-Step Auxiliary Optimisation — Numerical Verification

This repository implements and numerically validates the **cutoff (tax-the-richest) allocation rule** for systemic-risk mean-field control, as described in the reference paper. The project covers six experiments ranging from purely static welfare comparisons through finite-particle-to-mean-field convergence analysis.

## Repository Structure

```
qca-one-step-auxiliary-optimization--numerical-verific/
├── data/                      # Data directory (no external data required — synthetic only)
├── src/                       # Core reusable modules
│   ├── cutoff_operator.py     # Cutoff threshold solver (T_c) and water-filling map
│   ├── absorbed_brownian.py   # Absorbed Brownian semigroup S_Δ on a spatial grid
│   ├── welfare.py             # Payoff function library and welfare integration
│   └── benchmarks.py         # Alternative feasible allocation rules
├── exp/                       # Experiment scripts
│   ├── exp1_one_step_optimization.py      # Static one-step cutoff dominance
│   ├── exp2_discrete_dynamic_control.py   # Discrete-time repeated cutoff
│   ├── exp3_vanishing_delta_convergence.py# Δ→0 stabilisation
│   ├── exp4_continuous_time_limit.py      # Reflected limit: MC + PDE
│   ├── exp5_finite_player_boundary.py     # Small-n finite-player optimality
│   └── exp6_finite_particle_mf_convergence.py  # N-particle convergence
├── tests/                     # pytest test suite (187 tests)
├── results/                   # CSV tables, PNG plots, RESULTS.md summary
└── AGENTS.md                  # Agent memory / implementation notes
```

## Experiments

| # | Title | Key claim verified |
|---|-------|--------------------|
| 1 | One-step auxiliary optimisation | Cutoff T_c dominates all feasible alternatives in 72/72 test cases |
| 2 | Discrete-time mean-field control | Repeated cutoff achieves highest welfare among dynamic policies |
| 3 | Vanishing-Δ convergence | Terminal value and boundary process stabilise as Δ→0 |
| 4 | Continuous-time reflected limit | Balance identity E[L_t]=αP(τ≤t); support consistent with [0,Λ_t] |
| 5 | Finite-player boundary optimisation | Water-filling map Γ_c(y) = CVXPY optimiser for m=2,3,4 survivors |
| 6 | Finite-particle-to-mean-field convergence | W₁ distance decreases from 0.245 (N=10) to 0.066 (N=200) |

Full numerical results are in [`results/RESULTS.md`](results/RESULTS.md).

## Installation

```bash
pip install numpy scipy pandas matplotlib seaborn cvxpy pytest
# Optional extras used by specific experiments:
pip install POT numba
```

All computations use **synthetic data only** — no external market data or API
keys are required.

## Running Experiments

```bash
# Run a single experiment
python exp/exp1_one_step_optimization.py

# Run all experiments
for f in exp/exp*.py; do python "$f"; done
```

## Running Tests

```bash
pytest tests/ -v
# Expected: 187 passed
```

## Core Concepts

### Cutoff Operator T_c μ

Given an admissible law μ on [0,∞) and loss amount c ≥ 0, the threshold x*
is the unique solution of

    g(y) = ∫(x − y)⁺ μ(dx) − c = 0

found by bisection (monotone, piecewise-linear on a grid).  The post-allocation
law is the pushforward of μ under x → min(x, x*).

### Absorbed Brownian Semigroup S_Δ

For a spatial grid with masses w_j at positions x_j > 0, one step uses the
absorbed heat kernel on ℝ₊:

    p_abs(Δ; x, y) = φ_Δ(y − x) − φ_Δ(y + x),   x, y > 0

with default probability P_x(τ ≤ Δ) = 2Φ(−x/√Δ).

### Finite-Player Water-Filling Map Γ_c(y)

For survivor vector y ∈ ℝ₊ᵐ and loss c, solve

    Σⱼ (yⱼ − κ)⁺ = c

for threshold κ, then set zⱼ = min(yⱼ, κ).  This is the unique optimiser
of Σⱼ f(zⱼ) over the feasible set K_m(y, c) for any non-decreasing concave f.
