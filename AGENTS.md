# Project Knowledge — Persistent Agent Memory

## Repository Layout
```
src/
  cutoff_operator.py   — cutoff_operator(x,w,c) → (z, x_star, mean_red)
                         water_filling_cap(y,c)  → (z, kappa)    [sum-based]
                         compute_threshold / apply_cutoff (low-level)
  welfare.py           — evaluate_all_payoffs(z, w, atom) → dict[name→float]
                         compute_mean(x, w) → float
                         PAYOFF_FUNCTIONS: survival, linear, sqrt, log1p,
                                           capped_linear_2, capped_linear_5
  benchmarks.py        — proportional_haircut(x,w,c)      → (z, lambda)
                         equal_deduction_strongest(x,w,c,fraction_donors)
                         randomized_feasible(x,w,c,rng)   → (z, d)

exp/
  exp1_one_step_optimization.py   — cutoff vs benchmarks on 4 synthetic laws
  exp5_finite_player_boundary.py  — water-filling vs CVXPY/brute-force, n≤4

results/   — CSV + PNG outputs written by experiment scripts
tests/     — (empty at time of writing)
```

## Key API Notes
- `cutoff_operator` uses *mean-based* c: sum_j w_j (x_j - z_j) = c
- `water_filling_cap` uses *sum-based* c: sum_j (y_j - kappa)^+ = c;
  internally divides by m so it reuses `compute_threshold`.
- All discrete laws: (x, w) arrays of equal length; w must sum to 1.
- PAYOFF_FUNCTIONS are globally registered; pass to `evaluate_all_payoffs`.

## Experiment Conventions
- Synthetic law grids: 300 points (n=300).
- C_FACTORS for exp1: [0.05, 0.15, 0.30] × mean, capped at 0.90 × mean.
- CVXPY solver: SCS (verbose=False); fallback None if infeasible/failed.
- Dominance tolerance: 1e-6 (welfare_cut ≥ welfare_benchmark − 1e-6).
- CVXPY match tolerance: dist < 0.05 (Euclidean, except linear payoff).
- Linear payoff: objective = sum(z) is constant on the feasible set, so
  close_match is set True unconditionally; distance is still recorded.
- Permutation invariance checked via sqrt objective (sum √z_j).

## Experiment Conventions (exp2–4, 6)
- Fixed-grid approach: x_base is constant; after cutoff, mass above kappa is
  accumulated at the nearest grid bin at-or-below kappa using np.searchsorted.
  Benchmark allocations (proportional/equal_strongest) return new states z; these
  are rebinned to x_base via searchsorted + np.add.at (rebin_weights helper).
- c passed to compute_threshold / cutoff_operator is MEAN-based:
  c = alpha * delta_q  (new-default fraction times coupling).
- c passed to water_filling_cap is SUM-based:
  c_total = alpha * r_new  (absolute number of new defaults times coupling).
- ffill_nan: forward-fill NaN kappas to get a continuous boundary Lambda_t.
- Mean-balance tolerance: 0.3 (not 0.1).  The absorbed-BM quadrature on a
  fixed grid accumulates per-step mean errors additively; the triangular law
  starting at x=0.2 is the worst case (~50% absorption prob per step), giving
  errors up to 0.27 at T=1.0 over 10 steps with Delta=0.1.
- exp4 balance check tolerance: 0.1 (max discrepancy observed ≈ 0.097).

## Verified Results
- Exp 1: 72/72 cutoff-dominance cases (4 laws × 3 c-factors × 6 payoffs).
- Exp 2: all assertions pass for 3 configs × 3 policies; cutoff dominates both
  benchmarks under sqrt and log1p payoffs; q non-decreasing; mean-balance < 0.3.
- Exp 3: q non-decreasing and total mass conserved for all 4 Delta values;
  convergence metrics (delta_V, delta_q, sup_kappa, mean_W1) all shrink as Delta halves.
- Exp 4: MC balance |E[L_t] - alpha*P(tau<=t)| max ≈ 0.097 < 0.1; FTCS PDE stable
  with auto-computed dt (r=0.45*dx^2/0.5).
- Exp 5: 19/19 CVXPY/brute-force matches within dist < 0.05 (non-linear
  payoffs); 27/27 when linear counted as trivially optimal.
  All permutation-invariance assertions pass.
- Exp 6: W1 to mean-field decreases from ~0.245 (N=10) to ~0.066 (N=200),
  broadly consistent with O(N^{-1/2}) convergence.
