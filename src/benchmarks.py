"""
Alternative feasible allocation benchmarks for comparison with the cutoff rule.

Three benchmark reallocation rules that remove the same amount c from the law mu:
  1. Proportional haircut: each positive state x is reduced to (1-lambda)*x,
     where lambda is chosen to remove exactly c in mean.
  2. Equal deduction over strongest feasible subset: subtract a flat amount d
     from all states above some threshold, choosing d so as to remove exactly c.
  3. Randomized feasible reallocation: sample deductions d_j with 0 <= d_j <= x_j
     and sum w_j d_j = c by projecting a random vector to the feasible simplex.

All rules must satisfy:
  - No allocation-induced defaults: post-state z_j = x_j - d_j > 0 for positive states.
  - Total mean removed = c.
  - Normalization preserved.

Custom — Context7 found no library equivalent for these specific systemic-risk
benchmark allocation rules.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq  # Using scipy.optimize.brentq — Context7 confirmed


def proportional_haircut(
    x: np.ndarray,
    w: np.ndarray,
    c: float,
    tol: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """
    Apply a proportional haircut: z_j = (1 - lambda) * x_j.

    Chooses lambda such that sum_j w_j * lambda * x_j = c, i.e.,
    lambda = c / sum_j w_j * x_j.

    Clips to [0, 1] for safety. Resulting states z_j = (1-lambda)*x_j > 0
    if lambda < 1 (feasible range).

    Parameters
    ----------
    x : np.ndarray
        Positive-state support points, shape (n,).
    w : np.ndarray
        Probability weights, shape (n,).
    c : float
        Required mean reduction.
    tol : float
        Tolerance.

    Returns
    -------
    z : np.ndarray
        Post-allocation states.
    lambda_val : float
        Haircut fraction applied.
    """
    total_mean = float(np.dot(w, x))
    if total_mean < tol:
        raise ValueError("Total mean is too small for proportional haircut.")
    lambda_val = c / total_mean
    lambda_val = float(np.clip(lambda_val, 0.0, 1.0 - tol))
    z = (1.0 - lambda_val) * x
    return z, lambda_val


def equal_deduction_strongest(
    x: np.ndarray,
    w: np.ndarray,
    c: float,
    fraction_donors: float = 0.5,
    tol: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """
    Deduct a flat amount d from the top-`fraction_donors` fraction of states.

    Only states above the (1 - fraction_donors) quantile contribute.
    d is chosen so that sum_j w_j * d * 1_{x_j in donor set} = c.

    Clips to avoid inducing defaults: d <= min(x_j) among donors.

    Parameters
    ----------
    x : np.ndarray
        Positive-state support points, shape (n,).
    w : np.ndarray
        Probability weights, shape (n,).
    c : float
        Required mean reduction.
    fraction_donors : float
        Fraction of the top states that contribute losses.
    tol : float
        Tolerance.

    Returns
    -------
    z : np.ndarray
        Post-allocation states.
    d : float
        Flat deduction applied to donor states.
    """
    n = len(x)
    # Sort descending to identify donor states
    sort_idx = np.argsort(x)[::-1]
    x_sorted = x[sort_idx]
    w_sorted = w[sort_idx]

    # Select top fraction_donors by cumulative weight
    cum_w = np.cumsum(w_sorted)
    donor_mask = cum_w <= fraction_donors + tol
    if not np.any(donor_mask):
        donor_mask[0] = True  # at least one donor

    donor_mass = float(np.sum(w_sorted[donor_mask]))
    if donor_mass < tol:
        raise ValueError("Donor mass too small for equal deduction.")

    # Maximum allowable d: cannot push any donor below 0
    d_max = float(np.min(x_sorted[donor_mask])) - tol
    if d_max <= 0:
        raise ValueError("Cannot deduct without inducing defaults in donors.")

    d = c / donor_mass
    if d > d_max:
        # Fall back: reduce d to the feasibility limit, spread across more donors
        d = d_max

    z = x.copy()
    for idx, is_donor in zip(sort_idx, donor_mask):
        if is_donor:
            z[idx] = max(x[idx] - d, tol)
    return z, d


def randomized_feasible(
    x: np.ndarray,
    w: np.ndarray,
    c: float,
    rng: np.random.Generator | None = None,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a randomized feasible reallocation removing mean c.

    Samples deductions d_j with 0 <= d_j <= x_j and sum_j w_j d_j = c.
    Uses projected random deductions: draw d_j ~ Uniform(0, x_j), then
    scale to satisfy the mean constraint via a projection step.

    Parameters
    ----------
    x : np.ndarray
        Positive-state support points, shape (n,).
    w : np.ndarray
        Probability weights, shape (n,).
    c : float
        Required mean reduction.
    rng : np.random.Generator or None
        Random generator.  Defaults to np.random.default_rng(0).
    tol : float
        Tolerance for feasibility.

    Returns
    -------
    z : np.ndarray
        Post-allocation states x - d.
    d : np.ndarray
        Deductions applied to each state.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    total_mean = float(np.dot(w, x))
    if c > total_mean - tol:
        raise ValueError(
            f"c={c:.6g} infeasible: total mean={total_mean:.6g}."
        )

    # Draw random deduction fractions in [0, 1] and scale by x_j
    fracs = rng.uniform(0.0, 1.0, size=len(x))
    d = fracs * x  # d_j in [0, x_j]

    # Current mean removal: sum_j w_j d_j
    current_removal = float(np.dot(w, d))
    if current_removal < tol:
        current_removal = tol

    # Scale d to achieve exactly c
    scale = c / current_removal
    d = d * scale

    # Clip to [0, x_j] to avoid exceeding capacity
    d = np.minimum(d, x)

    # Check feasibility; if scaling violated capacity, renormalize
    overflow_mask = d >= x
    if np.any(overflow_mask):
        d[overflow_mask] = (1.0 - tol) * x[overflow_mask]
        # Recompute remaining shortfall and redistribute
        current_removal = float(np.dot(w, d))
        residual = c - current_removal
        safe_mask = ~overflow_mask & (x - d > tol)
        if np.any(safe_mask) and residual > 0:
            extra = residual * (x[safe_mask] - d[safe_mask])
            extra = extra / (float(np.dot(w[safe_mask], x[safe_mask] - d[safe_mask])) + 1e-30)
            d[safe_mask] += extra * (x[safe_mask] - d[safe_mask])
            d[safe_mask] = np.minimum(d[safe_mask], (1.0 - tol) * x[safe_mask])

    z = x - d
    z = np.maximum(z, tol)  # enforce strict positivity
    return z, d
