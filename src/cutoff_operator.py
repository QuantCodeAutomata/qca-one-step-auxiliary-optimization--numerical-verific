"""
Cutoff (tax-the-richest) operator for one-step auxiliary optimisation.

For a discrete probability law represented by sorted points x_j with weights w_j,
the cutoff operator T_c maps every positive state x to min(x, x*), where the
threshold x* is the largest y satisfying

    g(y) = sum_j w_j * max(x_j - y, 0) - c = 0.

This is the analytical solution to the welfare-maximising reallocation
described in the paper (Section on auxiliary optimisation / Prop. on T_c).

Custom — Context7 found no library equivalent for this specific measure-transport
cutoff operator (paper's T_c definition).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq  # Using scipy.optimize.brentq — Context7 confirmed


def _removable_wealth(y: float, x: np.ndarray, w: np.ndarray) -> float:
    """
    Compute g(y) = sum_j w_j * max(x_j - y, 0).

    This is the expected excess wealth above threshold y under law (x, w).
    Monotone decreasing in y, continuous, piecewise-linear on a finite grid.

    Parameters
    ----------
    y : float
        Candidate threshold.
    x : np.ndarray
        Support points (positive states), shape (n,).
    w : np.ndarray
        Probability weights, shape (n,).

    Returns
    -------
    float
        Sum of w_j * (x_j - y)^+ over all j.
    """
    return float(np.dot(w, np.maximum(x - y, 0.0)))


def compute_threshold(
    x: np.ndarray,
    w: np.ndarray,
    c: float,
    tol: float = 1e-12,
) -> float:
    """
    Compute the cutoff threshold x* solving g(x*) = c.

    Uses scipy.optimize.brentq on [0, max(x)] because g is continuous,
    strictly decreasing on the support of mu, and satisfies:
        g(0)  = sum_j w_j x_j  >= c  (feasibility assumption)
        g(max(x)) = 0           <= c

    Parameters
    ----------
    x : np.ndarray
        Positive-state support points of law mu, shape (n,).
    w : np.ndarray
        Probability weights summing to at most 1 (positive-state mass only).
    c : float
        Required loss amount.  Must satisfy 0 <= c <= sum(w * x).
    tol : float
        Solver tolerance.

    Returns
    -------
    float
        Threshold x* such that sum_j w_j * (x_j - x*)^+ = c.

    Raises
    ------
    ValueError
        If c > total mean (infeasible) or c < 0.
    """
    if c < 0:
        raise ValueError(f"Loss amount c={c} must be non-negative.")
    total_mean = float(np.dot(w, x))
    if c > total_mean + tol:
        raise ValueError(
            f"c={c:.6g} exceeds total mean {total_mean:.6g}: infeasible."
        )
    if c <= 0.0:
        return float(np.max(x)) + 1.0  # identity map: no cutoff needed

    x_max = float(np.max(x))
    # g(0) = total_mean >= c, g(x_max) = 0 <= c
    if total_mean <= c + tol:
        # c exactly equals full mean: threshold = 0
        return 0.0

    def objective(y: float) -> float:
        return _removable_wealth(y, x, w) - c

    # Using scipy.optimize.brentq — Context7 confirmed
    x_star = brentq(objective, 0.0, x_max, xtol=tol, rtol=tol)
    return float(x_star)


def apply_cutoff(
    x: np.ndarray,
    x_star: float,
) -> np.ndarray:
    """
    Apply the cutoff map z_j = min(x_j, x*) to an array of states.

    Parameters
    ----------
    x : np.ndarray
        State values (positive states only, no atoms at 0 needed here).
    x_star : float
        Cutoff threshold.

    Returns
    -------
    np.ndarray
        Post-allocation states z = min(x, x*), same shape as x.
    """
    return np.minimum(x, x_star)


def cutoff_operator(
    x: np.ndarray,
    w: np.ndarray,
    c: float,
    tol: float = 1e-12,
) -> tuple[np.ndarray, float, float]:
    """
    Full cutoff operator T_c applied to a discrete law (x, w).

    Returns the post-allocation states z = min(x, x*) together with
    the computed threshold and the resulting mean reduction.

    Parameters
    ----------
    x : np.ndarray
        Positive-state support points, shape (n,).
    w : np.ndarray
        Probability weights of positive states, shape (n,).
    c : float
        Required mean reduction.
    tol : float
        Solver tolerance.

    Returns
    -------
    z : np.ndarray
        Post-cutoff states min(x, x*).
    x_star : float
        Cutoff threshold solving the removable-wealth equation.
    mean_reduction : float
        Actual mean reduction sum_j w_j (x_j - z_j), should equal c.
    """
    x_star = compute_threshold(x, w, c, tol=tol)
    z = apply_cutoff(x, x_star)
    mean_reduction = float(np.dot(w, x - z))
    return z, x_star, mean_reduction


def water_filling_cap(
    y: np.ndarray,
    c: float,
    tol: float = 1e-12,
) -> tuple[np.ndarray, float]:
    """
    Water-filling / cap solver for the finite-player boundary allocation.

    Computes kappa solving sum_j (y_j - kappa)^+ = c and returns
    Gamma_c(y) = min(y_j, kappa) for each survivor j.

    This is mathematically identical to cutoff_operator on uniform weights.

    Parameters
    ----------
    y : np.ndarray
        Survivor states (all strictly positive), shape (m,).
    c : float
        Total loss to absorb from survivors.
    tol : float
        Solver tolerance.

    Returns
    -------
    z : np.ndarray
        Post-allocation survivor states Gamma_c(y).
    kappa : float
        Common cap threshold.
    """
    m = len(y)
    w = np.ones(m) / m  # uniform weights
    # scale c to match the mean convention (sum vs mean)
    # For finite-player: sum_j (y_j - kappa)^+ = c (sum, not mean).
    # Rewrite as mean: sum_j (1/m)(y_j - kappa)^+ = c/m.
    c_mean = c / m
    x_star = compute_threshold(y, w, c_mean, tol=tol)
    z = apply_cutoff(y, x_star)
    return z, x_star
