"""
Absorbed Brownian motion semigroup S_Delta on R_+.

For the absorbed heat kernel on [0, infinity), the transition density from
positive state x to positive state y over time interval Delta is:

    p_abs(Delta; x, y) = phi_Delta(y - x) - phi_Delta(y + x)

where phi_Delta(z) = (1 / sqrt(2*pi*Delta)) * exp(-z^2 / (2*Delta)).

The probability of absorption (default) starting from x over interval Delta is:

    P_x(tau <= Delta) = 2 * Phi(-x / sqrt(Delta))

where Phi is the standard normal CDF.

These formulae implement exact operator evolution of a discrete law represented
as a weighted grid (x_j, w_j) over positive states.

Custom — Context7 found no library equivalent for this specific absorbed
Brownian semigroup operator on weighted discrete laws (paper notation S_Delta).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm  # Using scipy.stats.norm — Context7 confirmed


def absorbed_heat_kernel(
    Delta: float,
    x_from: np.ndarray,
    x_to: np.ndarray,
) -> np.ndarray:
    """
    Evaluate the absorbed Brownian heat kernel.

    Returns a matrix K[i, j] = p_abs(Delta; x_from[i], x_to[j]).

    Parameters
    ----------
    Delta : float
        Time step (> 0).
    x_from : np.ndarray
        Starting states (all > 0), shape (n,).
    x_to : np.ndarray
        Target states (all > 0), shape (m,).

    Returns
    -------
    np.ndarray
        Kernel matrix of shape (n, m), with K[i, j] >= 0.
    """
    if Delta <= 0:
        raise ValueError(f"Delta must be positive, got {Delta}.")
    sigma = np.sqrt(Delta)
    # Broadcast: x_from -> (n,1), x_to -> (1,m)
    xf = x_from[:, None]
    xt = x_to[None, :]
    K = norm.pdf((xt - xf) / sigma) / sigma - norm.pdf((xt + xf) / sigma) / sigma
    return np.maximum(K, 0.0)  # numerical safety: enforce non-negativity


def default_probability(
    Delta: float,
    x: np.ndarray,
) -> np.ndarray:
    """
    Compute absorption (default) probability P_x(tau <= Delta) for each state x.

    P_x(tau <= Delta) = 2 * Phi(-x / sqrt(Delta)) = 2 * (1 - Phi(x / sqrt(Delta))).

    Parameters
    ----------
    Delta : float
        Time interval (> 0).
    x : np.ndarray
        Starting states (all > 0), shape (n,).

    Returns
    -------
    np.ndarray
        Default probabilities in [0, 1], shape (n,).
    """
    sigma = np.sqrt(Delta)
    return 2.0 * norm.cdf(-x / sigma)


def evolve_law(
    x_grid: np.ndarray,
    w_pos: np.ndarray,
    atom_at_0: float,
    Delta: float,
    x_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Evolve a law (atom_at_0, x_grid, w_pos) under S_Delta.

    The law is represented as:
        - atom_at_0: probability mass at state 0 (already-defaulted banks)
        - x_grid: positive-state grid points (> 0), shape (n,)
        - w_pos: probability weights on positive states, shape (n,)

    After evolution:
        - New atom at 0 = atom_at_0 + sum_j w_j * P_{x_j}(tau <= Delta)
        - Positive-state density on x_out obtained by kernel convolution.

    Parameters
    ----------
    x_grid : np.ndarray
        Positive-state grid, shape (n,).
    w_pos : np.ndarray
        Weights on positive states, shape (n,).
    atom_at_0 : float
        Probability mass at 0 before evolution.
    Delta : float
        Time step.
    x_out : np.ndarray or None
        Output grid for evolved positive states. If None, uses x_grid.

    Returns
    -------
    x_out : np.ndarray
        Output grid (same as input x_out or x_grid).
    w_out : np.ndarray
        Evolved positive-state weights on x_out.
    new_atom : float
        Updated mass at state 0 after absorption.
    """
    if x_out is None:
        x_out = x_grid.copy()

    dx = np.diff(x_out)
    # Midpoint quadrature weights for integration over x_out
    # For non-uniform grid or single point:
    if len(x_out) == 1:
        quad_w = np.ones(1)
    else:
        # Trapezoidal-like bin widths (each point owns half the spacing to each neighbour)
        left = np.concatenate([[dx[0]], dx])
        right = np.concatenate([dx, [dx[-1]]])
        quad_w = 0.5 * (left + right)

    # Kernel matrix: K[i, j] = p_abs(Delta; x_grid[i], x_out[j])
    K = absorbed_heat_kernel(Delta, x_grid, x_out)  # shape (n_in, n_out)

    # Evolved density at x_out[j] (un-normalised per output cell):
    # rho_out[j] = sum_i w_i * K[i, j]
    rho_out = w_pos @ K  # shape (n_out,)

    # Convert density to weights: w_out[j] = rho_out[j] * quad_w[j]
    w_out = rho_out * quad_w
    w_out = np.maximum(w_out, 0.0)

    # Absorbed mass from positive states
    abs_probs = default_probability(Delta, x_grid)  # shape (n,)
    new_absorbed = float(np.dot(w_pos, abs_probs))

    new_atom = atom_at_0 + new_absorbed

    # Normalise w_out to conserve total mass
    total_mass = float(np.sum(w_out)) + new_atom
    if total_mass > 1e-12:
        scale = (1.0 - new_atom) / max(np.sum(w_out), 1e-30)
        w_out *= scale

    return x_out, w_out, new_atom


def evolve_law_gridded(
    x_grid: np.ndarray,
    w_pos: np.ndarray,
    atom_at_0: float,
    Delta: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Convenience wrapper: evolve on the same grid (x_grid = x_out).

    Uses the same grid for input and output, which is the standard choice
    for mean-field operator iteration experiments.

    Parameters
    ----------
    x_grid : np.ndarray
        Positive-state grid, shape (n,).
    w_pos : np.ndarray
        Probability weights on x_grid, shape (n,).
    atom_at_0 : float
        Probability mass at 0.
    Delta : float
        Time step.

    Returns
    -------
    x_grid : np.ndarray
        Same grid (unchanged).
    w_new : np.ndarray
        Updated weights after one S_Delta step.
    new_atom : float
        Updated mass at 0.
    """
    return evolve_law(x_grid, w_pos, atom_at_0, Delta, x_out=x_grid.copy())
