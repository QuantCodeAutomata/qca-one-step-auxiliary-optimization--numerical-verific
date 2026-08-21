"""
Payoff function library and welfare evaluation.

Welfare under law (x_grid, w_pos, atom_at_0) for payoff f is:

    W(f, mu) = atom_at_0 * f(0) + sum_j w_j * f(x_j)

The paper requires non-decreasing concave payoff functions f.
We provide a standard library used across experiments.

Custom — Context7 found no library equivalent for this specific
welfare functional evaluation on discrete probability laws.
"""
from __future__ import annotations

import numpy as np
from typing import Callable


# ── Payoff function library ──────────────────────────────────────────────────

def _survival(x: np.ndarray) -> np.ndarray:
    """Survival indicator proxy: 1 if x > 0, else 0."""
    return (x > 0.0).astype(float)


def _linear_truncated(x: np.ndarray, K: float = 2.0) -> np.ndarray:
    """Capped-linear payoff min(x, K)."""
    return np.minimum(x, K)


def _sqrt_payoff(x: np.ndarray) -> np.ndarray:
    """Square-root payoff sqrt(x)."""
    return np.sqrt(np.maximum(x, 0.0))


def _log1p_payoff(x: np.ndarray) -> np.ndarray:
    """Log payoff log(1 + x)."""
    return np.log1p(np.maximum(x, 0.0))


def _linear_payoff(x: np.ndarray) -> np.ndarray:
    """Identity payoff f(x) = x (linear; non-decreasing concave)."""
    return np.asarray(x, dtype=float)


PAYOFF_FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "survival":         _survival,
    "linear":           _linear_payoff,
    "sqrt":             _sqrt_payoff,
    "log1p":            _log1p_payoff,
    "capped_linear_2":  lambda x: _linear_truncated(x, K=2.0),
    "capped_linear_5":  lambda x: _linear_truncated(x, K=5.0),
}


# ── Welfare computation ───────────────────────────────────────────────────────

def evaluate_welfare(
    f: Callable[[np.ndarray], np.ndarray],
    x_grid: np.ndarray,
    w_pos: np.ndarray,
    atom_at_0: float = 0.0,
) -> float:
    """
    Compute welfare W(f, mu) = atom_at_0 * f(0) + sum_j w_j * f(x_j).

    Parameters
    ----------
    f : callable
        Payoff function mapping np.ndarray -> np.ndarray.
    x_grid : np.ndarray
        Positive-state support points, shape (n,).
    w_pos : np.ndarray
        Probability weights on positive states, shape (n,).
    atom_at_0 : float
        Probability mass at state 0.

    Returns
    -------
    float
        Expected welfare.
    """
    default_contribution = atom_at_0 * float(f(np.array([0.0]))[0])
    positive_contribution = float(np.dot(w_pos, f(x_grid)))
    return default_contribution + positive_contribution


def evaluate_all_payoffs(
    x_grid: np.ndarray,
    w_pos: np.ndarray,
    atom_at_0: float = 0.0,
    payoff_fns: dict[str, Callable] | None = None,
) -> dict[str, float]:
    """
    Evaluate welfare under all payoff functions in the library.

    Parameters
    ----------
    x_grid : np.ndarray
        Positive-state support points, shape (n,).
    w_pos : np.ndarray
        Probability weights, shape (n,).
    atom_at_0 : float
        Mass at state 0.
    payoff_fns : dict or None
        Payoff function library. Defaults to PAYOFF_FUNCTIONS.

    Returns
    -------
    dict[str, float]
        Mapping from payoff name to welfare value.
    """
    if payoff_fns is None:
        payoff_fns = PAYOFF_FUNCTIONS
    return {
        name: evaluate_welfare(f, x_grid, w_pos, atom_at_0)
        for name, f in payoff_fns.items()
    }


def compute_mean(x_grid: np.ndarray, w_pos: np.ndarray) -> float:
    """
    Compute E[X] = sum_j w_j * x_j over positive states only.

    Parameters
    ----------
    x_grid : np.ndarray
        Positive states, shape (n,).
    w_pos : np.ndarray
        Weights on positive states, shape (n,).

    Returns
    -------
    float
        Weighted mean over positive states.
    """
    return float(np.dot(w_pos, x_grid))


def total_mass(w_pos: np.ndarray, atom_at_0: float) -> float:
    """Return total probability mass (should be ~1.0)."""
    return float(np.sum(w_pos)) + atom_at_0
