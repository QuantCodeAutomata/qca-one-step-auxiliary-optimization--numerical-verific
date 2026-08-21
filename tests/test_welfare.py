"""
Tests for src/welfare.py

Covers:
  - PAYOFF_FUNCTIONS: non-decreasing, concavity at representative points
  - evaluate_welfare: correctness, atom contribution, edge cases
  - evaluate_all_payoffs: consistency across payoff library
  - compute_mean, total_mass: simple arithmetic checks
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.welfare import (
    PAYOFF_FUNCTIONS,
    evaluate_welfare,
    evaluate_all_payoffs,
    compute_mean,
    total_mass,
)


# ---------------------------------------------------------------------------
# Payoff function properties
# ---------------------------------------------------------------------------

class TestPayoffFunctions:
    _x = np.linspace(0.0, 3.0, 200)

    def test_all_payoffs_nondecreasing(self):
        for name, f in PAYOFF_FUNCTIONS.items():
            vals = f(self._x)
            diffs = np.diff(vals)
            assert np.all(diffs >= -1e-10), (
                f"Payoff '{name}' is not non-decreasing: min diff = {diffs.min()}"
            )

    def test_all_payoffs_nonnegative_for_nonneg_input(self):
        for name, f in PAYOFF_FUNCTIONS.items():
            vals = f(self._x)
            assert np.all(vals >= -1e-12), (
                f"Payoff '{name}' produces negative values for x >= 0"
            )

    def test_survival_at_zero(self):
        f = PAYOFF_FUNCTIONS["survival"]
        assert float(f(np.array([0.0]))[0]) == 0.0

    def test_survival_positive_states(self):
        f = PAYOFF_FUNCTIONS["survival"]
        vals = f(np.array([0.1, 0.5, 1.0, 2.0]))
        np.testing.assert_array_equal(vals, np.ones(4))

    def test_linear_identity(self):
        f = PAYOFF_FUNCTIONS["linear"]
        x_test = np.array([0.5, 1.0, 2.5])
        np.testing.assert_allclose(f(x_test), x_test)

    def test_sqrt_at_zero(self):
        f = PAYOFF_FUNCTIONS["sqrt"]
        assert float(f(np.array([0.0]))[0]) == pytest.approx(0.0)

    def test_log1p_at_zero(self):
        f = PAYOFF_FUNCTIONS["log1p"]
        assert float(f(np.array([0.0]))[0]) == pytest.approx(0.0)

    def test_capped_linear_caps_at_k(self):
        f = PAYOFF_FUNCTIONS["capped_linear_2"]
        vals = f(np.array([0.5, 1.0, 2.0, 3.0, 5.0]))
        assert np.all(vals <= 2.0 + 1e-12)

    def test_concavity_of_sqrt(self):
        """Jensen's inequality: f(E[X]) >= E[f(X)] for concave f."""
        f = PAYOFF_FUNCTIONS["sqrt"]
        x = np.array([1.0, 4.0])
        w = np.array([0.5, 0.5])
        ex = float(np.dot(w, x))  # 2.5
        e_fx = float(np.dot(w, f(x)))  # 0.5*1 + 0.5*2 = 1.5
        f_ex = float(f(np.array([ex]))[0])  # sqrt(2.5) ≈ 1.581
        assert f_ex >= e_fx - 1e-10, "sqrt not concave"

    def test_concavity_of_log1p(self):
        f = PAYOFF_FUNCTIONS["log1p"]
        x = np.array([0.5, 2.5])
        w = np.array([0.5, 0.5])
        ex = float(np.dot(w, x))
        e_fx = float(np.dot(w, f(x)))
        f_ex = float(f(np.array([ex]))[0])
        assert f_ex >= e_fx - 1e-10, "log1p not concave"


# ---------------------------------------------------------------------------
# evaluate_welfare
# ---------------------------------------------------------------------------

class TestEvaluateWelfare:
    def test_pure_positive_no_atom(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        f = PAYOFF_FUNCTIONS["linear"]
        w_val = evaluate_welfare(f, x, w, atom_at_0=0.0)
        assert abs(w_val - 2.0) < 1e-10

    def test_atom_contribution_with_survival(self):
        x = np.array([1.0])
        w = np.array([0.6])
        atom = 0.4
        f = PAYOFF_FUNCTIONS["survival"]
        # survival: f(0)=0, f(1)=1  -> 0.4*0 + 0.6*1 = 0.6
        w_val = evaluate_welfare(f, x, w, atom_at_0=atom)
        assert abs(w_val - 0.6) < 1e-10

    def test_welfare_monotone_in_allocation(self):
        """Higher allocations -> higher welfare for non-decreasing f."""
        x_low = np.array([0.5, 1.0])
        x_high = np.array([1.0, 1.5])
        w = np.array([0.5, 0.5])
        for name, f in PAYOFF_FUNCTIONS.items():
            if name == "survival":
                continue  # survival is step-like; skip
            w_low = evaluate_welfare(f, x_low, w)
            w_high = evaluate_welfare(f, x_high, w)
            assert w_high >= w_low - 1e-10, (
                f"Payoff '{name}': welfare not monotone ({w_low:.4f} -> {w_high:.4f})"
            )

    def test_empty_positive_mass_all_at_atom(self):
        f = PAYOFF_FUNCTIONS["sqrt"]
        x = np.array([1.0])
        w = np.array([0.0])
        atom = 1.0
        w_val = evaluate_welfare(f, x, w, atom_at_0=atom)
        # 1.0 * f(0) + 0.0 * f(1.0) = 0
        assert abs(w_val - 0.0) < 1e-12

    def test_known_value_sqrt(self):
        x = np.array([0.0, 1.0, 4.0])
        w = np.array([0.0, 0.5, 0.5])
        f = PAYOFF_FUNCTIONS["sqrt"]
        # 0 + 0.5*1 + 0.5*2 = 1.5
        w_val = evaluate_welfare(f, x, w)
        assert abs(w_val - 1.5) < 1e-10

    def test_known_value_capped_linear(self):
        x = np.array([1.0, 3.0])
        w = np.array([0.5, 0.5])
        f = PAYOFF_FUNCTIONS["capped_linear_2"]
        # 0.5*min(1,2) + 0.5*min(3,2) = 0.5*1 + 0.5*2 = 1.5
        w_val = evaluate_welfare(f, x, w)
        assert abs(w_val - 1.5) < 1e-10


# ---------------------------------------------------------------------------
# evaluate_all_payoffs
# ---------------------------------------------------------------------------

class TestEvaluateAllPayoffs:
    def test_returns_all_keys(self):
        x = np.linspace(0.5, 2.0, 50)
        w = np.ones(50) / 50
        result = evaluate_all_payoffs(x, w)
        assert set(result.keys()) == set(PAYOFF_FUNCTIONS.keys())

    def test_all_values_finite(self):
        x = np.linspace(0.5, 2.0, 50)
        w = np.ones(50) / 50
        result = evaluate_all_payoffs(x, w, atom_at_0=0.1)
        for name, val in result.items():
            assert np.isfinite(val), f"Payoff '{name}' gives non-finite welfare"

    def test_custom_payoff_library(self):
        custom = {"quad": lambda x: np.minimum(x**2, 4.0)}  # concave for x <=2
        x = np.array([1.0, 2.0])
        w = np.array([0.5, 0.5])
        result = evaluate_all_payoffs(x, w, payoff_fns=custom)
        assert "quad" in result


# ---------------------------------------------------------------------------
# compute_mean
# ---------------------------------------------------------------------------

class TestComputeMean:
    def test_uniform_grid(self):
        x = np.linspace(1.0, 3.0, 100)
        w = np.ones(100) / 100
        mean = compute_mean(x, w)
        assert abs(mean - 2.0) < 0.02

    def test_single_point(self):
        x = np.array([2.5])
        w = np.array([1.0])
        assert compute_mean(x, w) == pytest.approx(2.5)

    def test_weighted_mean(self):
        x = np.array([1.0, 3.0])
        w = np.array([0.3, 0.7])
        expected = 0.3 * 1.0 + 0.7 * 3.0
        assert compute_mean(x, w) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# total_mass
# ---------------------------------------------------------------------------

class TestTotalMass:
    def test_sums_to_one(self):
        x = np.linspace(0.5, 2.0, 100)
        w = np.ones(100) / 100
        # Simulate 10% default
        w_scaled = w * 0.9
        assert abs(total_mass(w_scaled, 0.1) - 1.0) < 1e-10

    def test_all_mass_at_atom(self):
        w = np.array([0.0])
        assert total_mass(w, 1.0) == pytest.approx(1.0)

    def test_no_atom(self):
        w = np.ones(10) / 10
        assert total_mass(w, 0.0) == pytest.approx(1.0)
