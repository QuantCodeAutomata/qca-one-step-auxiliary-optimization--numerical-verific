"""
Tests for src/benchmarks.py

Covers:
  - proportional_haircut: mean removal, no defaults, correctness
  - equal_deduction_strongest: mean removal, feasibility, donor restriction
  - randomized_feasible: mean removal, bounds, feasibility
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.benchmarks import (
    proportional_haircut,
    equal_deduction_strongest,
    randomized_feasible,
)


# ---------------------------------------------------------------------------
# proportional_haircut
# ---------------------------------------------------------------------------

class TestProportionalHaircut:
    def _uniform_law(self, a=0.5, b=2.0, n=100):
        x = np.linspace(a, b, n)
        w = np.ones(n) / n
        return x, w

    def test_mean_reduction_correct(self):
        x, w = self._uniform_law()
        c = 0.3
        z, lam = proportional_haircut(x, w, c)
        actual_reduction = float(np.dot(w, x - z))
        assert abs(actual_reduction - c) < 1e-8

    def test_no_defaults_induced(self):
        x, w = self._uniform_law()
        z, _ = proportional_haircut(x, w, 0.3)
        assert np.all(z > 0.0)

    def test_lambda_in_range(self):
        x, w = self._uniform_law()
        _, lam = proportional_haircut(x, w, 0.3)
        assert 0.0 <= lam <= 1.0

    def test_proportionality(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.ones(3) / 3
        c = 0.5
        z, lam = proportional_haircut(x, w, c)
        # All states reduced by same fraction
        ratios = z / x
        np.testing.assert_allclose(ratios, ratios[0], rtol=1e-10)

    def test_zero_c_identity(self):
        x, w = self._uniform_law()
        z, lam = proportional_haircut(x, w, 0.0)
        np.testing.assert_allclose(z, x, rtol=1e-10)
        assert lam == pytest.approx(0.0)

    def test_small_total_mean_raises(self):
        x = np.array([1e-15])
        w = np.array([1.0])
        with pytest.raises(ValueError):
            proportional_haircut(x, w, 0.5)


# ---------------------------------------------------------------------------
# equal_deduction_strongest
# ---------------------------------------------------------------------------

class TestEqualDeductionStrongest:
    def _uniform_law(self, a=0.5, b=2.0, n=100):
        x = np.linspace(a, b, n)
        w = np.ones(n) / n
        return x, w

    def test_no_defaults_induced(self):
        x, w = self._uniform_law()
        z, d = equal_deduction_strongest(x, w, 0.2, fraction_donors=0.5)
        assert np.all(z > 0.0)

    def test_weak_states_unchanged(self):
        x = np.linspace(0.5, 2.0, 100)
        w = np.ones(100) / 100
        c = 0.15
        z, d = equal_deduction_strongest(x, w, c, fraction_donors=0.3)
        # Bottom 70% of states should be unchanged
        threshold_idx = int(0.7 * len(x))
        np.testing.assert_allclose(z[:threshold_idx], x[:threshold_idx], rtol=1e-10)

    def test_output_shape(self):
        x, w = self._uniform_law()
        z, d = equal_deduction_strongest(x, w, 0.2)
        assert z.shape == x.shape

    def test_deduction_nonnegative(self):
        x, w = self._uniform_law()
        z, d = equal_deduction_strongest(x, w, 0.2)
        assert d >= 0.0


# ---------------------------------------------------------------------------
# randomized_feasible
# ---------------------------------------------------------------------------

class TestRandomizedFeasible:
    def _uniform_law(self, a=0.5, b=2.0, n=100):
        x = np.linspace(a, b, n)
        w = np.ones(n) / n
        return x, w

    def test_no_defaults_induced(self):
        x, w = self._uniform_law()
        rng = np.random.default_rng(42)
        z, d = randomized_feasible(x, w, 0.3, rng=rng)
        assert np.all(z > 0.0)

    def test_deductions_within_capacity(self):
        x, w = self._uniform_law()
        rng = np.random.default_rng(42)
        z, d = randomized_feasible(x, w, 0.3, rng=rng)
        assert np.all(d >= 0.0)
        assert np.all(d <= x + 1e-12)

    def test_output_shape(self):
        x, w = self._uniform_law()
        rng = np.random.default_rng(0)
        z, d = randomized_feasible(x, w, 0.2, rng=rng)
        assert z.shape == x.shape
        assert d.shape == x.shape

    def test_infeasible_c_raises(self):
        x, w = self._uniform_law()
        total_mean = float(np.dot(w, x))
        with pytest.raises(ValueError):
            randomized_feasible(x, w, total_mean + 1.0)

    def test_reproducibility_with_seed(self):
        x, w = self._uniform_law()
        z1, _ = randomized_feasible(x, w, 0.2, rng=np.random.default_rng(7))
        z2, _ = randomized_feasible(x, w, 0.2, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(z1, z2)

    def test_different_seeds_give_different_results(self):
        x, w = self._uniform_law()
        z1, _ = randomized_feasible(x, w, 0.2, rng=np.random.default_rng(7))
        z2, _ = randomized_feasible(x, w, 0.2, rng=np.random.default_rng(99))
        assert not np.allclose(z1, z2)
