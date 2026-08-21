"""
Tests for src/cutoff_operator.py

Covers:
  - compute_threshold: correctness, boundary cases, infeasibility
  - apply_cutoff: mapping
  - cutoff_operator: end-to-end, mass preservation, mean reduction
  - water_filling_cap: finite-player wrapper correctness
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cutoff_operator import (
    _removable_wealth,
    compute_threshold,
    apply_cutoff,
    cutoff_operator,
    water_filling_cap,
)


# ---------------------------------------------------------------------------
# _removable_wealth
# ---------------------------------------------------------------------------

class TestRemovableWealth:
    def test_single_point_above_threshold(self):
        x = np.array([2.0])
        w = np.array([1.0])
        assert abs(_removable_wealth(1.0, x, w) - 1.0) < 1e-12

    def test_single_point_below_threshold(self):
        x = np.array([0.5])
        w = np.array([1.0])
        assert _removable_wealth(1.0, x, w) == pytest.approx(0.0)

    def test_uniform_law_threshold_zero(self):
        x = np.linspace(1.0, 3.0, 100)
        w = np.ones(100) / 100
        g0 = _removable_wealth(0.0, x, w)
        expected = float(np.mean(x))
        assert abs(g0 - expected) < 1e-10

    def test_at_max_threshold_returns_zero(self):
        x = np.array([1.0, 2.0, 3.0])
        w = np.ones(3) / 3
        g_max = _removable_wealth(float(np.max(x)), x, w)
        assert g_max == pytest.approx(0.0, abs=1e-12)

    def test_monotone_decreasing(self):
        x = np.linspace(0.5, 2.0, 50)
        w = np.ones(50) / 50
        thresholds = np.linspace(0.0, 2.0, 30)
        g_vals = [_removable_wealth(t, x, w) for t in thresholds]
        diffs = np.diff(g_vals)
        assert np.all(diffs <= 1e-12), "g(y) should be non-increasing in y"


# ---------------------------------------------------------------------------
# compute_threshold
# ---------------------------------------------------------------------------

class TestComputeThreshold:
    def _uniform_law(self, a, b, n=200):
        x = np.linspace(a, b, n)
        w = np.ones(n) / n
        return x, w

    def test_basic_correctness(self):
        x, w = self._uniform_law(0.0, 2.0)
        c = 0.3
        x_star = compute_threshold(x, w, c)
        residual = _removable_wealth(x_star, x, w) - c
        assert abs(residual) < 1e-6, f"Residual {residual} too large"

    def test_c_zero_returns_large_threshold(self):
        x, w = self._uniform_law(0.5, 2.0)
        x_star = compute_threshold(x, w, 0.0)
        assert x_star > np.max(x)

    def test_c_equals_total_mean_returns_zero(self):
        x, w = self._uniform_law(0.5, 2.0)
        total_mean = float(np.dot(w, x))
        x_star = compute_threshold(x, w, total_mean - 1e-10)
        assert x_star >= 0.0

    def test_negative_c_raises(self):
        x, w = self._uniform_law(0.5, 2.0)
        with pytest.raises(ValueError, match="non-negative"):
            compute_threshold(x, w, -0.1)

    def test_infeasible_c_raises(self):
        x, w = self._uniform_law(0.5, 2.0)
        total_mean = float(np.dot(w, x))
        with pytest.raises(ValueError, match="infeasible"):
            compute_threshold(x, w, total_mean + 1.0)

    def test_triangular_law(self):
        x = np.linspace(0.5, 3.0, 300)
        w = x - 0.5
        w /= w.sum()
        c = 0.4
        x_star = compute_threshold(x, w, c)
        residual = _removable_wealth(x_star, x, w) - c
        assert abs(residual) < 1e-6

    def test_discrete_law(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        w = np.array([0.25, 0.25, 0.25, 0.25])
        c = 0.5
        x_star = compute_threshold(x, w, c)
        residual = _removable_wealth(x_star, x, w) - c
        assert abs(residual) < 1e-6

    def test_threshold_within_support(self):
        x, w = self._uniform_law(0.5, 2.0)
        c = 0.2
        x_star = compute_threshold(x, w, c)
        # Threshold must be within [0, max(x)]
        assert 0.0 <= x_star <= float(np.max(x)) + 1.0

    def test_small_c_gives_large_threshold(self):
        x, w = self._uniform_law(0.5, 2.0)
        c_small = 1e-4
        x_star_small = compute_threshold(x, w, c_small)
        c_large = 0.5
        x_star_large = compute_threshold(x, w, c_large)
        assert x_star_small > x_star_large, "Larger c should give smaller threshold"


# ---------------------------------------------------------------------------
# apply_cutoff
# ---------------------------------------------------------------------------

class TestApplyCutoff:
    def test_all_below_threshold(self):
        x = np.array([0.5, 1.0, 1.5])
        z = apply_cutoff(x, 2.0)
        np.testing.assert_array_equal(z, x)

    def test_all_above_threshold(self):
        x = np.array([3.0, 4.0, 5.0])
        z = apply_cutoff(x, 2.0)
        np.testing.assert_array_equal(z, np.full_like(x, 2.0))

    def test_mixed(self):
        x = np.array([0.5, 1.5, 2.5, 3.5])
        x_star = 2.0
        z = apply_cutoff(x, x_star)
        expected = np.array([0.5, 1.5, 2.0, 2.0])
        np.testing.assert_allclose(z, expected)

    def test_no_mutation_of_input(self):
        x = np.array([1.0, 2.0, 3.0])
        x_orig = x.copy()
        _ = apply_cutoff(x, 2.0)
        np.testing.assert_array_equal(x, x_orig)

    def test_output_nonnegative(self):
        x = np.linspace(0.01, 5.0, 100)
        z = apply_cutoff(x, 1.5)
        assert np.all(z >= 0.0)


# ---------------------------------------------------------------------------
# cutoff_operator (end-to-end)
# ---------------------------------------------------------------------------

class TestCutoffOperator:
    def _uniform_law(self, a=0.5, b=2.0, n=200):
        x = np.linspace(a, b, n)
        w = np.ones(n) / n
        return x, w

    def test_mass_preserved(self):
        x, w = self._uniform_law()
        z, x_star, mean_red = cutoff_operator(x, w, 0.3)
        # z has same weights w; total mass = sum(w) should be ~1
        assert abs(np.sum(w) - 1.0) < 1e-8

    def test_mean_reduction_matches_c(self):
        x, w = self._uniform_law()
        c = 0.25
        z, x_star, mean_red = cutoff_operator(x, w, c)
        assert abs(mean_red - c) < 1e-5

    def test_no_state_below_zero(self):
        x, w = self._uniform_law()
        z, x_star, mean_red = cutoff_operator(x, w, 0.4)
        assert np.all(z >= 0.0)

    def test_only_upper_tail_taxed(self):
        x, w = self._uniform_law()
        z, x_star, _ = cutoff_operator(x, w, 0.3)
        # States below x_star are unchanged
        mask = x < x_star - 1e-9
        np.testing.assert_allclose(z[mask], x[mask], atol=1e-12)

    def test_welfare_cutoff_geq_proportional(self):
        """Key theorem: cutoff weakly dominates proportional haircut for sqrt payoff."""
        from src.benchmarks import proportional_haircut
        x, w = self._uniform_law()
        c = 0.3
        z_cut, _, _ = cutoff_operator(x, w, c)
        z_hair, _ = proportional_haircut(x, w, c)
        welfare_cut = float(np.dot(w, np.sqrt(z_cut)))
        welfare_hair = float(np.dot(w, np.sqrt(z_hair)))
        assert welfare_cut >= welfare_hair - 1e-6, (
            f"Cutoff welfare {welfare_cut:.6f} < proportional {welfare_hair:.6f}"
        )

    def test_multiple_c_values(self):
        x, w = self._uniform_law()
        total_mean = float(np.dot(w, x))
        for frac in [0.05, 0.1, 0.2, 0.3, 0.4]:
            c = frac * total_mean
            z, x_star, mean_red = cutoff_operator(x, w, c)
            assert abs(mean_red - c) < 1e-4, f"frac={frac}: mean_red={mean_red}, c={c}"

    def test_threshold_satisfies_removable_wealth(self):
        x, w = self._uniform_law()
        c = 0.2
        _, x_star, _ = cutoff_operator(x, w, c)
        residual = _removable_wealth(x_star, x, w) - c
        assert abs(residual) < 1e-5

    def test_bimodal_law(self):
        from scipy.stats import norm
        x = np.linspace(0.1, 3.0, 300)
        w = 0.5 * norm.pdf(x, 0.8, 0.2) + 0.5 * norm.pdf(x, 2.2, 0.2)
        w /= w.sum()
        c = 0.15
        z, x_star, mean_red = cutoff_operator(x, w, c)
        assert abs(mean_red - c) < 1e-4
        assert np.all(z >= 0.0)


# ---------------------------------------------------------------------------
# water_filling_cap
# ---------------------------------------------------------------------------

class TestWaterFillingCap:
    def test_basic_two_survivors(self):
        y = np.array([2.0, 1.0])
        c = 0.5
        z, kappa = water_filling_cap(y, c)
        # sum z_j = sum y_j - c
        assert abs(np.sum(z) - (np.sum(y) - c)) < 1e-6
        # all z_j > 0
        assert np.all(z > 0)
        # all z_j <= y_j
        assert np.all(z <= y + 1e-12)

    def test_equal_survivors(self):
        y = np.array([1.0, 1.0, 1.0])
        c = 0.3
        z, kappa = water_filling_cap(y, c)
        assert abs(np.sum(z) - (np.sum(y) - c)) < 1e-6
        # Equal survivors -> equal reduction
        np.testing.assert_allclose(z, np.full(3, (3.0 - 0.3) / 3), atol=1e-4)

    def test_richest_taxed_most(self):
        y = np.array([3.0, 0.5, 0.5])
        c = 0.6
        z, kappa = water_filling_cap(y, c)
        # The rich survivor y[0]=3.0 should bear most of the loss
        assert z[0] < y[0]
        assert z[1] >= y[1] - 1e-6  # small survivors unchanged
        assert z[2] >= y[2] - 1e-6

    def test_permutation_invariance(self):
        y = np.array([1.5, 0.8, 1.2])
        c = 0.3
        z, kappa = water_filling_cap(y, c)
        # Welfare under sqrt should be same for any permutation of y
        from itertools import permutations
        welfares = []
        for perm in permutations(range(3)):
            y_perm = y[list(perm)]
            z_perm, _ = water_filling_cap(y_perm, c)
            welfares.append(float(np.sum(np.sqrt(z_perm))))
        assert np.std(welfares) < 1e-6

    def test_kappa_solves_removable_wealth(self):
        y = np.array([2.0, 1.5, 1.0, 0.5])
        c = 0.4
        z, kappa = water_filling_cap(y, c)
        # sum_j (y_j - kappa)^+ = c
        removable = float(np.sum(np.maximum(y - kappa, 0.0)))
        assert abs(removable - c) < 1e-5
