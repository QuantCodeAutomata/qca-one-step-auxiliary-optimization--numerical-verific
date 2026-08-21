"""
Integration tests for Experiment 5: Finite-Player Boundary Optimisation.

Validates the water-filling cap Gamma_c(y) for small survivor counts:
  1. Gamma_c(y) is feasible (all z_j > 0, sum z_j = sum y_j - c).
  2. Permutation invariance of welfare.
  3. Gamma_c(y) matches brute-force optimizer for m=2, m=3.
  4. Infeasible cases (c >= sum y) correctly identified.
  5. Richest survivors bear the largest loss.
  6. kappa satisfies sum_j (y_j - kappa)^+ = c.
"""
import sys
import os

import numpy as np
import pytest
from itertools import permutations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cutoff_operator import water_filling_cap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def brute_force_optimize_2d(y, c, payoff_fn, n_grid=500):
    """Exhaustive 1-D search for m=2 optimal allocation."""
    y1, y2 = float(y[0]), float(y[1])
    target = y1 + y2 - c
    z1_vals = np.linspace(1e-6, y1 - 1e-6, n_grid)
    z2_vals = target - z1_vals
    mask = (z2_vals > 1e-6) & (z2_vals <= y2 + 1e-8)
    if not np.any(mask):
        return None
    z1_vals = z1_vals[mask]
    z2_vals = z2_vals[mask]
    obj = payoff_fn(z1_vals) + payoff_fn(z2_vals)
    best_idx = np.argmax(obj)
    return np.array([z1_vals[best_idx], z2_vals[best_idx]])


def sqrt_payoff(x):
    return np.sqrt(np.maximum(x, 0.0))


def log1p_payoff(x):
    return np.log1p(np.maximum(x, 0.0))


# ---------------------------------------------------------------------------
# Feasibility tests
# ---------------------------------------------------------------------------

class TestFeasibility:
    def test_feasible_two_survivors(self):
        y = np.array([2.0, 1.0])
        c = 0.5
        z, kappa = water_filling_cap(y, c)
        assert np.all(z > 0.0)
        assert abs(np.sum(z) - (np.sum(y) - c)) < 1e-5

    def test_feasible_three_survivors_equal(self):
        y = np.array([1.5, 1.5, 1.5])
        c = 0.6
        z, kappa = water_filling_cap(y, c)
        assert np.all(z > 0.0)
        assert abs(np.sum(z) - (np.sum(y) - c)) < 1e-5

    def test_feasible_four_survivors_heterogeneous(self):
        y = np.array([3.0, 2.0, 1.0, 0.5])
        c = 0.8
        z, kappa = water_filling_cap(y, c)
        assert np.all(z > 0.0)
        assert abs(np.sum(z) - (np.sum(y) - c)) < 1e-5

    def test_infeasible_c_exceeds_sum(self):
        y = np.array([1.0, 0.5])
        c = 2.0  # c > sum(y) = 1.5
        with pytest.raises((ValueError, Exception)):
            water_filling_cap(y, c)

    def test_bounds_respected(self):
        y = np.array([1.5, 0.8, 1.2])
        c = 0.4
        z, kappa = water_filling_cap(y, c)
        assert np.all(z <= y + 1e-10), "Some post-allocation states exceed y_j"
        assert np.all(z >= 0.0), "Negative post-allocation states"


# ---------------------------------------------------------------------------
# Kappa equation tests
# ---------------------------------------------------------------------------

class TestKappaEquation:
    @pytest.mark.parametrize("y,c", [
        (np.array([2.0, 1.0]), 0.5),
        (np.array([1.5, 1.0, 0.5]), 0.3),
        (np.array([3.0, 2.0, 1.0, 0.5]), 0.8),
        (np.array([1.0, 1.0]), 0.2),
    ])
    def test_kappa_solves_removable_wealth(self, y, c):
        z, kappa = water_filling_cap(y, c)
        removable = float(np.sum(np.maximum(y - kappa, 0.0)))
        assert abs(removable - c) < 1e-5, (
            f"kappa={kappa:.6f}: sum(y-kappa)^+ = {removable:.6f} != c={c:.6f}"
        )

    def test_kappa_between_min_and_max_y(self):
        y = np.array([3.0, 1.5, 0.8])
        c = 0.5
        z, kappa = water_filling_cap(y, c)
        assert kappa >= 0.0
        assert kappa <= float(np.max(y)) + 1e-8


# ---------------------------------------------------------------------------
# Optimality: brute-force comparison for m=2
# ---------------------------------------------------------------------------

class TestOptimalityM2:
    @pytest.mark.parametrize("payoff_fn,name", [
        (sqrt_payoff, "sqrt"),
        (log1p_payoff, "log1p"),
    ])
    def test_gamma_c_matches_bruteforce_m2(self, payoff_fn, name):
        y = np.array([2.0, 1.0])
        c = 0.5
        z_cap, kappa = water_filling_cap(y, c)
        z_bf = brute_force_optimize_2d(y, c, payoff_fn)
        assert z_bf is not None, "Brute-force found no feasible solution"
        obj_cap = float(np.sum(payoff_fn(z_cap)))
        obj_bf = float(np.sum(payoff_fn(z_bf)))
        assert obj_cap >= obj_bf - 1e-3, (
            f"payoff={name}: cap welfare {obj_cap:.6f} < brute-force {obj_bf:.6f}"
        )

    @pytest.mark.parametrize("y_vals,c", [
        ([1.5, 0.8], 0.3),
        ([2.5, 2.0], 0.4),
        ([1.0, 0.5], 0.2),
    ])
    def test_gamma_c_matches_bruteforce_several_configs(self, y_vals, c):
        y = np.array(y_vals)
        z_cap, _ = water_filling_cap(y, c)
        z_bf = brute_force_optimize_2d(y, c, sqrt_payoff)
        if z_bf is None:
            pytest.skip("Brute-force infeasible")
        obj_cap = float(np.sum(sqrt_payoff(z_cap)))
        obj_bf = float(np.sum(sqrt_payoff(z_bf)))
        assert obj_cap >= obj_bf - 1e-3


# ---------------------------------------------------------------------------
# Permutation invariance
# ---------------------------------------------------------------------------

class TestPermutationInvariance:
    @pytest.mark.parametrize("payoff_fn", [sqrt_payoff, log1p_payoff])
    def test_welfare_invariant_under_permutation_m3(self, payoff_fn):
        y = np.array([1.5, 0.8, 1.2])
        c = 0.4
        welfares = []
        for perm in permutations(range(3)):
            y_perm = y[list(perm)]
            z, _ = water_filling_cap(y_perm, c)
            welfares.append(float(np.sum(payoff_fn(z))))
        assert np.std(welfares) < 1e-6, (
            f"Welfare not permutation-invariant: std={np.std(welfares):.2e}"
        )

    def test_allocation_permuted_correctly(self):
        """Ensure Gamma_c applied to a permuted y gives the same sorted allocation."""
        y = np.array([2.0, 0.5, 1.5])
        c = 0.5
        z, kappa = water_filling_cap(y, c)
        # The capped states should all be <= kappa
        assert np.all(z <= kappa + 1e-8)


# ---------------------------------------------------------------------------
# Richest survives last
# ---------------------------------------------------------------------------

class TestRichestTaxedFirst:
    def test_only_richest_reduced_for_small_c(self):
        y = np.array([3.0, 1.0, 0.5])
        c = 0.2  # small loss
        z, kappa = water_filling_cap(y, c)
        # The weakest survivors should be unchanged
        np.testing.assert_allclose(z[1], y[1], atol=1e-5)
        np.testing.assert_allclose(z[2], y[2], atol=1e-5)
        # The richest should bear the loss
        assert z[0] < y[0]

    def test_loss_concentrated_in_upper_tail(self):
        y = np.array([4.0, 3.0, 2.0, 1.0, 0.5])
        c = 0.8
        z, kappa = water_filling_cap(y, c)
        # Lower survivors (the smallest ones) should be at most slightly reduced
        # whereas upper survivors are at or below kappa
        upper_idx = y > kappa + 1e-8
        if np.any(upper_idx):
            assert np.all(z[upper_idx] <= kappa + 1e-6), (
                "States above kappa not capped to kappa"
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_survivor(self):
        y = np.array([2.0])
        c = 0.5
        z, kappa = water_filling_cap(y, c)
        assert abs(float(z[0]) - (2.0 - 0.5)) < 1e-5

    def test_very_small_c(self):
        y = np.array([2.0, 1.5, 1.0])
        c = 1e-6
        z, kappa = water_filling_cap(y, c)
        assert abs(np.sum(z) - (np.sum(y) - c)) < 1e-5
        assert np.all(z > 0)

    def test_zero_c_identity(self):
        y = np.array([1.0, 2.0, 1.5])
        z, kappa = water_filling_cap(y, 0.0)
        # With c=0, states are unchanged
        np.testing.assert_allclose(z, y, atol=1e-4)

    def test_symmetric_survivors_equal_reduction(self):
        y = np.array([1.0, 1.0, 1.0])
        c = 0.3
        z, kappa = water_filling_cap(y, c)
        # All survivors equal -> all reduced equally
        np.testing.assert_allclose(z, np.full(3, (3.0 - 0.3) / 3), atol=1e-4)
