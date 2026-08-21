"""
Tests for src/absorbed_brownian.py

Covers:
  - absorbed_heat_kernel: positivity, symmetry, mass conservation
  - default_probability: range, monotonicity, edge cases
  - evolve_law: mass preservation, atom growth, kernel property
  - evolve_law_gridded: convenience wrapper equivalence
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.absorbed_brownian import (
    absorbed_heat_kernel,
    default_probability,
    evolve_law,
    evolve_law_gridded,
)


# ---------------------------------------------------------------------------
# absorbed_heat_kernel
# ---------------------------------------------------------------------------

class TestAbsorbedHeatKernel:
    def test_nonnegative_values(self):
        x_from = np.array([0.5, 1.0, 2.0])
        x_to = np.array([0.5, 1.0, 2.0])
        K = absorbed_heat_kernel(0.1, x_from, x_to)
        assert np.all(K >= 0.0)

    def test_shape(self):
        x_from = np.linspace(0.5, 2.0, 10)
        x_to = np.linspace(0.5, 2.0, 15)
        K = absorbed_heat_kernel(0.2, x_from, x_to)
        assert K.shape == (10, 15)

    def test_large_delta_spreads_mass(self):
        # Large Delta -> wide diffusion -> more probability mass spread out
        x_from = np.array([1.0])
        x_to = np.linspace(0.1, 3.0, 50)
        K_small = absorbed_heat_kernel(0.01, x_from, x_to)
        K_large = absorbed_heat_kernel(1.0, x_from, x_to)
        # Large delta -> flatter kernel
        max_ratio = float(np.max(K_small)) / (float(np.max(K_large)) + 1e-30)
        assert max_ratio > 1.0, "Small Delta kernel should be more peaked"

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError):
            absorbed_heat_kernel(0.0, np.array([1.0]), np.array([1.0]))
        with pytest.raises(ValueError):
            absorbed_heat_kernel(-0.1, np.array([1.0]), np.array([1.0]))

    def test_mass_conservation_upper_bound(self):
        """Integral of kernel + default probability <= 1 (absorbed BM loses mass)."""
        x_from = np.array([1.0])
        x_to = np.linspace(0.01, 4.0, 500)
        dx = x_to[1] - x_to[0]
        Delta = 0.3
        K = absorbed_heat_kernel(Delta, x_from, x_to)
        integral_kernel = float(np.sum(K[0, :]) * dx)
        p_default = float(default_probability(Delta, x_from))
        total = integral_kernel + p_default
        # Total probability <= 1 (inequality because approximation)
        assert total <= 1.0 + 0.02  # allow 2% quadrature error


# ---------------------------------------------------------------------------
# default_probability
# ---------------------------------------------------------------------------

class TestDefaultProbability:
    def test_range_in_zero_one(self):
        x = np.linspace(0.01, 5.0, 100)
        p = default_probability(0.5, x)
        assert np.all(p >= 0.0)
        assert np.all(p <= 1.0)

    def test_monotone_decreasing_in_x(self):
        x = np.linspace(0.1, 3.0, 50)
        Delta = 0.3
        p = default_probability(Delta, x)
        assert np.all(np.diff(p) <= 1e-10), "P(tau<=Delta) should decrease in x"

    def test_large_x_near_zero(self):
        x = np.array([10.0, 20.0])
        p = default_probability(0.1, x)
        assert np.all(p < 0.01)

    def test_small_x_near_one(self):
        x = np.array([0.001, 0.01])
        p = default_probability(1.0, x)
        assert np.all(p > 0.99)

    def test_increasing_delta_increases_prob(self):
        x = np.array([1.0])
        p_small = float(default_probability(0.1, x))
        p_large = float(default_probability(1.0, x))
        assert p_large > p_small

    def test_formula_matches_normal_cdf(self):
        from scipy.stats import norm
        x = np.array([0.5, 1.0, 1.5])
        Delta = 0.25
        p = default_probability(Delta, x)
        expected = 2.0 * norm.cdf(-x / np.sqrt(Delta))
        np.testing.assert_allclose(p, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# evolve_law
# ---------------------------------------------------------------------------

class TestEvolveLaw:
    def _uniform_init(self, a=0.5, b=2.0, n=100):
        x = np.linspace(a, b, n)
        w = np.ones(n) / n
        return x, w

    def test_total_mass_conserved(self):
        x, w = self._uniform_init()
        _, w_new, new_atom = evolve_law(x, w, 0.0, 0.1)
        total = float(np.sum(w_new)) + new_atom
        assert abs(total - 1.0) < 1e-4

    def test_atom_grows(self):
        x, w = self._uniform_init(0.1, 2.0)
        _, _, atom_new = evolve_law(x, w, 0.0, 0.5)
        assert atom_new > 0.0, "Absorption should increase atom at 0"

    def test_atom_nondecreasing_over_steps(self):
        x, w = self._uniform_init()
        atom = 0.0
        prev_atom = 0.0
        for _ in range(5):
            _, w, atom = evolve_law(x, w, atom, 0.1)
            assert atom >= prev_atom - 1e-10
            prev_atom = atom

    def test_positive_weights(self):
        x, w = self._uniform_init()
        _, w_new, _ = evolve_law(x, w, 0.0, 0.1)
        assert np.all(w_new >= 0.0)

    def test_very_small_states_mostly_absorbed(self):
        x = np.array([0.01, 0.02, 0.05])
        w = np.ones(3) / 3
        _, _, new_atom = evolve_law(x, w, 0.0, 1.0)
        assert new_atom > 0.5, "Very small states should mostly default over large Delta"

    def test_very_large_states_mostly_survive(self):
        x = np.array([5.0, 6.0, 7.0])
        w = np.ones(3) / 3
        _, _, new_atom = evolve_law(x, w, 0.0, 0.1)
        assert new_atom < 0.01, "Large states should rarely default over small Delta"

    def test_initial_atom_preserved_and_grows(self):
        x, w = self._uniform_init()
        atom0 = 0.1
        _, w_new, new_atom = evolve_law(x, w, atom0, 0.1)
        assert new_atom >= atom0  # atom at 0 can only grow


# ---------------------------------------------------------------------------
# evolve_law_gridded (wrapper)
# ---------------------------------------------------------------------------

class TestEvolveLawGridded:
    def test_consistent_with_evolve_law(self):
        x = np.linspace(0.5, 2.0, 50)
        w = np.ones(50) / 50
        Delta = 0.1
        # evolve_law_gridded should match evolve_law with x_out=x
        xg, wg, ag = evolve_law_gridded(x, w, 0.0, Delta)
        xl, wl, al = evolve_law(x, w, 0.0, Delta, x_out=x.copy())
        np.testing.assert_allclose(wg, wl, rtol=1e-10)
        assert abs(ag - al) < 1e-12

    def test_returns_same_grid(self):
        x = np.linspace(0.5, 2.0, 80)
        w = np.ones(80) / 80
        x_out, _, _ = evolve_law_gridded(x, w, 0.0, 0.1)
        np.testing.assert_array_equal(x_out, x)

    def test_multiple_steps_mass_conserved(self):
        x = np.linspace(0.5, 2.0, 100)
        w = np.ones(100) / 100
        atom = 0.0
        for _ in range(10):
            x, w, atom = evolve_law_gridded(x, w, atom, 0.05)
        total = float(np.sum(w)) + atom
        assert abs(total - 1.0) < 1e-3
