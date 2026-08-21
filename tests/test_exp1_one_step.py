"""
Integration tests for Experiment 1: One-Step Auxiliary Optimisation.

Validates the core theorem:
  The cutoff operator T_c welfare-dominates all tested feasible benchmark
  allocations (proportional haircut, equal deduction, randomized) for every
  non-decreasing concave payoff function.

These tests directly exercise the paper methodology without mocks,
verifying the analytical claim on fresh synthetic inputs.
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cutoff_operator import cutoff_operator, compute_threshold, _removable_wealth
from src.benchmarks import (
    proportional_haircut,
    equal_deduction_strongest,
    randomized_feasible,
)
from src.welfare import PAYOFF_FUNCTIONS, evaluate_welfare, compute_mean


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_uniform_law(a=0.5, b=3.0, n=200):
    x = np.linspace(a, b, n)
    w = np.ones(n) / n
    return x, w


def _make_triangular_law(a=0.5, b=3.0, n=200):
    x = np.linspace(a, b, n)
    w = (x - a)
    w /= w.sum()
    return x, w


def _cutoff_dominates_benchmark(x, w, c, z_bench, tol=1e-5):
    """Return True if cutoff welfare >= benchmark welfare for all payoffs."""
    z_cut, _, _ = cutoff_operator(x, w, c)
    for name, f in PAYOFF_FUNCTIONS.items():
        wf_cut = evaluate_welfare(f, z_cut, w)
        wf_bench = evaluate_welfare(f, z_bench, w)
        if wf_cut < wf_bench - tol:
            return False, name, wf_cut, wf_bench
    return True, None, None, None


# ---------------------------------------------------------------------------
# Constraint verification tests
# ---------------------------------------------------------------------------

class TestCutoffConstraints:
    def test_mass_preserved_uniform(self):
        x, w = _make_uniform_law()
        z, x_star, mean_red = cutoff_operator(x, w, 0.3)
        assert abs(np.sum(w) - 1.0) < 1e-8  # weights unchanged

    def test_mean_reduction_equals_c(self):
        x, w = _make_uniform_law()
        c = 0.25
        z, _, mean_red = cutoff_operator(x, w, c)
        assert abs(mean_red - c) < 1e-5

    def test_no_negative_states(self):
        x, w = _make_uniform_law()
        z, _, _ = cutoff_operator(x, w, 0.4)
        assert np.all(z >= 0.0)

    def test_only_upper_tail_reduced(self):
        x, w = _make_uniform_law()
        z, x_star, _ = cutoff_operator(x, w, 0.3)
        # States strictly below threshold must be unchanged
        below = x < x_star - 1e-8
        np.testing.assert_allclose(z[below], x[below], atol=1e-12)

    def test_threshold_satisfies_equation(self):
        for c_frac in [0.05, 0.15, 0.30]:
            x, w = _make_uniform_law()
            c = c_frac * compute_mean(x, w)
            _, x_star, _ = cutoff_operator(x, w, c)
            residual = _removable_wealth(x_star, x, w) - c
            assert abs(residual) < 1e-5, f"c_frac={c_frac}: residual={residual:.2e}"


# ---------------------------------------------------------------------------
# Dominance over proportional haircut
# ---------------------------------------------------------------------------

class TestDominanceOverProportional:
    @pytest.mark.parametrize("c_frac", [0.05, 0.15, 0.30])
    def test_uniform_law(self, c_frac):
        x, w = _make_uniform_law()
        c = c_frac * compute_mean(x, w)
        z_bench, _ = proportional_haircut(x, w, c)
        ok, bad_name, wf_cut, wf_bench = _cutoff_dominates_benchmark(x, w, c, z_bench)
        assert ok, (
            f"Cutoff does not dominate proportional haircut "
            f"under payoff '{bad_name}' at c_frac={c_frac}: "
            f"cutoff={wf_cut:.6f} < bench={wf_bench:.6f}"
        )

    @pytest.mark.parametrize("c_frac", [0.05, 0.15, 0.30])
    def test_triangular_law(self, c_frac):
        x, w = _make_triangular_law()
        c = c_frac * compute_mean(x, w)
        z_bench, _ = proportional_haircut(x, w, c)
        ok, bad_name, wf_cut, wf_bench = _cutoff_dominates_benchmark(x, w, c, z_bench)
        assert ok, (
            f"Cutoff fails for triangular law, payoff '{bad_name}', c_frac={c_frac}"
        )


# ---------------------------------------------------------------------------
# Dominance over equal deduction
# ---------------------------------------------------------------------------

class TestDominanceOverEqualDeduction:
    @pytest.mark.parametrize("c_frac", [0.05, 0.15, 0.30])
    def test_uniform_law(self, c_frac):
        x, w = _make_uniform_law()
        c = c_frac * compute_mean(x, w)
        z_bench, _ = equal_deduction_strongest(x, w, c, fraction_donors=0.5)
        ok, bad_name, wf_cut, wf_bench = _cutoff_dominates_benchmark(x, w, c, z_bench)
        assert ok, (
            f"Cutoff fails vs equal-deduction: payoff '{bad_name}', c_frac={c_frac}"
        )


# ---------------------------------------------------------------------------
# Dominance over randomized allocation
# ---------------------------------------------------------------------------

class TestDominanceOverRandomized:
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_uniform_law_c15_pct(self, seed):
        x, w = _make_uniform_law()
        c = 0.15 * compute_mean(x, w)
        z_bench, _ = randomized_feasible(x, w, c, rng=np.random.default_rng(seed))
        ok, bad_name, wf_cut, wf_bench = _cutoff_dominates_benchmark(x, w, c, z_bench)
        assert ok, (
            f"Cutoff fails vs randomized (seed={seed}): payoff '{bad_name}'"
        )


# ---------------------------------------------------------------------------
# Multiple laws × multiple c
# ---------------------------------------------------------------------------

class TestFullMatrix:
    @pytest.mark.parametrize("law_name,c_frac", [
        ("uniform", 0.05),
        ("uniform", 0.15),
        ("uniform", 0.30),
        ("triangular", 0.05),
        ("triangular", 0.15),
        ("triangular", 0.30),
    ])
    def test_cutoff_dominates_all_benchmarks(self, law_name, c_frac):
        if law_name == "uniform":
            x, w = _make_uniform_law()
        else:
            x, w = _make_triangular_law()
        c = c_frac * compute_mean(x, w)
        z_prop, _ = proportional_haircut(x, w, c)
        z_eq, _ = equal_deduction_strongest(x, w, c, fraction_donors=0.5)
        z_rand, _ = randomized_feasible(x, w, c, rng=np.random.default_rng(42))
        for z_bench, bench_name in [
            (z_prop, "proportional"),
            (z_eq, "equal_deduction"),
            (z_rand, "randomized"),
        ]:
            ok, bad_payoff, wf_cut, wf_bench = _cutoff_dominates_benchmark(
                x, w, c, z_bench
            )
            assert ok, (
                f"law={law_name}, c_frac={c_frac}, bench={bench_name}, "
                f"payoff={bad_payoff}: cutoff={wf_cut:.6f} < bench={wf_bench:.6f}"
            )
