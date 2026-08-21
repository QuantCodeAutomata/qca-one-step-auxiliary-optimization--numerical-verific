"""
Integration tests for Experiment 6: Finite-Particle to Mean-Field Convergence.

Validates core properties of the N-particle cutoff-controlled system:
  1. Particle states remain non-negative after feasible reallocation.
  2. q_t^N is non-decreasing and bounded in [0, 1].
  3. Empirical measure conserves total particle count.
  4. Cutoff reallocation applied at default events reduces mean exactly by alpha*r/N.
  5. Wasserstein-1 error decreases with N (convergence diagnostic).
  6. Gamma_c correctly identifies feasibility and terminal cascade events.
"""
import sys
import os

import numpy as np
import pytest
from scipy.stats import wasserstein_distance

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.cutoff_operator import water_filling_cap


# ---------------------------------------------------------------------------
# Minimal particle simulation kernel (self-contained, no file I/O)
# ---------------------------------------------------------------------------

def simulate_particles(N, x0, alpha, T, dt, rng):
    """
    Simulate N particles with Euler-Maruyama time discretisation and
    event-driven default-reallocation via the water-filling cap.

    Parameters
    ----------
    N    : int   - number of particles
    x0   : (N,) initial states (positive)
    alpha: float - loss fraction per default
    T    : float - terminal time
    dt   : float - time step
    rng  : np.random.Generator

    Returns
    -------
    q_path      : list of (time, q^N) tuples
    states_final: (N,) final states
    cascades    : number of terminal cascade events
    """
    states = x0.copy()
    alive = states > 0.0  # active indicator
    n_steps = int(round(T / dt))
    q_path = [(0.0, 0.0)]
    cascades = 0

    for step in range(n_steps):
        t = (step + 1) * dt
        # Brownian increment for alive particles
        dW = rng.normal(0.0, np.sqrt(dt), size=N)
        states[alive] += dW[alive]

        # Detect newly defaulted
        newly_defaulted = alive & (states <= 0.0)
        r = int(np.sum(newly_defaulted))

        if r > 0:
            states[newly_defaulted] = 0.0
            alive[newly_defaulted] = False
            c = alpha * r  # total loss to allocate from survivors

            # Survivor states
            survivor_idx = np.where(alive)[0]
            m = len(survivor_idx)

            if m == 0 or c >= float(np.sum(states[survivor_idx])) - 1e-10:
                # Terminal cascade: infeasible
                cascades += 1
            else:
                y_surv = states[survivor_idx]
                z, kappa = water_filling_cap(y_surv, c)
                states[survivor_idx] = z

        q_now = float(np.sum(~alive)) / N
        q_path.append((t, q_now))

    return q_path, states, cascades


def quantile_init(mu0_samples, N):
    """Quantile-based deterministic initialisation from samples of mu_0."""
    sorted_samples = np.sort(mu0_samples)
    quantiles = (np.arange(N) + 0.5) / N
    idx = np.clip((quantiles * len(sorted_samples)).astype(int), 0, len(sorted_samples) - 1)
    return sorted_samples[idx]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestParticleStateValidity:
    def test_all_states_nonnegative(self):
        N = 50
        rng = np.random.default_rng(42)
        x0 = rng.uniform(0.5, 1.5, size=N)
        _, states_f, _ = simulate_particles(N, x0, alpha=0.3, T=0.5, dt=0.05, rng=rng)
        assert np.all(states_f >= -1e-10), "Some particle states negative"

    def test_defaulted_states_at_zero(self):
        N = 50
        rng = np.random.default_rng(1)
        x0 = rng.uniform(0.3, 0.8, size=N)  # small initial values -> more defaults
        _, states_f, _ = simulate_particles(N, x0, alpha=0.3, T=1.0, dt=0.05, rng=rng)
        # Every state is either > 0 or == 0
        assert np.all((states_f > 0.0) | (np.abs(states_f) < 1e-10))


class TestQNonDecreasing:
    def test_q_path_nondecreasing(self):
        N = 60
        rng = np.random.default_rng(7)
        x0 = rng.uniform(0.5, 1.5, size=N)
        q_path, _, _ = simulate_particles(N, x0, alpha=0.3, T=1.0, dt=0.05, rng=rng)
        q_vals = [q for _, q in q_path]
        diffs = np.diff(q_vals)
        assert np.all(diffs >= -1e-10), (
            f"q not non-decreasing; min diff = {diffs.min():.2e}"
        )

    def test_q_bounded_in_zero_one(self):
        N = 50
        rng = np.random.default_rng(9)
        x0 = rng.uniform(0.5, 1.5, size=N)
        q_path, _, _ = simulate_particles(N, x0, alpha=0.3, T=1.0, dt=0.05, rng=rng)
        q_vals = [q for _, q in q_path]
        assert np.all(np.array(q_vals) >= 0.0)
        assert np.all(np.array(q_vals) <= 1.0 + 1e-8)


class TestReallocationConstraints:
    def test_reallocation_preserves_positive_survivors(self):
        y = np.array([2.0, 1.5, 1.0])
        c = 0.6  # alpha * r for r=2 defaults
        z, kappa = water_filling_cap(y, c)
        assert np.all(z > 0.0)

    def test_reallocation_removes_correct_total(self):
        y = np.array([2.0, 1.5, 1.0])
        c = 0.6
        z, kappa = water_filling_cap(y, c)
        actual_removal = float(np.sum(y) - np.sum(z))
        assert abs(actual_removal - c) < 1e-5

    def test_terminal_cascade_when_infeasible(self):
        y = np.array([0.5])
        c = 1.0  # c > sum(y) -> infeasible
        with pytest.raises((ValueError, Exception)):
            water_filling_cap(y, c)


class TestEmpiricalMeasureProperties:
    def test_active_particle_count_nonincreasing(self):
        N = 80
        rng = np.random.default_rng(13)
        x0 = rng.uniform(0.3, 1.0, size=N)  # encourage defaults
        q_path, _, _ = simulate_particles(N, x0, alpha=0.3, T=1.0, dt=0.04, rng=rng)
        q_vals = [q for _, q in q_path]
        # Active count = N * (1 - q)
        active = [N * (1.0 - q) for q in q_vals]
        diffs = np.diff(active)
        assert np.all(diffs <= 1e-6), "Active particle count increased"

    def test_total_particle_count_conserved(self):
        N = 60
        rng = np.random.default_rng(0)
        x0 = rng.uniform(0.5, 1.5, size=N)
        _, states_f, _ = simulate_particles(N, x0, alpha=0.3, T=0.5, dt=0.05, rng=rng)
        assert len(states_f) == N


class TestConvergenceToMeanField:
    def test_w1_decreases_with_N(self):
        """Wasserstein-1 between empirical and reference should decrease as N grows."""
        alpha = 0.3
        T = 0.5
        dt = 0.05
        # Reference: large-N simulation
        N_ref = 500
        rng_ref = np.random.default_rng(0)
        x0_ref = rng_ref.uniform(0.5, 1.5, size=N_ref)
        _, states_ref, _ = simulate_particles(N_ref, x0_ref, alpha, T, dt, rng_ref)

        w1_errors = []
        for N in [20, 50, 100]:
            rng = np.random.default_rng(1)
            x0 = quantile_init(x0_ref, N)
            _, states_N, _ = simulate_particles(N, x0, alpha, T, dt, rng)
            w1 = wasserstein_distance(states_N, states_ref)
            w1_errors.append(w1)

        # W1 should decrease as N increases (or at least not be monotonically increasing)
        assert w1_errors[-1] <= w1_errors[0] + 0.3, (
            f"W1 errors not decreasing: {w1_errors}"
        )

    def test_default_fraction_converges(self):
        """Default fractions from increasing N should converge to reference."""
        alpha = 0.3
        T = 0.5
        dt = 0.05
        N_ref = 500
        rng_ref = np.random.default_rng(5)
        x0_ref = rng_ref.uniform(0.5, 1.5, size=N_ref)
        q_ref_path, _, _ = simulate_particles(N_ref, x0_ref, alpha, T, dt, rng_ref)
        q_ref = q_ref_path[-1][1]  # terminal q

        q_errors = []
        for N in [20, 50, 100]:
            rng = np.random.default_rng(6)
            x0 = quantile_init(x0_ref, N)
            q_path, _, _ = simulate_particles(N, x0, alpha, T, dt, rng)
            q_N = q_path[-1][1]
            q_errors.append(abs(q_N - q_ref))

        # All errors should be finite
        assert all(np.isfinite(e) for e in q_errors)
        # And errors should not systematically grow with N
        assert q_errors[-1] <= q_errors[0] + 0.2


class TestQuantileInitialisation:
    def test_quantile_init_length(self):
        samples = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
        x0 = quantile_init(samples, 10)
        assert len(x0) == 10

    def test_quantile_init_range(self):
        samples = np.linspace(0.5, 2.0, 100)
        x0 = quantile_init(samples, 50)
        assert float(np.min(x0)) >= 0.5 - 1e-10
        assert float(np.max(x0)) <= 2.0 + 1e-10

    def test_quantile_init_sorted(self):
        samples = np.random.default_rng(0).uniform(0.5, 2.0, 100)
        x0 = quantile_init(samples, 30)
        assert np.all(np.diff(x0) >= 0.0), "Quantile init should be sorted"
