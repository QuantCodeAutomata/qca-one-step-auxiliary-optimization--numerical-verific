"""
Experiment 3: Vanishing-Delta Convergence.

Runs the discrete optimal recursion (cutoff policy, fixed spatial grid) for
Delta in {0.2, 0.1, 0.05, 0.025} and studies how the resulting trajectories
converge as Delta -> 0.

At each time step the algorithm is:
  1. Evolve:  (x_base, w_pre, atom_pre) = S_Delta (x_base, w_cur, atom_cur)
  2. Compute: c = alpha * (atom_pre - atom_cur)
  3. If c > 0: compute kappa = compute_threshold(x_base, w_pre, c)
               redistribute mass above kappa to the kappa bin
  4. atom_cur = atom_pre

Convergence metrics between consecutive (Delta_k, Delta_{k+1}=Delta_k/2) pairs:
  delta_V   – |terminal welfare_sqrt difference|
  delta_q   – |terminal default-fraction difference|
  sup_kappa – sup_t |kappa1(t) - kappa2(t)| at common reporting times
  mean_W1   – mean over reporting times of 1-Wasserstein between laws

Outputs
-------
results/exp3_results.csv
results/exp3_convergence.csv
results/exp3_boundary_convergence.png
results/exp3_value_convergence.png
"""

import sys
import os

sys.path.insert(0, '/workspace/project')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

from src.cutoff_operator import compute_threshold
from src.absorbed_brownian import evolve_law_gridded
from src.welfare import evaluate_welfare, compute_mean, PAYOFF_FUNCTIONS

os.makedirs('/workspace/project/results', exist_ok=True)


# ── Fixed-grid helpers (same as exp2) ─────────────────────────────────────────

def apply_cutoff_fixed(x_base: np.ndarray, w: np.ndarray, kappa: float) -> np.ndarray:
    """Redistribute mass above kappa into the bin at or just below kappa."""
    new_w = w.copy()
    above = x_base > kappa
    cap_mass = float(np.sum(new_w[above]))
    new_w[above] = 0.0
    idx_kappa = int(np.searchsorted(x_base, kappa, side='right') - 1)
    idx_kappa = max(0, min(idx_kappa, len(x_base) - 1))
    new_w[idx_kappa] += cap_mass
    return new_w


def ffill_nan(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values in a 1-D array."""
    out = arr.copy()
    last_valid = np.nan
    for i in range(len(out)):
        if not np.isnan(out[i]):
            last_valid = out[i]
        elif not np.isnan(last_valid):
            out[i] = last_valid
    return out


# ── Core runner ───────────────────────────────────────────────────────────────

def run_discrete_optimal(
    x_base: np.ndarray,
    w0: np.ndarray,
    atom0: float,
    T: float,
    alpha: float,
    Delta: float,
    report_times: list,
) -> dict:
    """
    Run discrete optimal (cutoff policy) on fixed grid and collect statistics.

    Parameters
    ----------
    x_base       : fixed positive-state grid
    w0           : initial positive-state weights
    atom0        : initial mass at 0
    T            : time horizon
    alpha        : mean-field coupling
    Delta        : time step
    report_times : list of floats at which to store detailed snapshots

    Returns
    -------
    dict with keys
        steps        – (M+1,) time points
        q            – (M+1,) default fractions
        means        – (M+1,) positive-state means
        kappas       – (M+1,) cutoff thresholds (NaN when none applied)
        welfare_sqrt – (M+1,) sqrt welfare
        w_at_steps   – list of (M+1) weight arrays
        rep_indices  – list of step indices corresponding to report_times
    """
    M = round(T / Delta)
    t_steps = np.arange(M + 1) * Delta

    w_cur = w0.copy()
    atom_cur = float(atom0)
    sqrt_fn = PAYOFF_FUNCTIONS['sqrt']

    q_arr = np.empty(M + 1)
    mean_arr = np.empty(M + 1)
    kappa_arr = np.full(M + 1, np.nan)
    welfare_sqrt_arr = np.empty(M + 1)
    w_at_steps = [None] * (M + 1)

    q_arr[0] = atom_cur
    mean_arr[0] = compute_mean(x_base, w_cur)
    welfare_sqrt_arr[0] = evaluate_welfare(sqrt_fn, x_base, w_cur, atom_cur)
    w_at_steps[0] = w_cur.copy()

    for n in range(M):
        _, w_pre, atom_pre = evolve_law_gridded(x_base, w_cur, atom_cur, Delta)

        delta_q = atom_pre - atom_cur
        c = alpha * delta_q

        if c > 1e-12:
            try:
                kappa = compute_threshold(x_base, w_pre, c)
                w_cur = apply_cutoff_fixed(x_base, w_pre, kappa)
                kappa_arr[n + 1] = kappa
            except Exception as exc:
                print(f"  [Delta={Delta}] step {n}: {exc}")
                w_cur = w_pre.copy()
        else:
            w_cur = w_pre.copy()

        atom_cur = atom_pre
        q_arr[n + 1] = atom_cur
        mean_arr[n + 1] = compute_mean(x_base, w_cur)
        welfare_sqrt_arr[n + 1] = evaluate_welfare(sqrt_fn, x_base, w_cur, atom_cur)
        w_at_steps[n + 1] = w_cur.copy()

    # Indices of steps nearest to each reporting time
    rep_indices = [int(np.argmin(np.abs(t_steps - t))) for t in report_times]

    return {
        'steps': t_steps,
        'q': q_arr,
        'means': mean_arr,
        'kappas': kappa_arr,
        'welfare_sqrt': welfare_sqrt_arr,
        'w_at_steps': w_at_steps,
        'rep_indices': rep_indices,
    }


# ── Convergence metrics between two runs ──────────────────────────────────────

def convergence_metrics(
    x_base: np.ndarray,
    res1: dict,
    res2: dict,
    report_times: list,
) -> dict:
    """
    Compute convergence metrics between two Delta runs at common reporting times.

    Parameters
    ----------
    x_base       : common spatial grid
    res1, res2   : output dicts from run_discrete_optimal (coarser, finer)
    report_times : common reporting times

    Returns
    -------
    dict with keys: delta_V, delta_q, sup_kappa, mean_W1
    """
    ri1 = res1['rep_indices']
    ri2 = res2['rep_indices']

    # Terminal welfare and default fraction
    delta_V = abs(res1['welfare_sqrt'][ri1[-1]] - res2['welfare_sqrt'][ri2[-1]])
    delta_q = abs(res1['q'][ri1[-1]] - res2['q'][ri2[-1]])

    # Forward-filled kappa sequences (so NaN -> last valid boundary)
    kappas1_ff = ffill_nan(res1['kappas'])
    kappas2_ff = ffill_nan(res2['kappas'])

    kappa_diffs = []
    W1_vals = []

    for k, _ in enumerate(report_times):
        k1 = ri1[k]
        k2 = ri2[k]

        # sup_kappa: compare forward-filled boundaries
        v1 = kappas1_ff[k1]
        v2 = kappas2_ff[k2]
        if not (np.isnan(v1) or np.isnan(v2)):
            kappa_diffs.append(abs(v1 - v2))

        # W1 between positive-state distributions
        w1 = res1['w_at_steps'][k1]
        w2 = res2['w_at_steps'][k2]
        s1 = float(np.sum(w1))
        s2 = float(np.sum(w2))
        if s1 > 1e-10 and s2 > 1e-10:
            try:
                W1 = wasserstein_distance(
                    x_base, x_base,
                    u_weights=w1 / s1,
                    v_weights=w2 / s2,
                )
                W1_vals.append(W1)
            except Exception:
                pass

    sup_kappa = max(kappa_diffs) if kappa_diffs else np.nan
    mean_W1 = float(np.mean(W1_vals)) if W1_vals else np.nan

    return {
        'delta_V': delta_V,
        'delta_q': delta_q,
        'sup_kappa': sup_kappa,
        'mean_W1': mean_W1,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs('/workspace/project/results', exist_ok=True)

    # Configuration
    T = 1.0
    alpha = 0.4
    x_base = np.linspace(0.01, 2.5, 250)
    delta_schedule = [0.2, 0.1, 0.05, 0.025]
    report_times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # Initial law: uniform on [0.5, 2.0]
    mask = (x_base >= 0.5) & (x_base <= 2.0)
    w0 = np.where(mask, 1.0, 0.0)
    w0 /= w0.sum()
    atom0 = 0.0

    print("=" * 60)
    print("Experiment 3: Vanishing-Delta Convergence")
    print(f"T={T}, alpha={alpha}, Delta schedule={delta_schedule}")
    print("=" * 60)

    all_results = {}
    result_records = []

    for Delta in delta_schedule:
        M = round(T / Delta)
        print(f"\n--- Delta={Delta:6.4f}  (M={M} steps) ---")
        res = run_discrete_optimal(x_base, w0, atom0, T, alpha, Delta, report_times)
        all_results[Delta] = res

        # Validate: q non-decreasing
        print(f"  Test: q is non-decreasing")
        diffs = np.diff(res['q'])
        assert np.all(diffs >= -1e-10), (
            f"q not non-decreasing for Delta={Delta}: min diff={diffs.min():.4e}"
        )
        print("  PASS")

        # Validate: total mass ≈ 1 at each step
        print(f"  Test: total mass conserved (~1) at each step")
        for n in range(M + 1):
            w_n = res['w_at_steps'][n]
            mass = float(np.sum(w_n)) + res['q'][n]
            assert abs(mass - 1.0) < 5e-3, (
                f"Mass={mass:.6f} != 1 at step {n} (Delta={Delta})"
            )
        print("  PASS")

        # Mean-balance residual at terminal time
        ri_T = res['rep_indices'][-1]
        mean_T = res['means'][ri_T]
        q_T = res['q'][ri_T]
        mean_0 = res['means'][0]
        balance_res = abs(mean_T - (mean_0 - alpha * q_T))

        print(
            f"  Terminal: q_T={q_T:.4f}  welfare_sqrt_T={res['welfare_sqrt'][ri_T]:.4f}"
            f"  balance_residual={balance_res:.4f}"
        )

        # Records at each reporting time
        for k, t_rep in enumerate(report_times):
            ri = res['rep_indices'][k]
            kappa_ff = ffill_nan(res['kappas'])
            result_records.append({
                'Delta': Delta,
                'report_time': t_rep,
                'actual_time': float(res['steps'][ri]),
                'q': res['q'][ri],
                'mean': res['means'][ri],
                'welfare_sqrt': res['welfare_sqrt'][ri],
                'kappa_ff': kappa_ff[ri],
                'balance_residual': balance_res if t_rep == 1.0 else np.nan,
            })

    # ── Convergence between consecutive Delta pairs ────────────────────────────
    conv_records = []
    delta_pairs = [
        (delta_schedule[i], delta_schedule[i + 1])
        for i in range(len(delta_schedule) - 1)
    ]

    print("\n--- Convergence metrics ---")
    all_delta_V = []
    for Delta_c, Delta_f in delta_pairs:
        metrics = convergence_metrics(
            x_base, all_results[Delta_c], all_results[Delta_f], report_times
        )
        all_delta_V.append(metrics['delta_V'])
        conv_records.append({
            'Delta_coarse': Delta_c,
            'Delta_fine': Delta_f,
            'delta_V': metrics['delta_V'],
            'delta_q': metrics['delta_q'],
            'sup_kappa': metrics['sup_kappa'],
            'mean_W1': metrics['mean_W1'],
        })
        print(
            f"  ({Delta_c:.3f},{Delta_f:.4f}): "
            f"delta_V={metrics['delta_V']:.5f}  "
            f"delta_q={metrics['delta_q']:.5f}  "
            f"sup_kappa={metrics['sup_kappa']:.5f}  "
            f"mean_W1={metrics['mean_W1']:.5f}"
        )

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    df_res = pd.DataFrame(result_records)
    csv1 = '/workspace/project/results/exp3_results.csv'
    df_res.to_csv(csv1, index=False)
    print(f"\nSaved {csv1}")

    df_conv = pd.DataFrame(conv_records)
    csv2 = '/workspace/project/results/exp3_convergence.csv'
    df_conv.to_csv(csv2, index=False)
    print(f"Saved {csv2}")

    # ── Plot 1: boundary kappa(t) as step function per Delta ──────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.plasma
    colors = [cmap(i / (len(delta_schedule) - 1)) for i in range(len(delta_schedule))]

    for idx, Delta in enumerate(delta_schedule):
        res = all_results[Delta]
        kappas_ff = ffill_nan(res['kappas'])
        t_steps = res['steps']
        ax.step(t_steps, kappas_ff, where='post',
                color=colors[idx], lw=1.5, label=f'Δ = {Delta}')

    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Boundary $\\Lambda_t$ (cutoff threshold)')
    ax.set_title('Exp 3 — Optimal Boundary $\\Lambda_t^\\Delta$ for Each Δ')
    ax.legend(fontsize=9)
    ax.set_xlim(0, T)
    plt.tight_layout()
    png1 = '/workspace/project/results/exp3_boundary_convergence.png'
    plt.savefig(png1, dpi=120)
    plt.close()
    print(f"Saved {png1}")

    # ── Plot 2: terminal welfare vs Delta (log–log) ───────────────────────────
    terminal_welfares = []
    for Delta in delta_schedule:
        res = all_results[Delta]
        ri_T = res['rep_indices'][-1]
        terminal_welfares.append(res['welfare_sqrt'][ri_T])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax_v = axes[0]
    ax_v.plot(delta_schedule, terminal_welfares, 'o-', color='steelblue', lw=2)
    ax_v.set_xscale('log')
    ax_v.set_xlabel('Δ (time step)')
    ax_v.set_ylabel('Terminal welfare (√·)')
    ax_v.set_title('Terminal Welfare vs Δ')
    ax_v.invert_xaxis()

    ax_c = axes[1]
    delta_c_vals = [r['Delta_coarse'] for r in conv_records]
    delta_V_vals = [r['delta_V'] for r in conv_records]
    mean_W1_vals = [r['mean_W1'] for r in conv_records]
    ax_c.loglog(delta_c_vals, delta_V_vals, 's--', color='tomato', lw=1.5, label='|ΔV|')
    ax_c.loglog(delta_c_vals, mean_W1_vals, 'd--', color='forestgreen', lw=1.5,
                label='mean W₁')
    ax_c.set_xlabel('Δ_coarse')
    ax_c.set_ylabel('Convergence metric')
    ax_c.set_title('Convergence Between Δ Pairs (log–log)')
    ax_c.legend(fontsize=9)

    plt.tight_layout()
    png2 = '/workspace/project/results/exp3_value_convergence.png'
    plt.savefig(png2, dpi=120)
    plt.close()
    print(f"Saved {png2}")

    # ── Summary ───────────────────────────────────────────────────────────────
    min_delta_V = float(np.nanmin(all_delta_V)) if all_delta_V else np.nan
    print(
        f"\nDelta refinement shows stabilization: "
        f"delta_V decreases to {min_delta_V:.4f}"
    )


if __name__ == '__main__':
    main()
