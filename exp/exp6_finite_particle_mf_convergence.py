"""
Experiment 6: Finite-Particle to Mean-Field Convergence Under Cutoff Control.

For particle counts N in {10, 25, 50, 100, 200} and n_seeds=3 random seeds,
simulates N interacting Brownian particles with water-filling cap reallocation
triggered by each new default event.  Compares the empirical measure to the
mean-field (infinite-population) limit generated via the discrete optimal
recursion (Delta=0.005).

Metrics per (N, seed) at reporting times {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}:
  - q^N_t   : empirical default fraction
  - mean^N_t: mean of active particle states
  - upper_edge: maximum active state
  - W1      : 1-Wasserstein between empirical and mean-field distributions

Expected: W1 and |q^N - q^mf| decrease as N increases (O(1/sqrt(N)) rate).

Outputs
-------
results/exp6_results.csv
results/exp6_convergence.csv
results/exp6_convergence.png
results/exp6_paths.png
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

from src.cutoff_operator import compute_threshold, water_filling_cap
from src.absorbed_brownian import evolve_law_gridded
from src.welfare import evaluate_welfare, compute_mean, PAYOFF_FUNCTIONS

os.makedirs('/workspace/project/results', exist_ok=True)


# ── Fixed-grid helper ─────────────────────────────────────────────────────────

def apply_cutoff_fixed(x_base: np.ndarray, w: np.ndarray, kappa: float) -> np.ndarray:
    new_w = w.copy()
    above = x_base > kappa
    cap_mass = float(np.sum(new_w[above]))
    new_w[above] = 0.0
    idx_kappa = int(np.searchsorted(x_base, kappa, side='right') - 1)
    idx_kappa = max(0, min(idx_kappa, len(x_base) - 1))
    new_w[idx_kappa] += cap_mass
    return new_w


# ── Mean-field target generation ──────────────────────────────────────────────

def generate_mf_target(
    x_base: np.ndarray,
    w0: np.ndarray,
    T: float,
    alpha: float,
    Delta: float,
    report_times: list,
) -> dict:
    """
    Run discrete optimal (cutoff policy) on x_base and collect snapshots at
    each reporting time.

    Returns dict with keys:
        q       : (n_rep,) mean-field default fractions
        mean    : (n_rep,) mean-field positive-state means
        w       : list of (n_rep,) weight arrays
        Lambda  : (n_rep,) forward-filled cutoff boundaries
    """
    M = round(T / Delta)
    t_steps = np.arange(M + 1) * Delta

    w_cur = w0.copy()
    atom_cur = 0.0

    q_path = np.empty(M + 1)
    mean_path = np.empty(M + 1)
    kappa_path = np.full(M + 1, np.nan)
    w_path = [None] * (M + 1)

    q_path[0] = 0.0
    mean_path[0] = compute_mean(x_base, w_cur)
    w_path[0] = w_cur.copy()

    for n in range(M):
        _, w_pre, atom_pre = evolve_law_gridded(x_base, w_cur, atom_cur, Delta)
        c = alpha * (atom_pre - atom_cur)

        if c > 1e-12:
            try:
                kappa = compute_threshold(x_base, w_pre, c)
                w_cur = apply_cutoff_fixed(x_base, w_pre, kappa)
                kappa_path[n + 1] = kappa
            except Exception as exc:
                print(f"  [MF target] step {n}: {exc}")
                w_cur = w_pre.copy()
        else:
            w_cur = w_pre.copy()

        atom_cur = atom_pre
        q_path[n + 1] = atom_cur
        mean_path[n + 1] = compute_mean(x_base, w_cur)
        w_path[n + 1] = w_cur.copy()

    # Forward-fill kappa
    kappa_ff = kappa_path.copy()
    last = np.nan
    for i in range(len(kappa_ff)):
        if not np.isnan(kappa_ff[i]):
            last = kappa_ff[i]
        elif not np.isnan(last):
            kappa_ff[i] = last

    # Collect at reporting times
    rep_idx = [int(np.argmin(np.abs(t_steps - t))) for t in report_times]
    return {
        'q':      np.array([q_path[i] for i in rep_idx]),
        'mean':   np.array([mean_path[i] for i in rep_idx]),
        'w':      [w_path[i].copy() for i in rep_idx],
        'Lambda': np.array([kappa_ff[i] for i in rep_idx]),
    }


# ── N-particle simulation ──────────────────────────────────────────────────────

def simulate_particles(
    N: int,
    T: float,
    alpha: float,
    dt: float,
    report_times: list,
    seed: int,
) -> dict:
    """
    Simulate N Brownian particles with water-filling reallocation on defaults.

    Initial states: quantile-based uniform[0.5, 2.0] plus small noise.

    Parameters
    ----------
    N            : number of particles
    T, alpha, dt : time horizon, mean-field coupling, Euler-Maruyama step
    report_times : list of floats at which to record statistics
    seed         : RNG seed (noise scale = seed * 0.001)

    Returns
    -------
    dict with arrays of shape (n_rep,):
        q_N, mean_N, upper_edge
        x_active_list: list of active-particle state arrays at each report time
    """
    rng = np.random.default_rng(seed + 1000)  # reproducible per (N, seed)

    # Quantile-based initialisation
    i_vals = np.arange(1, N + 1)
    x = 0.5 + 1.5 * (i_vals - 0.5) / N   # uniform[0.5, 2.0] quantiles

    # Small perturbation (scale with seed so seed=0 is unperturbed)
    if seed > 0:
        noise = seed * 0.001 * rng.standard_normal(N)
        x = np.clip(x + noise, 1e-6, None)

    active = np.ones(N, dtype=bool)   # True = not yet defaulted

    n_steps = round(T / dt)
    rep_idx = [int(round(t / dt)) for t in report_times]

    q_N_list = []
    mean_N_list = []
    upper_edge_list = []
    x_active_list = []

    # Record initial state
    def record():
        n_active = int(np.sum(active))
        q_val = 1.0 - n_active / N
        if n_active > 0:
            x_act = x[active]
            mean_val = float(np.mean(x_act))
            upper_val = float(np.max(x_act))
        else:
            x_act = np.array([])
            mean_val = 0.0
            upper_val = 0.0
        q_N_list.append(q_val)
        mean_N_list.append(mean_val)
        upper_edge_list.append(upper_val)
        x_active_list.append(x_act.copy())

    if 0 in rep_idx:
        record()
    rep_set = set(rep_idx)

    cascade_flag = False
    for step in range(1, n_steps + 1):
        if cascade_flag:
            break

        # a) Brownian increment for active particles
        n_active = int(np.sum(active))
        if n_active == 0:
            break
        x[active] += np.sqrt(dt) * rng.standard_normal(n_active)

        # b) Detect new defaults
        newly_defaulted = active & (x <= 0.0)
        r_new = int(np.sum(newly_defaulted))

        if r_new > 0:
            active[newly_defaulted] = False
            x[newly_defaulted] = 0.0

            # c) Apply water-filling cap to surviving active particles
            survivors_mask = active   # still active after this batch of defaults
            n_surv = int(np.sum(survivors_mask))
            if n_surv > 0:
                y = x[survivors_mask]
                # c_total = total loss to redistribute (sum-based for water_filling_cap)
                c_total = alpha * r_new
                if c_total < float(np.sum(y)) - 1e-12:
                    try:
                        z_caps, _ = water_filling_cap(y, c_total)
                        x[survivors_mask] = z_caps
                    except Exception as exc:
                        print(
                            f"  [N={N}, seed={seed}] step {step}: "
                            f"water_filling_cap failed: {exc}"
                        )
                else:
                    print(
                        f"  [N={N}, seed={seed}] step {step}: "
                        f"terminal cascade (c_total={c_total:.4f} >= sum(y)={np.sum(y):.4f})"
                    )
                    cascade_flag = True

        if step in rep_set:
            record()

    # If cascade ended early, pad remaining report times with last values
    while len(q_N_list) < len(report_times):
        q_N_list.append(q_N_list[-1] if q_N_list else 1.0)
        mean_N_list.append(mean_N_list[-1] if mean_N_list else 0.0)
        upper_edge_list.append(upper_edge_list[-1] if upper_edge_list else 0.0)
        x_active_list.append(x_active_list[-1] if x_active_list else np.array([]))

    return {
        'q_N':          np.array(q_N_list),
        'mean_N':       np.array(mean_N_list),
        'upper_edge':   np.array(upper_edge_list),
        'x_active_list': x_active_list,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs('/workspace/project/results', exist_ok=True)

    # Parameters
    N_list = [10, 25, 50, 100, 200]
    T = 0.5
    alpha = 0.3
    dt = 0.01
    n_seeds = 3
    Delta_mf = 0.005
    report_times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    x_base_mf = np.linspace(0.01, 2.5, 400)
    mask = (x_base_mf >= 0.5) & (x_base_mf <= 2.0)
    w0_mf = np.where(mask, 1.0, 0.0).astype(float)
    w0_mf /= w0_mf.sum()

    # ── Generate mean-field target ────────────────────────────────────────────
    print("=" * 60)
    print("Exp 6 — Finite-Particle to Mean-Field Convergence")
    print(f"Generating MF target (Delta={Delta_mf}) ...")
    mf = generate_mf_target(x_base_mf, w0_mf, T, alpha, Delta_mf, report_times)
    print(
        f"MF target: q_T={mf['q'][-1]:.4f}  mean_T={mf['mean'][-1]:.4f}"
        f"  Lambda_T={mf['Lambda'][-1]:.4f}"
    )

    # ── Particle simulations ──────────────────────────────────────────────────
    all_records = []
    agg_by_N = {N: {'W1': [], 'dq': [], 'dmean': []} for N in N_list}

    for N in N_list:
        print(f"\n--- N = {N} ---")
        for seed in range(n_seeds):
            sim = simulate_particles(N, T, alpha, dt, report_times, seed)

            for k, t in enumerate(report_times):
                # Wasserstein distance between empirical and mean-field
                x_act = sim['x_active_list'][k]
                w_mf_k = mf['w'][k]
                s_mf = float(np.sum(w_mf_k))
                W1 = np.nan
                if len(x_act) > 0 and s_mf > 1e-10:
                    try:
                        W1 = wasserstein_distance(
                            x_act, x_base_mf,
                            u_weights=np.ones(len(x_act)) / len(x_act),
                            v_weights=w_mf_k / s_mf,
                        )
                    except Exception as exc:
                        print(f"  W1 failed N={N} seed={seed} t={t}: {exc}")

                q_mf_k = float(mf['q'][k])
                dq = abs(float(sim['q_N'][k]) - q_mf_k)
                dmean = abs(float(sim['mean_N'][k]) - float(mf['mean'][k]))

                all_records.append({
                    'N': N,
                    'seed': seed,
                    'time': t,
                    'q_N': sim['q_N'][k],
                    'q_mf': q_mf_k,
                    'delta_q': dq,
                    'W1': W1,
                    'mean_N': sim['mean_N'][k],
                    'mean_mf': mf['mean'][k],
                    'delta_mean': dmean,
                    'upper_edge_N': sim['upper_edge'][k],
                    'Lambda_mf': mf['Lambda'][k],
                })

                if t == T:   # terminal time: accumulate for aggregation
                    if not np.isnan(W1):
                        agg_by_N[N]['W1'].append(W1)
                    agg_by_N[N]['dq'].append(dq)
                    agg_by_N[N]['dmean'].append(dmean)

            print(
                f"  seed={seed}  q_T={sim['q_N'][-1]:.3f}  "
                f"mean_T={sim['mean_N'][-1]:.3f}  "
                f"upper={sim['upper_edge'][-1]:.3f}  "
                f"W1_T={all_records[-1]['W1']:.4f}"
                if not np.isnan(all_records[-1]['W1']) else
                f"  seed={seed}  q_T={sim['q_N'][-1]:.3f}  W1=NaN"
            )

    # ── Aggregate over seeds ──────────────────────────────────────────────────
    conv_records = []
    for N in N_list:
        W1_arr = np.array(agg_by_N[N]['W1'])
        dq_arr = np.array(agg_by_N[N]['dq'])
        dm_arr = np.array(agg_by_N[N]['dmean'])

        mean_W1 = float(np.nanmean(W1_arr)) if len(W1_arr) > 0 else np.nan
        std_W1 = float(np.nanstd(W1_arr)) if len(W1_arr) > 0 else np.nan
        mean_dq = float(np.mean(dq_arr)) if len(dq_arr) > 0 else np.nan
        mean_dm = float(np.mean(dm_arr)) if len(dm_arr) > 0 else np.nan

        conv_records.append({
            'N': N,
            'mean_W1': mean_W1,
            'std_W1': std_W1,
            'mean_delta_q': mean_dq,
            'mean_delta_mean': mean_dm,
        })
        print(f"N={N:4d}  mean W1={mean_W1:.4f}  mean |dq|={mean_dq:.4f}")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    df_res = pd.DataFrame(all_records)
    csv1 = '/workspace/project/results/exp6_results.csv'
    df_res.to_csv(csv1, index=False)
    print(f"\nSaved {csv1}")

    df_conv = pd.DataFrame(conv_records)
    csv2 = '/workspace/project/results/exp6_convergence.csv'
    df_conv.to_csv(csv2, index=False)
    print(f"Saved {csv2}")

    # ── Plot 1: log-log convergence of W1 vs N ────────────────────────────────
    N_vals = [r['N'] for r in conv_records]
    W1_vals = [r['mean_W1'] for r in conv_records]
    std_vals = [r['std_W1'] for r in conv_records]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(
        N_vals, W1_vals, yerr=std_vals,
        fmt='o-', color='steelblue', lw=2, capsize=4, label='mean $W_1$ (± std)'
    )
    # Reference slope -1/2
    N_ref = np.array([10.0, 200.0])
    ref_W1 = W1_vals[0] * np.sqrt(N_vals[0] / N_ref)
    ax.plot(N_ref, ref_W1, 'k--', lw=1.2, label='$O(N^{-1/2})$ reference')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Number of particles $N$')
    ax.set_ylabel('Mean $W_1$ (terminal time)')
    ax.set_title('Exp 6 — Finite-Particle to Mean-Field Convergence')
    ax.legend(fontsize=9)
    plt.tight_layout()
    png1 = '/workspace/project/results/exp6_convergence.png'
    plt.savefig(png1, dpi=120)
    plt.close()
    print(f"Saved {png1}")

    # ── Plot 2: 2×2 path comparison ───────────────────────────────────────────
    # Collect seed-0 results for each N for clean trajectory plots
    df_s0 = df_res[df_res['seed'] == 0].copy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        'Exp 6 — Particle Trajectories (seed=0) vs Mean-Field Target', fontsize=13
    )

    cmap_N = plt.cm.cool
    colors_N = {N: cmap_N(i / (len(N_list) - 1)) for i, N in enumerate(N_list)}

    # (0,0): q^N_t for different N + q^mf
    ax00 = axes[0, 0]
    for N in N_list:
        sub = df_s0[df_s0['N'] == N]
        ax00.plot(sub['time'], sub['q_N'], color=colors_N[N], lw=1.5,
                  label=f'N={N}')
    ax00.plot(report_times, mf['q'], 'k-', lw=2.5, label='MF target')
    ax00.set_title('Default fraction $q^N_t$')
    ax00.set_xlabel('Time')
    ax00.set_ylabel('$q^N_t$')
    ax00.legend(fontsize=7, ncol=2)

    # (0,1): mean^N_t for different N + mean^mf
    ax01 = axes[0, 1]
    for N in N_list:
        sub = df_s0[df_s0['N'] == N]
        ax01.plot(sub['time'], sub['mean_N'], color=colors_N[N], lw=1.5,
                  label=f'N={N}')
    ax01.plot(report_times, mf['mean'], 'k-', lw=2.5, label='MF target')
    ax01.set_title('Mean of active states')
    ax01.set_xlabel('Time')
    ax01.set_ylabel('Mean')
    ax01.legend(fontsize=7, ncol=2)

    # (1,0): upper edge vs Lambda^mf
    ax10 = axes[1, 0]
    for N in N_list:
        sub = df_s0[df_s0['N'] == N]
        ax10.plot(sub['time'], sub['upper_edge_N'], color=colors_N[N], lw=1.5,
                  label=f'N={N}')
    valid_lam = ~np.isnan(mf['Lambda'])
    if np.any(valid_lam):
        ax10.plot(
            np.array(report_times)[valid_lam],
            mf['Lambda'][valid_lam],
            'k--', lw=2, label='$\\Lambda^{mf}$'
        )
    ax10.set_title('Upper edge vs MF boundary $\\Lambda_t$')
    ax10.set_xlabel('Time')
    ax10.set_ylabel('Upper state / $\\Lambda$')
    ax10.legend(fontsize=7, ncol=2)

    # (1,1): W1 path over time (all N, seed=0)
    ax11 = axes[1, 1]
    for N in N_list:
        sub = df_s0[df_s0['N'] == N]
        ax11.plot(sub['time'], sub['W1'], color=colors_N[N], lw=1.5,
                  label=f'N={N}')
    ax11.set_title('$W_1$ to mean-field over time')
    ax11.set_xlabel('Time')
    ax11.set_ylabel('$W_1$')
    ax11.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    png2 = '/workspace/project/results/exp6_paths.png'
    plt.savefig(png2, dpi=120)
    plt.close()
    print(f"Saved {png2}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Summary ===")
    for r in conv_records:
        print(f"N={r['N']:4d}  mean W1={r['mean_W1']:.4f}  std W1={r['std_W1']:.4f}")


if __name__ == '__main__':
    main()
