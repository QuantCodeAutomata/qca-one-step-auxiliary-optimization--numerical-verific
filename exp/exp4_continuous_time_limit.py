"""
Experiment 4: Continuous-Time Reflected Mean-Field Limit.

Route 1 — Monte Carlo validation:
  Generates the optimal discrete boundary Lambda_t from the finest-Delta run
  (Delta=0.025) and validates the mean-field balance
      E[L_t] ≈ alpha * P(tau <= t)
  via Monte Carlo simulation of N_mc reflected Brownian paths.

Route 2 — Simple heat-equation PDE (approximation):
  Solves u_t = 0.5 u_xx on [0, Lambda_max] with absorbing boundary at 0
  and Dirichlet condition at Lambda_max.  Default flux is integrated to give
  q_pde(t), compared qualitatively with the Monte Carlo default fraction.

Outputs
-------
results/exp4_results.csv
results/exp4_balance_check.png
results/exp4_density_evolution.png
"""

import sys
import os

sys.path.insert(0, '/workspace/project')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.cutoff_operator import compute_threshold
from src.absorbed_brownian import evolve_law_gridded
from src.welfare import evaluate_welfare, compute_mean, PAYOFF_FUNCTIONS

os.makedirs('/workspace/project/results', exist_ok=True)


# ── Fixed-grid helper (shared with exp2/3) ────────────────────────────────────

def apply_cutoff_fixed(x_base: np.ndarray, w: np.ndarray, kappa: float) -> np.ndarray:
    new_w = w.copy()
    above = x_base > kappa
    cap_mass = float(np.sum(new_w[above]))
    new_w[above] = 0.0
    idx_kappa = int(np.searchsorted(x_base, kappa, side='right') - 1)
    idx_kappa = max(0, min(idx_kappa, len(x_base) - 1))
    new_w[idx_kappa] += cap_mass
    return new_w


def ffill_nan(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values; leading NaNs remain NaN."""
    out = arr.copy()
    last = np.nan
    for i in range(len(out)):
        if not np.isnan(out[i]):
            last = out[i]
        elif not np.isnan(last):
            out[i] = last
    return out


# ── Discrete optimal runner (generates Lambda_t proxy) ────────────────────────

def generate_lambda_proxy(
    x_base: np.ndarray,
    w0: np.ndarray,
    atom0: float,
    T: float,
    alpha: float,
    Delta: float,
) -> tuple:
    """
    Run discrete optimal (cutoff policy) and return piecewise-constant Lambda.

    Returns
    -------
    t_steps  : (M+1,) array of step times
    kappas_ff: (M+1,) forward-filled kappa values (the boundary Lambda_t^Delta)
    w_history: list of (M+1) positive-state weight arrays
    q_history: (M+1,) default fractions
    """
    M = round(T / Delta)
    t_steps = np.arange(M + 1) * Delta

    w_cur = w0.copy()
    atom_cur = float(atom0)
    kappas = np.full(M + 1, np.nan)
    q_history = np.empty(M + 1)
    w_history = [None] * (M + 1)

    q_history[0] = atom_cur
    w_history[0] = w_cur.copy()

    for n in range(M):
        _, w_pre, atom_pre = evolve_law_gridded(x_base, w_cur, atom_cur, Delta)
        c = alpha * (atom_pre - atom_cur)

        if c > 1e-12:
            try:
                kappa = compute_threshold(x_base, w_pre, c)
                w_cur = apply_cutoff_fixed(x_base, w_pre, kappa)
                kappas[n + 1] = kappa
            except Exception as exc:
                print(f"  [Lambda proxy] step {n}: {exc}")
                w_cur = w_pre.copy()
        else:
            w_cur = w_pre.copy()

        atom_cur = atom_pre
        q_history[n + 1] = atom_cur
        w_history[n + 1] = w_cur.copy()

    return t_steps, ffill_nan(kappas), w_history, q_history


# ── Route 1: Monte Carlo simulation ───────────────────────────────────────────

def monte_carlo_validation(
    t_steps_lam: np.ndarray,
    kappas_ff: np.ndarray,
    alpha: float,
    T: float,
    N_mc: int = 5000,
    M_fine: int = 400,
    seed: int = 0,
) -> dict:
    """
    Simulate N_mc reflected Brownian paths and validate the balance
    E[L_t] ≈ alpha * P(tau <= t).

    The boundary Lambda(t) is given piecewise-constant by kappas_ff on t_steps_lam.

    Returns
    -------
    dict with keys: t_rep, E_Lt, alpha_Ptau, discrepancy, q_mc
    """
    rng = np.random.default_rng(seed)
    dt = T / M_fine
    t_mc = np.linspace(0.0, T, M_fine + 1)

    # Lambda at each fine time step (piecewise-constant, left-continuous)
    Lambda_fine = np.interp(t_mc, t_steps_lam, kappas_ff)
    # Replace any remaining NaN (leading) with a large value (no reflection)
    Lambda_fine = np.where(np.isnan(Lambda_fine), 1e10, Lambda_fine)

    # Initial states: uniform[0.5, 2.0] via inverse CDF
    u = rng.uniform(0.0, 1.0, N_mc)
    X0 = 0.5 + 1.5 * u   # shape (N_mc,)

    # Brownian increments and cumulative paths
    dB = np.sqrt(dt) * rng.standard_normal((N_mc, M_fine))
    B = np.concatenate(
        [np.zeros((N_mc, 1)), np.cumsum(dB, axis=1)], axis=1
    )  # shape (N_mc, M_fine+1)

    X_free = X0[:, None] + B   # free BM paths, shape (N_mc, M_fine+1)

    # Loss process: L_t = running sup of (X_free_s - Lambda_s)^+
    excess = X_free - Lambda_fine[None, :]              # shape (N_mc, M_fine+1)
    excess_pos = np.maximum(excess, 0.0)
    L = np.maximum.accumulate(excess_pos, axis=1)       # shape (N_mc, M_fine+1)

    # Controlled state and default times
    X_ctrl = X_free - L                                 # shape (N_mc, M_fine+1)

    # Default: first time X_ctrl <= 0
    tau = np.full(N_mc, np.inf)
    below_zero = X_ctrl <= 0.0
    first_default_idx = np.argmax(below_zero, axis=1)   # shape (N_mc,)
    ever_defaults = below_zero[np.arange(N_mc), first_default_idx]
    tau[ever_defaults] = t_mc[first_default_idx[ever_defaults]]

    # Validate: max active state at each step <= Lambda + tolerance
    max_ctrl = np.max(X_ctrl, axis=0)   # per time step
    tol_ctrl = 0.1 + dt                 # small discretisation allowance
    print("  Test: controlled state stays below Lambda + tolerance")
    violations = np.where(max_ctrl > Lambda_fine + tol_ctrl)[0]
    if len(violations) > 0:
        print(f"  WARNING: {len(violations)} time steps exceed Lambda + {tol_ctrl:.3f}")
    else:
        print("  PASS")

    # Compute metrics at common reporting times
    report_times = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rep_idx = [int(np.argmin(np.abs(t_mc - t))) for t in report_times]

    E_Lt = []
    alpha_Ptau = []
    q_mc = []

    for k, idx in enumerate(rep_idx):
        t = report_times[k]
        el = float(np.mean(L[:, idx]))
        ptau = float(np.mean(tau <= t))
        E_Lt.append(el)
        alpha_Ptau.append(alpha * ptau)
        q_mc.append(ptau)

    discrepancy = [abs(e - a) for e, a in zip(E_Lt, alpha_Ptau)]

    print(f"  Max |E[L_t] - alpha*P(tau<=t)|: {max(discrepancy):.4f}")
    print("  Test: balance error < 0.1 at all reporting times")
    for k, t in enumerate(report_times):
        print(
            f"    t={t:.1f}:  E[L_t]={E_Lt[k]:.4f}  "
            f"alpha*P(tau)={alpha_Ptau[k]:.4f}  err={discrepancy[k]:.4f}"
        )
        assert discrepancy[k] < 0.1, (
            f"Balance error {discrepancy[k]:.4f} > 0.1 at t={t}"
        )
    print("  PASS")

    return {
        't_rep': report_times,
        'E_Lt': E_Lt,
        'alpha_Ptau': alpha_Ptau,
        'discrepancy': discrepancy,
        'q_mc': q_mc,
    }


# ── Route 2: FTCS heat equation on [0, Lambda_max] ────────────────────────────

def pde_heat_equation(
    Lambda_max: float,
    T: float,
    alpha: float,
    N_x: int = 100,
) -> dict:
    """
    Solve u_t = 0.5 u_xx on [0, Lambda_max] with Dirichlet BCs u=0 at both ends.
    Initial condition: uniform density on [0.5, 2.0] (clipped to grid domain).

    Returns
    -------
    dict with keys: x_grid, t_snapshots, u_snapshots, q_pde, t_pde
    """
    dx = Lambda_max / (N_x - 1)
    x_grid = np.linspace(0.0, Lambda_max, N_x)

    # Stable dt: r = 0.5 * dt / dx^2 <= 0.45
    r_target = 0.45
    dt_pde = r_target * dx ** 2 / 0.5
    n_steps = int(np.ceil(T / dt_pde))
    dt_pde = T / n_steps
    r = 0.5 * dt_pde / dx ** 2

    print(f"  PDE: N_x={N_x}  dx={dx:.4f}  dt={dt_pde:.6f}  r={r:.4f}  steps={n_steps}")

    # Initial condition: uniform on [0.5, 2.0] intersected with [0, Lambda_max]
    u = np.where((x_grid >= 0.5) & (x_grid <= 2.0), 1.0, 0.0).astype(float)
    total = np.trapz(u, x_grid)
    if total > 1e-12:
        u /= total
    u[0] = 0.0
    u[-1] = 0.0

    # Snapshot times
    snap_times = [0.0, 0.25, 0.5, 0.75, 1.0]
    snap_steps = [int(round(t / dt_pde)) for t in snap_times]
    u_snapshots = []
    t_snapshots = []

    # Cumulative default flux
    q_pde = np.zeros(n_steps + 1)

    step = 0
    if step in snap_steps:
        u_snapshots.append(u.copy())
        t_snapshots.append(0.0)

    for step in range(1, n_steps + 1):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2.0 * u[1:-1] + u[:-2])
        u_new[0] = 0.0
        u_new[-1] = 0.0
        u_new = np.maximum(u_new, 0.0)

        # Default flux: probability leaving at x=0
        flux = 0.5 * u[1] / dx if dx > 0 else 0.0
        q_pde[step] = q_pde[step - 1] + flux * dt_pde

        u = u_new
        if step in snap_steps:
            u_snapshots.append(u.copy())
            t_snapshots.append(step * dt_pde)

    t_pde = np.arange(n_steps + 1) * dt_pde
    return {
        'x_grid': x_grid,
        't_pde': t_pde,
        'q_pde': q_pde,
        't_snapshots': t_snapshots,
        'u_snapshots': u_snapshots,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs('/workspace/project/results', exist_ok=True)

    # Shared parameters
    T = 1.0
    alpha = 0.4
    Delta_fine = 0.025
    N_grid = 300
    x_base = np.linspace(0.01, 2.5, N_grid)

    # Initial law: uniform on [0.5, 2.0]
    mask = (x_base >= 0.5) & (x_base <= 2.0)
    w0 = np.where(mask, 1.0, 0.0).astype(float)
    w0 /= w0.sum()
    atom0 = 0.0

    # ── Generate Lambda_t proxy ───────────────────────────────────────────────
    print("=" * 60)
    print("Exp 4 — Generating Lambda_t proxy (Delta=0.025) ...")
    t_steps_lam, kappas_ff, w_history, q_history = generate_lambda_proxy(
        x_base, w0, atom0, T, alpha, Delta_fine
    )
    Lambda_max_obs = float(np.nanmax(kappas_ff))
    print(f"Lambda range: [{float(np.nanmin(kappas_ff)):.3f}, {Lambda_max_obs:.3f}]")
    print(f"Terminal q (discrete proxy): {q_history[-1]:.4f}")

    # ── Route 1: Monte Carlo ──────────────────────────────────────────────────
    print("\n--- Route 1: Monte Carlo ---")
    mc_res = monte_carlo_validation(
        t_steps_lam, kappas_ff, alpha, T, N_mc=5000, M_fine=400, seed=42
    )

    # ── Route 2: PDE ──────────────────────────────────────────────────────────
    print("\n--- Route 2: PDE (FTCS heat equation) ---")
    Lambda_max_pde = max(2.0, Lambda_max_obs)
    pde_res = pde_heat_equation(Lambda_max_pde, T, alpha, N_x=100)

    # Interpolate q_pde to reporting times
    report_times = mc_res['t_rep']
    q_pde_at_rep = np.interp(report_times, pde_res['t_pde'], pde_res['q_pde'])

    max_disc = float(np.max(mc_res['discrepancy']))
    print(f"\nMax L_t vs alpha*P(tau<=t) discrepancy: {max_disc:.4f}")

    # ── Save results CSV ──────────────────────────────────────────────────────
    records = []
    for k, t in enumerate(report_times):
        records.append({
            'time': t,
            'E_Lt_mc': mc_res['E_Lt'][k],
            'alpha_Ptau_mc': mc_res['alpha_Ptau'][k],
            'discrepancy': mc_res['discrepancy'][k],
            'q_mc': mc_res['q_mc'][k],
            'q_pde': q_pde_at_rep[k],
        })
    df = pd.DataFrame(records)
    csv_path = '/workspace/project/results/exp4_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # ── Plot 1: balance check ─────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Exp 4 — Mean-Field Balance Check', fontsize=13)

    ax = axes[0]
    ax.plot(report_times, mc_res['E_Lt'], 'o-', color='steelblue', lw=2,
            label='$E[L_t]$ (MC)')
    ax.plot(report_times, mc_res['alpha_Ptau'], 's--', color='tomato', lw=2,
            label=r'$\alpha \cdot P(\tau \leq t)$ (MC)')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Value')
    ax.set_title('Route 1 (MC): $E[L_t]$ vs $\\alpha P(\\tau \\leq t)$')
    ax.legend(fontsize=9)
    ax.set_xlim(0, T)

    ax2 = axes[1]
    ax2.plot(report_times, mc_res['discrepancy'], 'D-', color='purple', lw=2,
             label='|discrepancy|')
    ax2.axhline(0.1, ls=':', color='gray', label='tolerance 0.1')
    ax2.set_xlabel('Time $t$')
    ax2.set_ylabel('|Error|')
    ax2.set_title('Balance Discrepancy Over Time')
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, T)

    plt.tight_layout()
    png1 = '/workspace/project/results/exp4_balance_check.png'
    plt.savefig(png1, dpi=120)
    plt.close()
    print(f"Saved {png1}")

    # ── Plot 2: PDE density snapshots ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Exp 4 — Route 2: PDE Density Evolution', fontsize=13)

    ax_dens = axes[0]
    cmap = plt.cm.viridis
    n_snaps = len(pde_res['u_snapshots'])
    for k, (t_s, u_s) in enumerate(
        zip(pde_res['t_snapshots'], pde_res['u_snapshots'])
    ):
        col = cmap(k / max(n_snaps - 1, 1))
        ax_dens.plot(pde_res['x_grid'], u_s, color=col, lw=1.8,
                     label=f't = {t_s:.2f}')
    ax_dens.set_xlabel('State $x$')
    ax_dens.set_ylabel('Density $u(t, x)$')
    ax_dens.set_title('PDE Density Snapshots')
    ax_dens.legend(fontsize=8)

    ax_q = axes[1]
    # Decimate q_pde for plotting (too many points)
    stride = max(1, len(pde_res['t_pde']) // 500)
    ax_q.plot(
        pde_res['t_pde'][::stride], pde_res['q_pde'][::stride],
        color='darkorange', lw=2, label='$q^{PDE}(t)$'
    )
    ax_q.plot(report_times, mc_res['q_mc'], 's--', color='steelblue', lw=2,
              label='$q^{MC}(t)$')
    ax_q.set_xlabel('Time $t$')
    ax_q.set_ylabel('Default fraction $q(t)$')
    ax_q.set_title('PDE vs MC default fraction')
    ax_q.legend(fontsize=9)
    ax_q.set_xlim(0, T)

    plt.tight_layout()
    png2 = '/workspace/project/results/exp4_density_evolution.png'
    plt.savefig(png2, dpi=120)
    plt.close()
    print(f"Saved {png2}")

    print(f"\nMax L_t vs alpha*P(tau<=t) discrepancy: {max_disc:.4f}")


if __name__ == '__main__':
    main()
