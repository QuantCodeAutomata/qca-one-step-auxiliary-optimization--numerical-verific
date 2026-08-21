"""
Experiment 5: Finite-Player Boundary Optimisation for Small n.

Validates the water-filling cap (Gamma_c) against CVXPY exact solutions and
brute-force search, across a range of small finite-player configurations.
Confirms permutation invariance of the objective and correct feasibility
detection.

Outputs
-------
results/exp5_results.csv
results/exp5_feasible_region.png
"""

import sys
import os

sys.path.insert(0, '/workspace/project')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.optimize import minimize, brentq

from src.cutoff_operator import water_filling_cap

# ── Helpers ───────────────────────────────────────────────────────────────────

def is_feasible(y, c, tol=1e-10):
    """Return True iff the problem (y, c) is strictly feasible."""
    return c < np.sum(y) - tol and np.all(np.array(y) > 0)


def solve_cvxpy(y, c, payoff_name='sqrt'):
    """
    Solve  max  f(z)  s.t.  0 < z <= y,  sum(z) = sum(y) - c
    via CVXPY (SCS solver).

    Returns
    -------
    np.ndarray or None
        Optimal z, or None if the solver fails.
    """
    y = np.array(y, dtype=float)
    m = len(y)
    z = cp.Variable(m)

    if payoff_name == 'sqrt':
        obj = cp.sum(cp.sqrt(z))
    elif payoff_name == 'log1p':
        obj = cp.sum(cp.log(1 + z))
    elif payoff_name == 'linear':
        obj = cp.sum(z)
    elif payoff_name == 'capped':
        K = 2.0
        obj = cp.sum(z) - cp.sum(cp.pos(z - K))
    else:
        obj = cp.sum(cp.sqrt(z))

    constraints = [z >= 1e-8, z <= y, cp.sum(z) == np.sum(y) - c]
    prob = cp.Problem(cp.Maximize(obj), constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if z.value is None:
        return None
    return np.array(z.value)


def brute_force_2d(y, c, payoff_fn, n_grid=400):
    """Exhaustive search for the m=2 case on a fine 1-D grid."""
    y1, y2 = float(y[0]), float(y[1])
    target  = y1 + y2 - c
    z1_vals = np.linspace(1e-6, y1 - 1e-6, n_grid)
    z2_vals = target - z1_vals
    mask    = (z2_vals > 1e-6) & (z2_vals < y2 - 1e-6)
    if not np.any(mask):
        return None
    z1_f = z1_vals[mask]
    z2_f = z2_vals[mask]
    obj  = payoff_fn(z1_f) + payoff_fn(z2_f)
    idx  = np.argmax(obj)
    return np.array([z1_f[idx], z2_f[idx]])


# ── Test configurations ───────────────────────────────────────────────────────

CONFIGS = [
    {'n': 2, 'y': [1.5, 0.5], 'alpha': 0.3, 'r': 1},
    {'n': 2, 'y': [1.0, 1.0], 'alpha': 0.3, 'r': 1},
    {'n': 2, 'y': [2.0, 0.3], 'alpha': 0.5, 'r': 1},
    {'n': 3, 'y': [2.0, 1.0, 0.5], 'alpha': 0.3, 'r': 1},
    {'n': 3, 'y': [1.5, 1.5, 1.5], 'alpha': 0.3, 'r': 1},
    {'n': 3, 'y': [3.0, 0.5, 0.5], 'alpha': 0.5, 'r': 2},
    {'n': 4, 'y': [2.0, 1.5, 1.0, 0.5], 'alpha': 0.3, 'r': 1},
    {'n': 4, 'y': [1.0, 1.0, 1.0, 1.0], 'alpha': 0.3, 'r': 2},
]

PAYOFFS = ['sqrt', 'log1p', 'linear']

# ── Test runner ───────────────────────────────────────────────────────────────

def run_tests():
    """Execute all test configurations; return list of result dicts."""
    results = []

    for cfg in CONFIGS:
        y     = np.array(cfg['y'], dtype=float)
        c     = cfg['alpha'] * cfg['r']
        n     = cfg['n']
        y_str = str(cfg['y'])

        print(f'\n  Config n={n}, y={y_str}, c={c:.3f}')

        # ── Feasibility guard ─────────────────────────────────────────────────
        if not is_feasible(y, c):
            print(f'    → INFEASIBLE (skipping)')
            results.append({
                'n': n, 'y': y_str, 'c': c, 'feasible': False,
                'payoff': '-', 'dist_cvxpy': np.nan,
                'close_match': np.nan, 'kappa': np.nan,
            })
            continue

        # ── Water-filling solution ────────────────────────────────────────────
        z_cap, kappa = water_filling_cap(y, c)
        print(f'    z_cap = {np.round(z_cap, 4)},  κ = {kappa:.4f}')

        # ── CVXPY comparison for each payoff ──────────────────────────────────
        for payoff in PAYOFFS:
            z_cvx = solve_cvxpy(y, c, payoff_name=payoff)

            if z_cvx is None:
                dist        = np.nan
                close_match = np.nan
                print(f'    [{payoff}] CVXPY returned None')
            elif payoff == 'linear':
                # Objective is constant on the feasible set; any z is optimal.
                dist        = float(np.linalg.norm(z_cap - z_cvx))
                close_match = True   # water-filling is trivially optimal
                print(f'    [{payoff}] linear (const obj), dist={dist:.4f}  → match=True')
            else:
                dist        = float(np.linalg.norm(z_cap - z_cvx))
                close_match = dist < 0.05
                status      = 'OK' if close_match else 'WARNING'
                print(f'    [{payoff}] dist={dist:.4f}  → {status}')
                if not close_match:
                    print(f'      z_cap={np.round(z_cap, 4)},  z_cvx={np.round(z_cvx, 4)}')

            results.append({
                'n': n, 'y': y_str, 'c': c, 'feasible': True,
                'payoff': payoff, 'dist_cvxpy': dist,
                'close_match': close_match, 'kappa': kappa,
            })

        # ── Brute-force verification for m=2 ─────────────────────────────────
        if n == 2:
            z_bf = brute_force_2d(y, c, np.sqrt, n_grid=400)
            if z_bf is not None:
                dist_bf    = float(np.linalg.norm(z_cap - z_bf))
                bf_match   = dist_bf < 0.05
                status     = 'OK' if bf_match else 'WARNING'
                print(f'    [brute_force_sqrt] dist={dist_bf:.4f}  → {status}')
                print(f'      z_bf={np.round(z_bf, 4)}')
                results.append({
                    'n': n, 'y': y_str, 'c': c, 'feasible': True,
                    'payoff': 'brute_force_sqrt', 'dist_cvxpy': dist_bf,
                    'close_match': bf_match, 'kappa': kappa,
                })

        # ── Permutation invariance ────────────────────────────────────────────
        y_perm = y.copy()
        np.random.default_rng(0).shuffle(y_perm)
        z_perm, kappa_perm = water_filling_cap(y_perm, c)
        obj_orig = float(np.sum(np.sqrt(z_cap)))
        obj_perm = float(np.sum(np.sqrt(z_perm)))
        perm_ok  = abs(obj_orig - obj_perm) < 1e-6
        perm_str = 'PASS' if perm_ok else 'FAIL'
        print(f'    [permutation] obj_orig={obj_orig:.6f}, obj_perm={obj_perm:.6f}  → {perm_str}')
        assert perm_ok, (
            f'Permutation invariance FAILED for y={y_str}: '
            f'obj_orig={obj_orig:.8f}, obj_perm={obj_perm:.8f}'
        )

    # ── Infeasibility detection ───────────────────────────────────────────────
    y_inf = np.array([1.0, 1.0])
    c_inf = float(np.sum(y_inf)) + 1.0
    assert not is_feasible(y_inf, c_inf), (
        f'Expected is_feasible to return False for c={c_inf} > sum(y)={np.sum(y_inf)}'
    )
    print('\n  Infeasibility check: PASSED  (c > sum(y) correctly detected as infeasible)')

    return results


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_feasible_region(out_path):
    """
    2-D contour plot in (z1, z2) space for y=[1.5, 0.5], c=0.3.
    Shows objective iso-curves, the feasible segment, and the water-filling
    optimal point marked with a star.
    """
    y  = np.array([1.5, 0.5])
    c  = 0.3
    y1, y2 = y[0], y[1]
    target = y1 + y2 - c   # = 1.7

    z_cap, kappa = water_filling_cap(y, c)

    # Grid for contour
    z1g = np.linspace(0.02, y1 + 0.12, 350)
    z2g = np.linspace(0.02, y2 + 0.12, 350)
    Z1, Z2 = np.meshgrid(z1g, z2g)
    OBJ = np.sqrt(Z1) + np.sqrt(Z2)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Objective contour fill
    cf = ax.contourf(Z1, Z2, OBJ, levels=25, cmap='viridis', alpha=0.65)
    fig.colorbar(cf, ax=ax, label='Objective  √z₁ + √z₂')
    ax.contour(Z1, Z2, OBJ, levels=15, colors='white', linewidths=0.4, alpha=0.5)

    # Feasible segment:  z1 + z2 = target,  0 < z1 <= y1,  0 < z2 <= y2
    eps     = 1e-4
    z1_lo   = max(target - y2 + eps, eps)   # ≈ 1.2
    z1_hi   = min(y1 - eps, target - eps)   # ≈ 1.5
    z1_seg  = np.linspace(z1_lo, z1_hi, 400)
    z2_seg  = target - z1_seg
    ax.plot(z1_seg, z2_seg, 'w-', linewidth=2.8, label='Feasible segment', zorder=3)

    # Box constraint lines
    ax.axvline(y1, color='silver', linestyle='--', linewidth=1.0, alpha=0.8, label=f'z₁ ≤ y₁={y1}')
    ax.axhline(y2, color='silver', linestyle=':',  linewidth=1.0, alpha=0.8, label=f'z₂ ≤ y₂={y2}')

    # Water-filling optimum
    ax.scatter(
        [z_cap[0]], [z_cap[1]],
        marker='*', s=450, color='red', zorder=5,
        label=f'z_cap = ({z_cap[0]:.3f}, {z_cap[1]:.3f}),  κ = {kappa:.3f}',
    )

    ax.set_xlabel('z₁', fontsize=11)
    ax.set_ylabel('z₂', fontsize=11)
    ax.set_title(
        f'Feasible region  —  y = [1.5, 0.5],  c = {c}\n'
        'Objective contours √z₁ + √z₂;  water-filling optimum ★',
        fontsize=11,
    )
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(0.0, y1 + 0.18)
    ax.set_ylim(0.0, y2 + 0.18)
    ax.grid(alpha=0.18, linestyle='--')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'  Saved: {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs('/workspace/project/results', exist_ok=True)

    print('=' * 62)
    print('Experiment 5: Finite-Player Boundary Optimisation')
    print('=' * 62)

    results = run_tests()

    # ── Summary ───────────────────────────────────────────────────────────────
    # Non-trivial payoffs: exclude linear (vacuously true) and infeasible rows
    concave_rows = [
        r for r in results
        if r['feasible'] and r['payoff'] in ('sqrt', 'log1p', 'brute_force_sqrt')
    ]
    n_total = len(concave_rows)
    n_match = sum(1 for r in concave_rows if r['close_match'] is True)

    all_checked = [
        r for r in results
        if r['feasible'] and r['payoff'] not in ('-',)
        and r['close_match'] is not np.nan
    ]
    n_all = len(all_checked)
    n_all_ok = sum(1 for r in all_checked if r['close_match'] is True)

    print(f'\nCVXPY / brute-force match (sqrt + log1p): {n_match}/{n_total} cases with dist < 0.05')
    print(f'All payoffs (incl. linear):               {n_all_ok}/{n_all} cases optimal / trivially optimal')

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    csv_path = '/workspace/project/results/exp5_results.csv'
    df.to_csv(csv_path, index=False)
    print(f'\n  Saved: {csv_path}  ({len(df)} rows × {len(df.columns)} cols)')

    # ── Plot ──────────────────────────────────────────────────────────────────
    print('\nGenerating plot …')
    plot_feasible_region('/workspace/project/results/exp5_feasible_region.png')

    print('\nAll assertions passed. Experiment 5 complete.')


if __name__ == '__main__':
    main()
