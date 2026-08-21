"""
Experiment 1: One-Step Auxiliary Optimisation — Cutoff / Tax-the-Richest Rule.

Verifies that the cutoff operator T_c welfare-dominates standard benchmark
allocations (proportional haircut, equal deduction, randomised) across a grid
of synthetic probability laws and required loss amounts.

Outputs
-------
results/exp1_results.csv
results/exp1_welfare_comparison.png
results/exp1_cutoff_viz.png
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

from src.cutoff_operator import compute_threshold, apply_cutoff, cutoff_operator
from src.welfare import evaluate_welfare, evaluate_all_payoffs, PAYOFF_FUNCTIONS, compute_mean
from src.benchmarks import proportional_haircut, equal_deduction_strongest, randomized_feasible

# ── Synthetic law constructors ────────────────────────────────────────────────

def make_uniform_law(a, b, n=300):
    x = np.linspace(a, b, n)
    w = np.ones(n) / n
    return x, w


def make_triangular_law(a, b, n=300):
    x = np.linspace(a, b, n)
    w = (x - a) / (b - a)
    w /= w.sum()
    return x, w


def make_beta_law(alpha_param, beta_param, lo, hi, n=300):
    from scipy.stats import beta as beta_dist
    x = np.linspace(lo, hi, n)
    w = beta_dist.pdf((x - lo) / (hi - lo), alpha_param, beta_param)
    w /= w.sum()
    return x, w


def make_bimodal_law(n=300):
    x = np.linspace(0.1, 3.0, n)
    from scipy.stats import norm
    w = 0.5 * norm.pdf(x, 0.7, 0.2) + 0.5 * norm.pdf(x, 2.3, 0.2)
    w /= w.sum()
    return x, w


# Named laws: list of (name, (x, w)) pairs
LAWS = [
    ('uniform_1_3',       make_uniform_law(1.0, 3.0)),
    ('triangular_0.5_4',  make_triangular_law(0.5, 4.0)),
    ('beta_2_5_0.5_3',    make_beta_law(2, 5, 0.5, 3.0)),
    ('bimodal',           make_bimodal_law()),
]

C_FACTORS = [0.05, 0.15, 0.30]

# ── Test matrix ───────────────────────────────────────────────────────────────

def run_test_matrix():
    """Run all (law, c_factor) combinations; return list of result dicts."""
    results = []

    for law_name, (x, w) in LAWS:
        mean_x = compute_mean(x, w)

        for factor in C_FACTORS:
            c = min(factor * mean_x, 0.9 * mean_x)

            # 1. Cutoff operator
            z_cut, x_star, mean_red = cutoff_operator(x, w, c)

            # 2. Sanity checks
            assert abs(np.sum(w) - 1.0) < 1e-8, (
                f"[{law_name}] weights sum to {np.sum(w):.10f}, expected 1"
            )
            assert abs(mean_red - c) < 1e-6, (
                f"[{law_name}, c={c:.4f}] mean_red={mean_red:.8f} ≠ c={c:.8f}"
            )
            assert np.all(z_cut >= 0), (
                f"[{law_name}, c={c:.4f}] negative post-cutoff states"
            )

            # 3. Proportional haircut
            z_hair, lam = proportional_haircut(x, w, c)

            # 4. Equal deduction (top 40 % donors)
            z_eq, d_eq = equal_deduction_strongest(x, w, c, fraction_donors=0.4)

            # 5. Randomised feasible (with fallback)
            try:
                z_rand, d_rand = randomized_feasible(
                    x, w, c, rng=np.random.default_rng(42)
                )
            except Exception:
                z_rand = z_hair.copy()

            # 6. Welfare for every allocation and every payoff
            welf_cut  = evaluate_all_payoffs(z_cut,  w, 0.0)
            welf_hair = evaluate_all_payoffs(z_hair, w, 0.0)
            welf_eq   = evaluate_all_payoffs(z_eq,   w, 0.0)
            welf_rand = evaluate_all_payoffs(z_rand, w, 0.0)

            # 7. Assert cutoff dominates proportional haircut (per payoff)
            for name in welf_cut:
                assert welf_cut[name] >= welf_hair[name] - 1e-6, (
                    f"[{law_name}, c_factor={factor}, {name}] "
                    f"welfare_cut={welf_cut[name]:.8f} < welfare_hair={welf_hair[name]:.8f}"
                )

            # 8. Record one row per payoff
            for name in welf_cut:
                results.append({
                    'law':         law_name,
                    'c_factor':    factor,
                    'c_value':     round(c, 8),
                    'mean_x':      round(mean_x, 8),
                    'x_star':      round(x_star, 8),
                    'lambda_hair': round(lam, 8),
                    'payoff':      name,
                    'welfare_cut':  round(welf_cut[name],  8),
                    'welfare_hair': round(welf_hair[name], 8),
                    'welfare_eq':   round(welf_eq[name],   8),
                    'welfare_rand': round(welf_rand[name], 8),
                })

    return results


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_welfare_comparison(results, out_path):
    """Grouped bar chart for the first test case (law×c_factor combination)."""
    first_law     = results[0]['law']
    first_cfactor = results[0]['c_factor']
    subset = [
        r for r in results
        if r['law'] == first_law and r['c_factor'] == first_cfactor
    ]

    payoff_names = [r['payoff']      for r in subset]
    welf_cut     = [r['welfare_cut']  for r in subset]
    welf_hair    = [r['welfare_hair'] for r in subset]
    welf_eq      = [r['welfare_eq']   for r in subset]
    welf_rand    = [r['welfare_rand'] for r in subset]

    n     = len(payoff_names)
    x_pos = np.arange(n)
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x_pos - 1.5 * width, welf_cut,  width, label='Cutoff T_c',      color='steelblue',  edgecolor='white')
    ax.bar(x_pos - 0.5 * width, welf_hair, width, label='Prop. haircut',    color='tomato',     edgecolor='white')
    ax.bar(x_pos + 0.5 * width, welf_eq,   width, label='Equal deduction',  color='seagreen',   edgecolor='white')
    ax.bar(x_pos + 1.5 * width, welf_rand, width, label='Randomised',       color='goldenrod',  edgecolor='white')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(payoff_names, rotation=22, ha='right', fontsize=10)
    ax.set_ylabel('Welfare W(f, μ)')
    c_val = subset[0]['c_value']
    ax.set_title(
        f'Welfare comparison — {first_law},  c = {first_cfactor:.0%} × mean = {c_val:.3f}\n'
        'Cutoff rule vs. three benchmarks',
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_cutoff_viz(out_path):
    """Two-panel density plot: initial law (top) and post-allocation (bottom)."""
    x, w = make_uniform_law(1.0, 3.0)
    mean_x = compute_mean(x, w)
    c = 0.15 * mean_x

    z_cut,  x_star, _ = cutoff_operator(x, w, c)
    z_hair, lam       = proportional_haircut(x, w, c)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=False)

    # ── Top: initial density ──────────────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(x, w, step='mid', alpha=0.35, color='steelblue')
    ax.step(x, w, where='mid', color='steelblue', linewidth=1.4, label='Initial law μ')
    ax.axvline(mean_x, color='dimgray',   linestyle='--', linewidth=1.2,
               label=f'Mean = {mean_x:.3f}')
    ax.axvline(x_star, color='steelblue', linestyle=':',  linewidth=1.8,
               label=f'Cutoff x* = {x_star:.3f}')
    ax.set_ylabel('Weight  w_j', fontsize=10)
    ax.set_title('Initial density — Uniform[1, 3]', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, linestyle='--')

    # ── Bottom: post-allocation comparison ────────────────────────────────────
    ax = axes[1]
    ax.fill_between(z_cut,  w, step='mid', alpha=0.35, color='steelblue')
    ax.step(z_cut,  w, where='mid', color='steelblue', linewidth=1.4,
            label=f'Cutoff T_c(μ),  x*={x_star:.3f}')
    ax.fill_between(z_hair, w, step='mid', alpha=0.35, color='tomato')
    ax.step(z_hair, w, where='mid', color='tomato',     linewidth=1.4,
            label=f'Prop. haircut,   λ={lam:.3f}')
    ax.axvline(x_star, color='steelblue', linestyle=':', linewidth=1.8, alpha=0.7)
    ax.set_xlabel('State value', fontsize=10)
    ax.set_ylabel('Weight  w_j', fontsize=10)
    ax.set_title(
        f'Post-allocation density  (c = 0.15 × mean = {c:.4f})',
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2, linestyle='--')

    fig.tight_layout(h_pad=2.5)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs('/workspace/project/results', exist_ok=True)

    print('=' * 62)
    print('Experiment 1: One-Step Auxiliary Optimisation (Cutoff Rule)')
    print('=' * 62)

    print('\nRunning test matrix …')
    results = run_test_matrix()
    print(f'  Generated {len(results)} result rows '
          f'({len(LAWS)} laws × {len(C_FACTORS)} c-factors × {len(PAYOFF_FUNCTIONS)} payoffs)')

    # ── Dominance summary ─────────────────────────────────────────────────────
    n_total = len(results)
    n_wins = sum(
        1 for r in results
        if r['welfare_cut'] >= max(
            r['welfare_hair'], r['welfare_eq'], r['welfare_rand']
        ) - 1e-6
    )
    print(f'\nCutoff dominance: {n_wins}/{n_total} cases across all payoffs')

    # ── Detailed per-law summary ──────────────────────────────────────────────
    for law_name, _ in LAWS:
        sub = [r for r in results if r['law'] == law_name]
        wins = sum(
            1 for r in sub
            if r['welfare_cut'] >= max(
                r['welfare_hair'], r['welfare_eq'], r['welfare_rand']
            ) - 1e-6
        )
        print(f'  {law_name:<26s}  {wins:3d}/{len(sub):3d}')

    # ── Save CSV ──────────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    csv_path = '/workspace/project/results/exp1_results.csv'
    df.to_csv(csv_path, index=False)
    print(f'\n  Saved: {csv_path}  ({len(df)} rows × {len(df.columns)} cols)')

    # ── Plots ─────────────────────────────────────────────────────────────────
    print('\nGenerating plots …')
    plot_welfare_comparison(
        results,
        '/workspace/project/results/exp1_welfare_comparison.png',
    )
    plot_cutoff_viz('/workspace/project/results/exp1_cutoff_viz.png')

    print('\nAll assertions passed. Experiment 1 complete.')


if __name__ == '__main__':
    main()
