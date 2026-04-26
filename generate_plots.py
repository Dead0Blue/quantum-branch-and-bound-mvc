"""
generate_plots.py

WP2 — Generates paper-ready plots from the batch experiment CSV data.

Plots produced:
    1. Runtime: baseline vs improved solver
    2. Explored nodes: baseline vs improved
    3. Pruning ratio vs graph family / density
    4. Effect of preprocessing/propagation on runtime and nodes
    5. (Bonus) Runtime vs optimal cover size
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})


def load_data(results_dir='results'):
    """Load the merged CSV."""
    path = os.path.join(results_dir, 'results_all.csv')
    df = pd.read_csv(path)
    # Only finished runs
    df = df[df['finished'] == True].copy()
    return df


def prepare_comparison(df):
    """
    Pivot so that each instance has both baseline and improved results
    side by side.
    """
    # Create a unique instance key
    df['instance_key'] = df['family'] + '|' + df['params'] + '|' + df['seed'].astype(str)

    baseline = df[df['solver_type'] == 'baseline'].set_index('instance_key')
    improved = df[df['solver_type'] == 'improved'].set_index('instance_key')

    # Only keep instances where both solvers finished
    common = baseline.index.intersection(improved.index)
    baseline = baseline.loc[common]
    improved = improved.loc[common]

    return baseline, improved


# ──────────────────────────────────────────────────────────────────
# Plot 1: Runtime — baseline vs improved
# ──────────────────────────────────────────────────────────────────

def plot_runtime_comparison(df, output_dir):
    baseline, improved = prepare_comparison(df)

    fig, ax = plt.subplots(figsize=(8, 6))

    families = baseline['family'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(families)))
    family_colors = dict(zip(families, colors))

    for fam in families:
        mask = baseline['family'] == fam
        ax.scatter(
            baseline.loc[mask, 'runtime_seconds'],
            improved.loc[mask, 'runtime_seconds'],
            label=fam.replace('_', ' ').title(),
            color=family_colors[fam],
            alpha=0.7, edgecolors='k', linewidth=0.3, s=40
        )

    # Identity line
    max_val = max(baseline['runtime_seconds'].max(), improved['runtime_seconds'].max()) * 1.1
    min_val = min(baseline['runtime_seconds'].min(), improved['runtime_seconds'].min())
    min_val = max(min_val, 1e-6)  # avoid log(0)
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.4, label='Equal')

    ax.set_xlabel('Baseline runtime (s)')
    ax.set_ylabel('Improved runtime (s)')
    ax.set_title('Plot 1: Runtime — Baseline vs Improved Solver')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, 'plot1_runtime_comparison.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────
# Plot 2: Explored nodes — baseline vs improved
# ──────────────────────────────────────────────────────────────────

def plot_nodes_comparison(df, output_dir):
    baseline, improved = prepare_comparison(df)

    fig, ax = plt.subplots(figsize=(8, 6))

    families = baseline['family'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(families)))
    family_colors = dict(zip(families, colors))

    for fam in families:
        mask = baseline['family'] == fam
        ax.scatter(
            baseline.loc[mask, 'nodes_explored'],
            improved.loc[mask, 'nodes_explored'],
            label=fam.replace('_', ' ').title(),
            color=family_colors[fam],
            alpha=0.7, edgecolors='k', linewidth=0.3, s=40
        )

    max_val = max(baseline['nodes_explored'].max(), improved['nodes_explored'].max()) * 1.1
    min_val = max(1, min(baseline['nodes_explored'].min(), improved['nodes_explored'].min()))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.4, label='Equal')

    ax.set_xlabel('Baseline nodes explored')
    ax.set_ylabel('Improved nodes explored')
    ax.set_title('Plot 2: Explored Nodes — Baseline vs Improved Solver')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, 'plot2_nodes_comparison.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────
# Plot 3: Pruning ratio vs graph family or density
# ──────────────────────────────────────────────────────────────────

def plot_pruning_ratio(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── 3a: Pruning ratio by family ──
    ax = axes[0]
    for solver_type, marker, offset in [('baseline', 's', -0.15), ('improved', 'o', 0.15)]:
        sub = df[df['solver_type'] == solver_type].copy()
        sub['pruning_ratio'] = sub['nodes_pruned'] / sub['nodes_explored'].clip(lower=1)

        families = sorted(sub['family'].unique())
        means = [sub[sub['family'] == f]['pruning_ratio'].mean() for f in families]
        stds = [sub[sub['family'] == f]['pruning_ratio'].std() for f in families]

        x = np.arange(len(families))
        ax.bar(x + offset, means, width=0.3, label=solver_type.title(),
               alpha=0.8, yerr=stds, capsize=3)

    ax.set_xticks(np.arange(len(families)))
    ax.set_xticklabels([f.replace('_', '\n') for f in families], fontsize=9)
    ax.set_ylabel('Pruning ratio (pruned / explored)')
    ax.set_title('3a: Pruning Ratio by Graph Family')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # ── 3b: Pruning ratio vs density (Erdos-Renyi only) ──
    ax = axes[1]
    er = df[df['family'] == 'erdos_renyi'].copy()
    if not er.empty:
        # Extract p from params string
        er['p'] = er['params'].str.extract(r'p=([0-9.]+)').astype(float)
        er['pruning_ratio'] = er['nodes_pruned'] / er['nodes_explored'].clip(lower=1)

        for solver_type in ['baseline', 'improved']:
            sub = er[er['solver_type'] == solver_type]
            grouped = sub.groupby('p')['pruning_ratio'].agg(['mean', 'std'])
            ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                       marker='o', label=solver_type.title(), capsize=3)

    ax.set_xlabel('Edge probability p')
    ax.set_ylabel('Pruning ratio')
    ax.set_title('3b: Pruning Ratio vs Density (Erdos-Renyi)')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, 'plot3_pruning_ratio.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────
# Plot 4: Effect of preprocessing/propagation
# ──────────────────────────────────────────────────────────────────

def plot_preprocessing_effect(df, output_dir):
    baseline, improved = prepare_comparison(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── 4a: Speedup ratio by family ──
    ax = axes[0]
    speedup = baseline['runtime_seconds'] / improved['runtime_seconds'].clip(lower=1e-9)
    combined = pd.DataFrame({
        'family': baseline['family'].values,
        'n': baseline['n'].values,
        'speedup': speedup.values,
    })

    families = sorted(combined['family'].unique())
    x = np.arange(len(families))
    means = [combined[combined['family'] == f]['speedup'].mean() for f in families]
    stds = [combined[combined['family'] == f]['speedup'].std() for f in families]

    bars = ax.bar(x, means, yerr=stds, capsize=4, color=plt.cm.Paired(np.linspace(0.1, 0.9, len(families))),
                  edgecolor='k', linewidth=0.5, alpha=0.85)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No improvement')
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace('_', '\n') for f in families], fontsize=9)
    ax.set_ylabel('Speedup (baseline / improved runtime)')
    ax.set_title('4a: Runtime Speedup by Graph Family')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # ── 4b: Node reduction ratio by n ──
    ax = axes[1]
    node_reduction = baseline['nodes_explored'] / improved['nodes_explored'].clip(lower=1)
    combined2 = pd.DataFrame({
        'family': baseline['family'].values,
        'n': baseline['n'].values,
        'node_reduction': node_reduction.values,
    })

    # Group by n
    grouped = combined2.groupby('n')['node_reduction'].agg(['mean', 'std'])
    ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
               marker='s', color='darkblue', capsize=4, linewidth=1.5)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No improvement')
    ax.set_xlabel('Number of vertices (n)')
    ax.set_ylabel('Node reduction ratio (baseline / improved)')
    ax.set_title('4b: Explored Nodes Reduction vs Graph Size')
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, 'plot4_preprocessing_effect.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────
# Plot 5 (bonus): Runtime vs optimal cover size
# ──────────────────────────────────────────────────────────────────

def plot_runtime_vs_cover(df, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))

    for solver_type, marker in [('baseline', 's'), ('improved', 'o')]:
        sub = df[df['solver_type'] == solver_type]
        ax.scatter(
            sub['optimal_cover_size'], sub['runtime_seconds'],
            marker=marker, alpha=0.5, label=solver_type.title(),
            edgecolors='k', linewidth=0.3, s=35
        )

    ax.set_xlabel('Optimal cover size')
    ax.set_ylabel('Runtime (s)')
    ax.set_title('Plot 5: Runtime vs Optimal Cover Size')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)

    path = os.path.join(output_dir, 'plot5_runtime_vs_cover.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def generate_all_plots(results_dir='results', plots_dir='plots'):
    os.makedirs(plots_dir, exist_ok=True)
    df = load_data(results_dir)
    print(f"Loaded {len(df)} rows from results_all.csv")

    plot_runtime_comparison(df, plots_dir)
    plot_nodes_comparison(df, plots_dir)
    plot_pruning_ratio(df, plots_dir)
    plot_preprocessing_effect(df, plots_dir)
    plot_runtime_vs_cover(df, plots_dir)

    print(f"\nAll plots saved to: {plots_dir}/")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    plots_dir = os.path.join(script_dir, 'plots')
    generate_all_plots(results_dir, plots_dir)
