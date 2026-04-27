"""
generate_plots.py

WP2 — Generates paper-ready plots from the batch experiment CSV data.

Plots produced:
    1. Runtime versus number of vertices (Baseline, Raw Montanaro, Enhanced Montanaro)
    2. Number of explored tree nodes versus number of vertices
    3. Pruning effect plot (nodes before/after rules)
    4. Ablation study (incremental effect of rules)
    5. Runtime/node count vs graph density
    6. Threshold-search behavior (C vs tree size)
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import ast

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
    df = df[df['finished'] == True].copy()
    
    # Pre-parse threshold trajectory if present
    if 'threshold_trajectory' in df.columns:
        df['threshold_trajectory'] = df['threshold_trajectory'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
        )
    return df


# ──────────────────────────────────────────────────────────────────
# Plot 1 & 2: Runtime and Nodes vs Number of Vertices
# ──────────────────────────────────────────────────────────────────

def plot_vs_vertices(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    solvers_to_plot = ['baseline', 'raw_emulation', 'enhanced_emulation']
    labels = {
        'baseline': 'Baseline B&B',
        'raw_emulation': 'Raw Montanaro-inspired',
        'enhanced_emulation': 'Enhanced Montanaro-inspired'
    }
    colors = {'baseline': '#1f77b4', 'raw_emulation': '#ff7f0e', 'enhanced_emulation': '#2ca02c'}
    markers = {'baseline': 'o', 'raw_emulation': 's', 'enhanced_emulation': '^'}
    
    # Group by n and solver type
    # We use median to be robust to outliers from hard instances
    grouped = df.groupby(['n', 'solver_type'])[['runtime_seconds', 'nodes_explored']].median().reset_index()

    # Runtime vs Vertices
    ax = axes[0]
    for solver in solvers_to_plot:
        sub = grouped[grouped['solver_type'] == solver]
        if not sub.empty:
            ax.plot(sub['n'], sub['runtime_seconds'], marker=markers[solver], 
                    color=colors[solver], label=labels[solver], linewidth=2, markersize=8)
    
    ax.set_yscale('log')
    ax.set_xlabel('Number of vertices (n)')
    ax.set_ylabel('Median Runtime (seconds)')
    ax.set_title('Runtime vs Number of Vertices')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Nodes vs Vertices
    ax = axes[1]
    for solver in solvers_to_plot:
        sub = grouped[grouped['solver_type'] == solver]
        if not sub.empty:
            ax.plot(sub['n'], sub['nodes_explored'], marker=markers[solver], 
                    color=colors[solver], label=labels[solver], linewidth=2, markersize=8)
    
    ax.set_yscale('log')
    ax.set_xlabel('Number of vertices (n)')
    ax.set_ylabel('Median Explored Nodes')
    ax.set_title('Explored Nodes vs Number of Vertices')
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, 'plot1_2_vs_vertices.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────
# Plot 3: Pruning Effect
# ──────────────────────────────────────────────────────────────────

def plot_pruning_effect(df, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Compare raw_emulation vs enhanced_emulation
    raw = df[df['solver_type'] == 'raw_emulation'].set_index(['family', 'params', 'seed'])
    enh = df[df['solver_type'] == 'enhanced_emulation'].set_index(['family', 'params', 'seed'])
    
    common = raw.index.intersection(enh.index)
    raw = raw.loc[common].reset_index()
    enh = enh.loc[common].reset_index()
    
    families = raw['family'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(families)))
    family_colors = dict(zip(families, colors))
    
    for fam in families:
        mask = raw['family'] == fam
        ax.scatter(raw.loc[mask, 'nodes_explored'], enh.loc[mask, 'nodes_explored'],
                   label=fam.replace('_', ' ').title(), color=family_colors[fam],
                   alpha=0.7, edgecolors='k')
                   
    # Identity line
    max_val = max(raw['nodes_explored'].max(), enh['nodes_explored'].max()) * 1.5
    min_val = min(raw['nodes_explored'].min(), enh['nodes_explored'].min()) * 0.5
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='No Improvement')
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Explored Nodes (Raw Montanaro-inspired)')
    ax.set_ylabel('Explored Nodes (Enhanced Montanaro-inspired)')
    ax.set_title('Pruning Effect: Search Space Reduction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    path = os.path.join(output_dir, 'plot3_pruning_effect.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────
# Plot 4: Ablation Study
# ──────────────────────────────────────────────────────────────────

def plot_ablation_study(df, output_dir):
    solvers = ['raw_emulation', 'ablation_prep', 'ablation_prep_lb', 'enhanced_emulation']
    labels = ['Raw', '+ Prep', '+ LB', '+ Branching']
    
    # Take median nodes across all instances for each solver
    medians = []
    for s in solvers:
        sub = df[df['solver_type'] == s]
        if not sub.empty:
            medians.append(sub['nodes_explored'].median())
        else:
            medians.append(0)
            
    if sum(medians) == 0:
        print("No ablation data found.")
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    x = np.arange(len(labels))
    ax.bar(x, medians, color='skyblue', edgecolor='black')
    
    # Add text labels on bars
    for i, v in enumerate(medians):
        ax.text(i, v * 1.05, f"{int(v)}", ha='center', fontweight='bold')
        
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Median Explored Nodes (Log Scale)')
    ax.set_yscale('log')
    ax.set_title('Ablation Study: Impact of Classical Rules')
    ax.grid(axis='y', alpha=0.3)
    
    path = os.path.join(output_dir, 'plot4_ablation_study.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────
# Plot 5: Density Study
# ──────────────────────────────────────────────────────────────────

def plot_density_study(df, output_dir):
    density_df = df[df['family'] == 'erdos_renyi_density'].copy()
    if density_df.empty:
        print("No density data found.")
        return
        
    # Extract p from params
    density_df['p'] = density_df['params'].str.extract(r'p=([0-9.]+)').astype(float)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    solvers = ['baseline', 'raw_emulation', 'enhanced_emulation']
    labels = {
        'baseline': 'Baseline B&B',
        'raw_emulation': 'Raw Montanaro',
        'enhanced_emulation': 'Enhanced Montanaro'
    }
    
    grouped = density_df.groupby(['p', 'solver_type'])[['nodes_explored', 'runtime_seconds']].median().reset_index()
    
    ax = axes[0]
    for s in solvers:
        sub = grouped[grouped['solver_type'] == s]
        if not sub.empty:
            ax.plot(sub['p'], sub['nodes_explored'], marker='o', label=labels[s])
    ax.set_yscale('log')
    ax.set_xlabel('Edge Probability (p)')
    ax.set_ylabel('Median Explored Nodes')
    ax.set_title('Difficulty by Density (n=20)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1]
    for s in solvers:
        sub = grouped[grouped['solver_type'] == s]
        if not sub.empty:
            ax.plot(sub['p'], sub['runtime_seconds'], marker='o', label=labels[s])
    ax.set_yscale('log')
    ax.set_xlabel('Edge Probability (p)')
    ax.set_ylabel('Median Runtime (s)')
    ax.set_title('Runtime by Density (n=20)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    path = os.path.join(output_dir, 'plot5_density_study.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────
# Plot 6: Threshold-Search Behavior
# ──────────────────────────────────────────────────────────────────

def plot_threshold_behavior(df, output_dir):
    # Find a representative hard instance for raw_emulation
    raw_runs = df[df['solver_type'] == 'raw_emulation']
    if raw_runs.empty or 'threshold_trajectory' not in raw_runs.columns:
        print("No threshold trajectory data found.")
        return
        
    # Pick a run that had many explored nodes to show a nice trajectory
    raw_runs = raw_runs.sort_values('nodes_explored', ascending=False)
    rep_run = raw_runs.iloc[0]
    
    traj = rep_run['threshold_trajectory']
    if not traj:
        return
        
    # traj is a list of (T, C, tree_size)
    c_values = [x[1] for x in traj]
    tree_sizes = [x[2] for x in traj]
    iterations = list(range(len(traj)))
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    color = 'tab:red'
    ax1.set_xlabel('Algorithm Iteration')
    ax1.set_ylabel('Cost Threshold $C$', color=color)
    ax1.plot(iterations, c_values, marker='o', color=color, linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel(r'Built Tree Size $|\mathcal{T}_C|$', color=color)  
    ax2.plot(iterations, tree_sizes, marker='s', color=color, linestyle='--', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log')
    
    fig.tight_layout()
    plt.title(f"Threshold Behavior (Instance: {rep_run['family']} {rep_run['params']})")
    
    path = os.path.join(output_dir, 'plot6_threshold_behavior.png')
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

    plot_vs_vertices(df, plots_dir)
    plot_pruning_effect(df, plots_dir)
    plot_ablation_study(df, plots_dir)
    plot_density_study(df, plots_dir)
    plot_threshold_behavior(df, plots_dir)

    print(f"\nAll plots saved to: {plots_dir}/")


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    plots_dir = os.path.join(script_dir, 'plots')
    generate_all_plots(results_dir, plots_dir)
