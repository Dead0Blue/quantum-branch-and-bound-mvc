"""
batch_experiments.py

WP2 — Experiment instrumentation and batch runs for the MVC paper.

Generates graph instances from multiple families, runs both baseline
and improved branch-and-bound solvers, and logs all metrics to CSV.
"""

import os
import sys
import time
import csv
import itertools
import networkx as nx
import numpy as np

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from instance_generator import generate_mvc_instance, get_exact_mvc_solution
from classical_solvers import bb_mvc_baseline, bb_mvc_improved


# ======================================================================
# Graph instance generation
# ======================================================================

def generate_experiment_instances():
    """
    Generate all graph instances for the batch experiments.
    
    Returns a list of dicts, each with:
        'graph': nx.Graph
        'family': str
        'params': str  (human-readable parameter summary)
        'n': int
        'm': int
        'seed': int
        'param_dict': dict  (raw parameters)
    """
    instances = []
    seeds = [42, 123, 256, 789, 1024]

    # ── Erdos-Renyi ──────────────────────────────────────────────
    for n in [10, 15, 20, 25]:
        for p in [0.2, 0.3, 0.5]:
            for seed in seeds:
                inst = generate_mvc_instance(
                    model="erdos_renyi", n=n, p=p, seed=seed
                )
                G = inst['graph']
                instances.append({
                    'graph': G,
                    'family': 'erdos_renyi',
                    'params': f'n={n}, p={p}',
                    'n': G.number_of_nodes(),
                    'm': G.number_of_edges(),
                    'seed': seed,
                    'param_dict': {'n': n, 'p': p},
                })

    # ── Barabasi-Albert ──────────────────────────────────────────
    for n in [10, 15, 20, 25]:
        for m in [1, 2, 3]:
            for seed in seeds:
                inst = generate_mvc_instance(
                    model="barabasi_albert", n=n, m=m, seed=seed
                )
                G = inst['graph']
                instances.append({
                    'graph': G,
                    'family': 'barabasi_albert',
                    'params': f'n={n}, m={m}',
                    'n': G.number_of_nodes(),
                    'm': G.number_of_edges(),
                    'seed': seed,
                    'param_dict': {'n': n, 'm': m},
                })

    # ── Watts-Strogatz ───────────────────────────────────────────
    for n in [10, 15, 20, 25]:
        for p in [0.1, 0.3, 0.5]:
            for seed in seeds:
                inst = generate_mvc_instance(
                    model="watts_strogatz", n=n, k=4, p=p, seed=seed
                )
                G = inst['graph']
                instances.append({
                    'graph': G,
                    'family': 'watts_strogatz',
                    'params': f'n={n}, k=4, p={p}',
                    'n': G.number_of_nodes(),
                    'm': G.number_of_edges(),
                    'seed': seed,
                    'param_dict': {'n': n, 'k': 4, 'p': p},
                })

    # ── Random Regular ───────────────────────────────────────────
    for n in [10, 14, 20, 24]:  # n*d must be even
        for d in [3, 4]:
            if (n * d) % 2 != 0:
                continue
            for seed in seeds:
                inst = generate_mvc_instance(
                    model="regular", n=n, d=d, seed=seed
                )
                G = inst['graph']
                instances.append({
                    'graph': G,
                    'family': 'random_regular',
                    'params': f'n={n}, d={d}',
                    'n': G.number_of_nodes(),
                    'm': G.number_of_edges(),
                    'seed': seed,
                    'param_dict': {'n': n, 'd': d},
                })

    # ── Toy graphs ───────────────────────────────────────────────
    for model in ['toy_5', 'toy_8', 'toy_9_mvc_2', 'toy_11_mvc_3', 'toy_15_star']:
        inst = generate_mvc_instance(model=model, seed=0)
        G = inst['graph']
        instances.append({
            'graph': G,
            'family': 'toy',
            'params': model,
            'n': G.number_of_nodes(),
            'm': G.number_of_edges(),
            'seed': 0,
            'param_dict': {'model': model},
        })

    return instances


# ======================================================================
# CSV logging
# ======================================================================

CSV_HEADER = [
    'family', 'params', 'seed', 'n', 'm',
    'solver_type', 'preprocessing_enabled', 'propagation_enabled', 'lower_bound_type',
    'optimal_cover_size', 'runtime_seconds',
    'nodes_explored', 'nodes_pruned', 'max_depth',
    'forced_assignments', 'preprocessing_reductions',
    'vertices_removed_by_preprocessing',
    'root_lower_bound', 'finished',
]


def run_single_experiment(instance, solver_type='baseline', timeout=120):
    """
    Run a single solver on a single instance and return a result dict.
    """
    G = instance['graph']

    if solver_type == 'baseline':
        result = bb_mvc_baseline(G, timeout=timeout)
        config = {
            'solver_type': 'baseline',
            'preprocessing_enabled': False,
            'propagation_enabled': False,
            'lower_bound_type': 'trivial',
        }
    elif solver_type == 'improved':
        result = bb_mvc_improved(G, timeout=timeout)
        config = {
            'solver_type': 'improved',
            'preprocessing_enabled': True,
            'propagation_enabled': True,
            'lower_bound_type': 'maximal_matching',
        }
    else:
        raise ValueError(f"Unknown solver_type: {solver_type}")

    row = {
        'family': instance['family'],
        'params': instance['params'],
        'seed': instance['seed'],
        'n': instance['n'],
        'm': instance['m'],
        **config,
        'optimal_cover_size': result['cover_size'],
        'runtime_seconds': round(result['runtime'], 6),
        'nodes_explored': result['nodes_explored'],
        'nodes_pruned': result['nodes_pruned'],
        'max_depth': result['max_depth'],
        'forced_assignments': result['forced_assignments'],
        'preprocessing_reductions': result['preprocessing_reductions'],
        'vertices_removed_by_preprocessing': result['vertices_removed_by_preprocessing'],
        'root_lower_bound': result['root_lower_bound'],
        'finished': result['finished'],
    }
    return row


def run_batch_experiments(output_dir='results', timeout=120):
    """
    Run all experiments and save to CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    instances = generate_experiment_instances()

    all_rows = []
    family_rows = {}  # family -> list of rows

    total = len(instances) * 2  # baseline + improved for each
    done = 0

    for inst in instances:
        for solver_type in ['baseline', 'improved']:
            done += 1
            family = inst['family']
            n = inst['n']
            params = inst['params']
            seed = inst['seed']

            print(f"[{done}/{total}] {family} | {params} | seed={seed} | {solver_type} | n={n} ...", end=" ", flush=True)

            try:
                row = run_single_experiment(inst, solver_type=solver_type, timeout=timeout)
                print(f"done in {row['runtime_seconds']:.3f}s | nodes={row['nodes_explored']} | cover={row['optimal_cover_size']}")
            except Exception as e:
                print(f"ERROR: {e}")
                # Log a failure row
                row = {col: '' for col in CSV_HEADER}
                row['family'] = inst['family']
                row['params'] = inst['params']
                row['seed'] = inst['seed']
                row['n'] = inst['n']
                row['m'] = inst['m']
                row['solver_type'] = solver_type
                row['finished'] = False

            all_rows.append(row)
            family_rows.setdefault(family, []).append(row)

    # ── Write per-family CSVs ────────────────────────────────────
    for family, rows in family_rows.items():
        filepath = os.path.join(output_dir, f'results_{family}.csv')
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {filepath} ({len(rows)} rows)")

    # ── Write merged CSV ─────────────────────────────────────────
    merged_path = os.path.join(output_dir, 'results_all.csv')
    with open(merged_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Saved: {merged_path} ({len(all_rows)} rows)")

    return all_rows


# ======================================================================
# Main entry point
# ======================================================================

if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    print("=" * 70)
    print("WP2 — Batch MVC Experiments")
    print("=" * 70)
    run_batch_experiments(output_dir=output_dir, timeout=120)
    print("\nAll experiments completed.")
