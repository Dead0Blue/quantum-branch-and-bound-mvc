# Experiment README — MVC Branch-and-Bound Comparison

## Overview

This experiment compares a **baseline** and an **improved** exact branch-and-bound solver for the Minimum Vertex Cover (MVC) problem across multiple graph families.

## Solver Modes

| Mode       | Propagation | Preprocessing | Lower Bound        |
|------------|-------------|---------------|---------------------|
| Baseline   | ✗           | ✗             | Trivial (selected count) |
| Improved   | ✓ Forced-neighbor | ✓ Isolated vertex + degree-1 | Maximal matching |

### Improvements in the "Improved" Solver

1. **Forced-neighbor propagation**: When a vertex is excluded (set to 0), all its neighbors are immediately included (set to 1), since those edges must be covered. Applied recursively.

2. **Isolated vertex deletion**: Vertices with no uncovered edges are set to 0 (irrelevant to the cover).

3. **Degree-1 rule**: If a vertex has exactly one unassigned neighbor, that neighbor is included in the cover.

4. **Maximal matching lower bound**: A greedy maximal matching on the residual graph gives a tighter lower bound than just counting selected vertices.

5. **High-degree branching heuristic**: The improved solver branches on the unassigned vertex with the highest residual degree.

## Graph Families

| Family          | Parameters                             | Sizes         |
|-----------------|----------------------------------------|---------------|
| Erdos-Renyi     | p ∈ {0.2, 0.3, 0.5}                  | n ∈ {10, 15, 20, 25} |
| Barabasi-Albert | m ∈ {1, 2, 3}                         | n ∈ {10, 15, 20, 25} |
| Watts-Strogatz  | k=4, p ∈ {0.1, 0.3, 0.5}             | n ∈ {10, 15, 20, 25} |
| Random Regular  | d ∈ {3, 4}                            | n ∈ {10, 14, 20, 24} |
| Toy graphs      | toy_5, toy_8, toy_9, toy_11, toy_15   | Fixed |

## Seeds

5 seeds per parameter combination: {42, 123, 256, 789, 1024}.

## Instance Count

| Family          | Configs (sizes × params) | Seeds | Total instances |
|-----------------|--------------------------|-------|-----------------|
| Erdos-Renyi     | 4 × 3 = 12              | 5     | 60              |
| Barabasi-Albert | 4 × 3 = 12              | 5     | 60              |
| Watts-Strogatz  | 4 × 3 = 12              | 5     | 60              |
| Random Regular  | 4 × 2 = 8               | 5     | 40              |
| Toy             | 5                        | 1     | 5               |
| **Total**       |                          |       | **225 instances × 2 solvers = 450 runs** |

## Metrics Logged (per run)

- Graph family, parameters, seed, n, m
- Solver type, preprocessing/propagation flags, lower bound type
- Optimal cover size
- Runtime (seconds)
- Nodes explored and pruned
- Maximum search depth
- Forced assignments count
- Preprocessing reductions count
- Root lower bound
- Completion status

## Output Files

- `results/results_erdos_renyi.csv` — per-family CSV
- `results/results_barabasi_albert.csv`
- `results/results_watts_strogatz.csv`
- `results/results_random_regular.csv`
- `results/results_toy.csv`
- `results/results_all.csv` — merged CSV (all families)

## Plots

- `plots/plot1_runtime_comparison.png` — Runtime: baseline vs improved
- `plots/plot2_nodes_comparison.png` — Explored nodes: baseline vs improved
- `plots/plot3_pruning_ratio.png` — Pruning ratio by family and density
- `plots/plot4_preprocessing_effect.png` — Speedup and node reduction
- `plots/plot5_runtime_vs_cover.png` — Runtime vs optimal cover size

## How to Reproduce

```bash
# Step 1: Run batch experiments (generates CSVs)
python batch_experiments.py

# Step 2: Generate plots from CSV data
python generate_plots.py
```

All random seeds are fixed for full reproducibility.

## Timeout

Each individual solver run has a 120-second timeout. Runs that exceed this are flagged as `finished=False` in the CSV.
