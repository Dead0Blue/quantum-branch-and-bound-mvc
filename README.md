# Quantum-Inspired Classical Emulation for Minimum Vertex Cover

Implementation of a Montanaro-style quantum branch-and-bound logic applied as a **quantum-inspired classical emulation** for the Minimum Vertex Cover (MVC) problem. This repository includes an exact baseline solver, a raw emulation of Montanaro's branch-and-bound logic, and an enhanced emulation integrating mathematically sound classical heuristic reductions.

## Overview

This project implements and benchmarks:

1. **Classical Exact Solvers** — A baseline brute-force B&B solver.
2. **Raw Montanaro-Inspired Emulation** — An emulation that maps Montanaro's quantum search structure (tree construction, cost thresholding) directly into a classical environment without heuristic rules.
3. **Enhanced Montanaro-Inspired Emulation** — An exact, optimized version combining the Montanaro logic with strict classical pruning (forced-neighbor propagation, degree-based preprocessing, and maximal matching lower bounds).
4. **Benchmarking Pipeline** — Batch experiments across 5 graph families with comprehensive ablation and density studies.

## Project Structure

```
├── classical_solvers.py      # Baseline B&B and branching heuristics
├── montanaro_emulation.py    # Raw and Enhanced Quantum-Inspired Emulators
├── problem_encoding.py       # MVC cost, branching, feasibility
├── instance_generator.py     # Graph generation (ER, BA, WS, regular, toy)
├── visualization.py          # Graph and solution plotting
├── batch_experiments.py      # Batch experiment runner (Ablation, Density)
├── generate_plots.py         # Paper-ready plot generation
├── generate_summary_report.py# Automated PDF Research Summary generator
├── sanity_checks.py          # Exactness verification script
├── results/                  # CSV experiment data
│   ├── results_all.csv           # Merged (1530 runs)
│   ├── results_erdos_renyi.csv
│   ├── results_erdos_renyi_density.csv
│   ├── results_barabasi_albert.csv
│   ├── results_watts_strogatz.csv
│   ├── results_random_regular.csv
│   └── results_toy.csv
└── plots/                    # Generated figures
    ├── plot1_2_vs_vertices.png
    ├── plot3_pruning_effect.png
    ├── plot4_ablation_study.png
    ├── plot5_density_study.png
    └── plot6_threshold_behavior.png
```

## Classical Solver Improvements (WP1)

The improved solver adds the following enhancements over the baseline:

| Feature | Baseline | Improved |
|---------|----------|----------|
| Forced-neighbor propagation | ✗ | ✓ |
| Preprocessing (isolated vertex + degree-1) | ✗ | ✓ |
| Maximal matching lower bound | ✗ | ✓ |
| High-degree branching heuristic | ✗ | ✓ |

### Forced-Neighbor Propagation
When a vertex `v` is excluded from the cover (set to 0), all its neighbors must be included (set to 1), since every edge incident to `v` must be covered by the other endpoint. Applied recursively.

### Preprocessing Rules
- **Isolated vertex deletion**: vertices with no uncovered edges are removed
- **Degree-1 rule**: if a vertex has exactly one unassigned neighbor, that neighbor is included in the cover

### Maximal Matching Lower Bound
A greedy maximal matching on the residual subgraph provides a tighter lower bound than simply counting selected vertices.

## Experiment Results

**1,530 runs** across 5 graph families, verifying exactness across combinations of density and size using 6 distinct configurations:
- `baseline`, `improved`, `raw_emulation`, `ablation_prep`, `ablation_prep_lb`, `enhanced_emulation`.

### Key Findings

The enhanced emulation perfectly mirrors the mathematically sound optimal covers verified by the baseline solver, while demonstrating an exponential reduction in explored nodes:
- The **ablation study** isolates the contribution of each heuristic, proving that lower bounds and intelligent branching severely curtail the search space.
- The **threshold-search behavior** validates the algorithm dynamically adjusting the cost bounds to shrink the generated tree dimension.

### Generated Plots

| Plot | Description |
|------|-------------|
| Plot 1 & 2 | Scaling of runtime and explored nodes as $n$ increases |
| Plot 3 | Log-log scatter plot of search tree nodes before vs. after pruning rules |
| Plot 4 | Bar-chart ablation study isolating algorithmic impact |
| Plot 5 | Analysis of solver performance varying density parameters $p$ |
| Plot 6 | Line plot tracking dynamic threshold $C$ against $|\mathcal{T}_C|$ |

## Clarification on Quantum Claims

To maintain mathematical rigor, this implementation does not execute Montanaro's original phase estimation and quantum tree search on a quantum circuit (due to qubit limitations for meaningful graph sizes). Instead, it **classically emulates** the logic—structuring the search tree identically to the quantum version, applying bounds, and systematically finding the marked states. All classical heuristic rules (preprocessing, forced-neighbor propagation, etc.) have been proven to preserve the exactness of the problem.

## Requirements

```
networkx
numpy
matplotlib
pandas
qiskit
qiskit-aer
```

## Quick Start

```bash
# Install dependencies
pip install networkx numpy matplotlib pandas qiskit qiskit-aer

# Run batch experiments (generates CSV files)
python batch_experiments.py

# Generate plots from results
python generate_plots.py
```

## Reproducibility

All experiments use fixed random seeds (`{42, 123, 256, 789, 1024}`). Running `batch_experiments.py` reproduces the exact same results.

## License

MIT

## Authors

Research project on quantum optimization algorithms for combinatorial problems.
