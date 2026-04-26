# Quantum Branch-and-Bound for Minimum Vertex Cover

Implementation of a Montanaro-style quantum branch-and-bound algorithm for the Minimum Vertex Cover (MVC) problem, with an improved classical exact solver and comprehensive benchmarking.

## Overview

This project implements and benchmarks:

1. **Quantum Branch-and-Bound** — Montanaro's algorithm with quantum subroutines (tree size estimation via phase estimation, quantum tree search)
2. **Classical Exact Solvers** — Both a baseline and improved branch-and-bound solver
3. **Approximation** — Greedy MVC heuristic
4. **Benchmarking Pipeline** — Batch experiments across 5 graph families with full instrumentation

## Project Structure

```
├── classical_solvers.py      # Baseline & improved B&B solvers (WP1)
├── quantum_solvers.py        # Montanaro quantum B&B (Qiskit)
├── problem_encoding.py       # MVC cost, branching, feasibility
├── instance_generator.py     # Graph generation (ER, BA, WS, regular, toy)
├── visualization.py          # Graph and solution plotting
├── batch_experiments.py      # Batch experiment runner (WP2)
├── generate_plots.py         # Paper-ready plot generation (WP2)
├── main_workflow.ipynb       # Interactive notebook (quantum experiments)
├── EXPERIMENT_README.md      # Detailed experiment documentation
├── results/                  # CSV experiment data
│   ├── results_all.csv           # Merged (450 runs)
│   ├── results_erdos_renyi.csv
│   ├── results_barabasi_albert.csv
│   ├── results_watts_strogatz.csv
│   ├── results_random_regular.csv
│   └── results_toy.csv
└── plots/                    # Generated figures
    ├── plot1_runtime_comparison.png
    ├── plot2_nodes_comparison.png
    ├── plot3_pruning_ratio.png
    ├── plot4_preprocessing_effect.png
    └── plot5_runtime_vs_cover.png
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

## Experiment Results (WP2)

**450 runs** across 225 graph instances (5 families × multiple sizes/params × 5 seeds) × 2 solver modes.

### Key Findings

The improved solver achieves **100–5000× fewer explored nodes** and **1–3 orders of magnitude speedup** over the baseline:

| Instance | Baseline Nodes | Improved Nodes | Reduction |
|----------|---------------|----------------|-----------|
| BA n=25, m=1 | 16,511 | 3 | 5,503× |
| ER n=25, p=0.5 | 31,643 | 9 | 3,516× |
| Regular n=24, d=3 | 54,927 | 29 | 1,894× |
| WS n=25, k=4, p=0.1 | 34,129 | 91 | 375× |
| Star n=15 | 265 | 1 | 265× |

### Plots

| Plot | Description |
|------|-------------|
| Plot 1 | Runtime: baseline vs improved (log-log scatter) |
| Plot 2 | Explored nodes: baseline vs improved |
| Plot 3 | Pruning ratio by graph family and density |
| Plot 4 | Speedup and node reduction analysis |
| Plot 5 | Runtime vs optimal cover size |

## Quantum Solver

The quantum solver implements Montanaro's quantum branch-and-bound framework:

- **Quantum tree size estimation** via phase estimation on quantum walk operators
- **Quantum tree search** using the same walk to detect marked (solution) states
- **Binary search** on the cost parameter to find the optimal MVC
- Both **real quantum circuit** (Qiskit/AerSimulator) and **fast classical** emulations

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
