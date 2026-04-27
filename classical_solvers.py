"""
classical_solvers.py

Contains both baseline and improved exact branch-and-bound solvers for
the Minimum Vertex Cover (MVC) problem.

Baseline solver:
    - branches on one undecided vertex
    - uses count of selected vertices as lower bound
    - prunes if both endpoints of an edge are fixed to 0

Improved solver adds:
    1. Forced-neighbor propagation
    2. Preprocessing (isolated vertex deletion + degree-1 rule)
    3. Maximal matching lower bound
"""

import heapq
import itertools
import time
import networkx as nx
from problem_encoding import cost, branch, is_solution, decode_solution


# ──────────────────────────────────────────────────────────────────────
# Greedy approximation (unchanged)
# ──────────────────────────────────────────────────────────────────────

def greedy_mvc(graph):
    """
    selects iteratively the vertex with the highest degree and removes its edges until the graph has no more edges
    so in the MVC we take each time the most connected vertex

    returns a list of vertices in the vertex cover and its cost
    """
    G_copy = graph.copy()
    cover = []

    while G_copy.number_of_edges() > 0:
        #find the vertex with the highest degree
        node = max(G_copy.nodes(), key=lambda x: G_copy.degree(x))
        cover.append(node)
        
        #delete the node and its edges from the graph
        G_copy.remove_node(node)

    return sorted(cover), len(cover)

# ──────────────────────────────────────────────────────────────────────
# Brute force solver (unchanged)
# ──────────────────────────────────────────────────────────────────────

def brute_force_mvc(graph):
    """
    Generates all possible subsets of vertices in increasing order of size
    The first subset that covers all edges is guaranteed to be the minimum vertex cover
    """
    nodes = list(graph.nodes())
    edges = list(graph.edges())

    def is_valid_cover(cover_set):
        #check if every edge has at least one endpoint in the cover set
        for u, v in edges:
            if u not in cover_set and v not in cover_set:
                return False
        return True

    #test all combinations of size r (from 0 to the number of nodes)
    for r in range(len(nodes) + 1):
        for subset in itertools.combinations(nodes, r):
            cover_set = set(subset)
            if is_valid_cover(cover_set):
                #the first valid cover is the minimum
                return sorted(list(subset)), r

    return [], 0


# ======================================================================
# WP1 — Improved exact branch-and-bound solver for MVC
# ======================================================================


# ──────────────────────────────────────────────────────────────────────
# WP1.1 — Forced-neighbor propagation
# ──────────────────────────────────────────────────────────────────────
# Rule: if vertex v is fixed to 0, every neighbor of v must be fixed to 1
# (because every edge incident to v must be covered by the other endpoint).
# This is applied recursively: forcing a neighbor to 1 does not trigger
# new propagation, but if the branching later fixes another vertex to 0,
# propagation runs again.

def forced_neighbor_propagation(state, graph, stats):
    """
    Propagate forced assignments: if v is set to 0, all its neighbors
    must be set to 1.  Applies recursively until no new assignments.

    Parameters
    ----------
    state : dict  — current partial assignment {vertex: 0 or 1}
    graph : nx.Graph
    stats : dict  — mutable counter dict; increments 'forced_assignments'

    Returns
    -------
    state : dict  — updated assignment (or None if conflict detected)
    """
    changed = True
    while changed:
        changed = False
        for v in list(state.keys()):
            if state[v] != 0:
                continue
            for u in graph.neighbors(v):
                if u not in state:
                    state[u] = 1
                    stats['forced_assignments'] += 1
                    changed = True
                elif state[u] == 0:
                    # Conflict: both endpoints are 0 => infeasible
                    return None
    return state


# ──────────────────────────────────────────────────────────────────────
# WP1.2 — Preprocessing rules
# ──────────────────────────────────────────────────────────────────────

# Rule A: Isolated vertex deletion
# If a vertex has degree 0 in the residual graph (all neighbors already
# assigned), it covers no edge — exclude it (set to 0) and remove.
#
# Rule B: Degree-1 rule
# If vertex u has exactly one unassigned neighbor v, include v in the
# cover (set v=1).  The edge {u,v} is then covered.

def preprocessing(state, graph, stats):
    """
    Apply isolated-vertex deletion and degree-1 reduction rules
    repeatedly until no more changes.

    Parameters
    ----------
    state : dict
    graph : nx.Graph
    stats : dict — increments 'preprocessing_reductions' and
                   'vertices_removed_by_preprocessing'

    Returns
    -------
    state : dict (updated) or None if infeasible
    """
    changed = True
    while changed:
        changed = False
        for v in list(graph.nodes()):
            if v in state:
                continue

            # Compute the residual neighbors (unassigned vertices adjacent to v)
            residual_neighbors = [u for u in graph.neighbors(v) if u not in state]

            # Also check: does v have any uncovered edge?
            # An edge is uncovered if neither endpoint is set to 1
            uncovered_neighbors = []
            for u in graph.neighbors(v):
                if state.get(u) == 1:
                    continue  # edge already covered by u
                uncovered_neighbors.append(u)

            # Rule A: if v has no uncovered edges, it is effectively isolated
            if len(uncovered_neighbors) == 0:
                state[v] = 0
                stats['vertices_removed_by_preprocessing'] += 1
                stats['preprocessing_reductions'] += 1
                changed = True
                continue

            # Rule B: degree-1 rule on residual graph
            if len(residual_neighbors) == 1:
                # v has exactly one unassigned neighbor w
                w = residual_neighbors[0]
                # Include w in the cover (it covers the edge {v, w} and possibly more)
                if w not in state:
                    state[w] = 1
                    stats['preprocessing_reductions'] += 1
                    changed = True
                # Now v may become isolated — will be handled by Rule A next iteration

    return state


# ──────────────────────────────────────────────────────────────────────
# WP1.3 — Maximal matching lower bound
# ──────────────────────────────────────────────────────────────────────
# A maximal matching in the residual (uncovered) graph gives a lower
# bound on the number of additional vertices needed: each matching edge
# requires at least one new cover vertex.

def maximal_matching_lower_bound(state, graph):
    """
    Compute a lower bound on the remaining cover size using a greedy
    maximal matching on the residual (uncovered) subgraph.

    Returns
    -------
    int — lower bound on additional vertices needed
    """
    # Build the residual subgraph: only edges not yet covered
    matched = set()
    matching_size = 0

    for u, v in graph.edges():
        # Edge is covered if at least one endpoint is in the cover
        if state.get(u) == 1 or state.get(v) == 1:
            continue
        # Edge is infeasible if both endpoints are excluded
        if state.get(u) == 0 and state.get(v) == 0:
            return float('inf')
        # Both unassigned, or one unassigned and the other not yet decided
        if u not in matched and v not in matched:
            matched.add(u)
            matched.add(v)
            matching_size += 1

    return matching_size


def trivial_lower_bound(state, graph):
    """
    Baseline lower bound: just the count of already-selected vertices.
    """
    return 0  # no additional estimate beyond the selected count


def choose_branching_vertex(unassigned, state, graph):
    """
    Choose the next vertex to branch on from the unassigned list.
    Heuristic: pick the vertex with the highest residual degree.
    """
    return max(unassigned, key=lambda u: sum(
        1 for w in graph.neighbors(u) if w not in state
    ))

# ──────────────────────────────────────────────────────────────────────
# WP1.4 — Branch-and-bound solvers (baseline + improved)
# ──────────────────────────────────────────────────────────────────────

def _count_selected(state):
    """Number of vertices currently set to 1."""
    return sum(1 for v in state.values() if v == 1)


def bb_mvc_baseline(graph, timeout=300):
    """
    Baseline branch-and-bound solver for MVC.
    
    Uses:
      - simple branching on the first unassigned vertex
      - lower bound = number of selected vertices (trivial)
      - prunes if both endpoints of an edge are fixed to 0

    Returns
    -------
    dict with keys:
        'cover': list of vertices in the optimal cover
        'cover_size': int
        'nodes_explored': int
        'nodes_pruned': int
        'max_depth': int
        'forced_assignments': 0  (not used in baseline)
        'preprocessing_reductions': 0
        'root_lower_bound': int
        'runtime': float (seconds)
        'finished': bool
    """
    start_time = time.time()
    n = graph.number_of_nodes()
    
    best_cover_size = n  # worst case: all vertices
    best_state = {v: 1 for v in graph.nodes()}

    stats = {
        'nodes_explored': 0,
        'nodes_pruned': 0,
        'max_depth': 0,
        'forced_assignments': 0,
        'preprocessing_reductions': 0,
        'vertices_removed_by_preprocessing': 0,
        'root_lower_bound': 0,
        'finished': True,
    }

    # Stack-based DFS: (state, depth)
    stack = [({}, 0)]

    while stack:
        if time.time() - start_time > timeout:
            stats['finished'] = False
            break

        state, depth = stack.pop()
        stats['nodes_explored'] += 1
        stats['max_depth'] = max(stats['max_depth'], depth)

        selected = _count_selected(state)

        # Check for infeasibility (both endpoints = 0)
        infeasible = False
        for u, v in graph.edges():
            if state.get(u) == 0 and state.get(v) == 0:
                infeasible = True
                break
        if infeasible:
            stats['nodes_pruned'] += 1
            continue

        # Trivial lower bound: selected count
        lb = selected
        if depth == 0:
            stats['root_lower_bound'] = lb

        if lb >= best_cover_size:
            stats['nodes_pruned'] += 1
            continue

        # Check if all vertices assigned
        if len(state) == n:
            if selected < best_cover_size:
                best_cover_size = selected
                best_state = state.copy()
            continue

        # Branch on first unassigned vertex
        unassigned = [v for v in graph.nodes() if v not in state]
        if not unassigned:
            continue
        v = unassigned[0]

        # Push exclude-branch first (DFS will explore include-branch first)
        state_out = state.copy()
        state_out[v] = 0
        stack.append((state_out, depth + 1))

        state_in = state.copy()
        state_in[v] = 1
        stack.append((state_in, depth + 1))

    elapsed = time.time() - start_time
    cover = [v for v, s in best_state.items() if s == 1]

    return {
        'cover': sorted(cover),
        'cover_size': best_cover_size,
        'nodes_explored': stats['nodes_explored'],
        'nodes_pruned': stats['nodes_pruned'],
        'max_depth': stats['max_depth'],
        'forced_assignments': stats['forced_assignments'],
        'preprocessing_reductions': stats['preprocessing_reductions'],
        'vertices_removed_by_preprocessing': stats['vertices_removed_by_preprocessing'],
        'root_lower_bound': stats['root_lower_bound'],
        'runtime': elapsed,
        'finished': stats['finished'],
    }


def bb_mvc_improved(graph, timeout=300):
    """
    Improved branch-and-bound solver for MVC.

    Enhancements over baseline:
      1. Forced-neighbor propagation (if v=0, neighbors -> 1)
      2. Preprocessing (isolated vertex deletion + degree-1 rule)
      3. Maximal matching lower bound

    Returns the same dict structure as bb_mvc_baseline.
    """
    start_time = time.time()
    n = graph.number_of_nodes()

    best_cover_size = n
    best_state = {v: 1 for v in graph.nodes()}

    stats = {
        'nodes_explored': 0,
        'nodes_pruned': 0,
        'max_depth': 0,
        'forced_assignments': 0,
        'preprocessing_reductions': 0,
        'vertices_removed_by_preprocessing': 0,
        'root_lower_bound': 0,
        'finished': True,
    }

    # Stack-based DFS: (state, depth)
    stack = [({}, 0)]

    while stack:
        if time.time() - start_time > timeout:
            stats['finished'] = False
            break

        state, depth = stack.pop()
        stats['nodes_explored'] += 1
        stats['max_depth'] = max(stats['max_depth'], depth)

        # --- WP1.1: Forced-neighbor propagation ---
        state = forced_neighbor_propagation(state, graph, stats)
        if state is None:
            stats['nodes_pruned'] += 1
            continue

        # --- WP1.2: Preprocessing ---
        state = preprocessing(state, graph, stats)
        if state is None:
            stats['nodes_pruned'] += 1
            continue

        selected = _count_selected(state)

        # --- WP1.3: Maximal matching lower bound ---
        remaining_lb = maximal_matching_lower_bound(state, graph)
        if remaining_lb == float('inf'):
            stats['nodes_pruned'] += 1
            continue

        lb = selected + remaining_lb

        if depth == 0:
            stats['root_lower_bound'] = lb

        if lb >= best_cover_size:
            stats['nodes_pruned'] += 1
            continue

        # Check if all vertices assigned
        if len(state) == n:
            if selected < best_cover_size:
                best_cover_size = selected
                best_state = state.copy()
            continue

        # Branch on first unassigned vertex
        unassigned = [v for v in graph.nodes() if v not in state]
        if not unassigned:
            # All assigned already (by propagation / preprocessing)
            if selected < best_cover_size:
                best_cover_size = selected
                best_state = state.copy()
            continue

        # Choose vertex to branch on: pick vertex with highest residual degree
        # (heuristic: branching on high-degree vertex tends to reduce tree faster)
        v = choose_branching_vertex(unassigned, state, graph)

        # Push exclude-branch first (DFS processes include-branch first)
        state_out = state.copy()
        state_out[v] = 0
        stack.append((state_out, depth + 1))

        state_in = state.copy()
        state_in[v] = 1
        stack.append((state_in, depth + 1))

    elapsed = time.time() - start_time
    cover = [v for v, s in best_state.items() if s == 1]

    return {
        'cover': sorted(cover),
        'cover_size': best_cover_size,
        'nodes_explored': stats['nodes_explored'],
        'nodes_pruned': stats['nodes_pruned'],
        'max_depth': stats['max_depth'],
        'forced_assignments': stats['forced_assignments'],
        'preprocessing_reductions': stats['preprocessing_reductions'],
        'vertices_removed_by_preprocessing': stats['vertices_removed_by_preprocessing'],
        'root_lower_bound': stats['root_lower_bound'],
        'runtime': elapsed,
        'finished': stats['finished'],
    }
