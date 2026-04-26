"""
quantum_solvers.py
implementation of quantum algorithms (the two subroutines and Montanaro algorithm)
the two subroutines are also implemented classically to validate the logic on larger graphs where we can't run real quantum circuits

"""

import math
import numpy as np
import networkx as nx
import warnings

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.circuit.library import QFT, UnitaryGate

from problem_encoding import cost, branch, is_solution

warnings.filterwarnings("ignore", category=DeprecationWarning)

_TREE_CACHE = {}

def get_cached_tree(root_state, graph, c_limit, T_limit):
    state_key = str(sorted(root_state.items()))
    key = (state_key, c_limit, T_limit)
    if key not in _TREE_CACHE:
        _TREE_CACHE[key] = build_oracle_tree(root_state, graph, c_limit, T_limit)
    return _TREE_CACHE[key]


def build_oracle_tree(root_state, graph, c_limit, T_limit):
    """
    generates the tree of the problem
    Prunes branches whose cost exceeds c_limit 
    stops if the tree becomes too large for the allocated number of qubits (T_limit).
    """
    G_dir = nx.DiGraph()
    queue = [(root_state, 0)]
    G_dir.add_node(0, state=root_state)
    next_id = 1
    
    while queue:
        curr_state, curr_id = queue.pop(0)
        
        if G_dir.number_of_nodes() > T_limit:
            return "overflow", None
            
        if is_solution(curr_state, graph):
            continue
            
        enfants_states = branch(curr_state, graph)
        for child_state in enfants_states:
            if cost(child_state, graph) <= c_limit:
                child_id = next_id
                next_id += 1
                
                G_dir.add_node(child_id, state=child_state)
                G_dir.add_edge(curr_id, child_id)
                queue.append((child_state, child_id))
                
                if G_dir.number_of_nodes() > T_limit:
                    return "overflow", None
                    
    return "Ok", G_dir


#version 1 : real quantum circuits (with qiskit)

def Count_quantum_dynamic(root_state, graph, c_limit, T0, n_depth, delta=0.5, epsilon=0.1):
    T_limit = math.floor((1 + delta) * T0)
    status, G_tree = get_cached_tree(root_state, graph, c_limit, T_limit)
    
    if status == "overflow": return "contains more than T0 nodes"
        
    real_edges = list(G_tree.edges())
    if len(G_tree.nodes()) <= 1: return 1

    alpha = np.sqrt(2 * n_depth / delta)
    delta_min = (delta**1.5) / (24 * np.sqrt(3 * n_depth * T0))
    t_eval = min(5, max(3, math.ceil(-math.log2(delta_min)) + 1))
    dim = 2 ** max(1, math.ceil(math.log2(T_limit + 1)))

    root_id = 0
    distances = nx.single_source_shortest_path_length(G_tree, root_id)
    VA = [node for node, dist in distances.items() if dist % 2 == 0]
    VB = [node for node, dist in distances.items() if dist % 2 != 0]

    def get_incident_edges(v):
        return [i + 1 for i, (src, dst) in enumerate(real_edges) if src == v or dst == v]

    #RA
    total_proj_A = np.zeros((dim, dim), dtype=complex)
    for v in VA:
        inc = get_incident_edges(v)
        if v == root_id: inc.append(0) 
        sv = np.zeros(dim, dtype=complex)
        for e in inc:
            if e >= dim: return "contains more than T0 nodes"
            if e == 0 and v == root_id: sv[e] = 1.0    
            elif e != 0 and v == root_id: sv[e] = alpha   
            else: sv[e] = 1.0    
        norm = np.linalg.norm(sv)
        if norm > 0: 
            sv = sv / norm
            total_proj_A += np.outer(sv, sv.conj())
    RA = np.eye(dim, dtype=complex) - 2 * total_proj_A

    #RB
    total_proj_B = np.zeros((dim, dim), dtype=complex)
    for v in VB:
        inc = get_incident_edges(v)
        sv = np.zeros(dim, dtype=complex)
        for e in inc:
            if e >= dim: return "contains more than T0 nodes"
            sv[e] = 1.0
        norm = np.linalg.norm(sv)
        if norm > 0: 
            sv = sv / norm
            total_proj_B += np.outer(sv, sv.conj())
    RB = np.eye(dim, dtype=complex) - 2 * total_proj_B

    U = RB @ RA

    q_eval = QuantumRegister(t_eval, 'eval')
    q_state = QuantumRegister(math.ceil(math.log2(dim)), 'state') 
    c_meas = ClassicalRegister(t_eval, 'meas')
    qc = QuantumCircuit(q_eval, q_state, c_meas)

    qc.h(q_eval)
    U_power = U
    for j in range(t_eval):
        cU_gate = UnitaryGate(U_power, label=f"U^{2**j}").control(1)
        qc.append(cU_gate, [q_eval[j]] + list(q_state))
        U_power = U_power @ U_power

    qc.append(QFT(num_qubits=t_eval, inverse=True, do_swaps=False).to_gate(), q_eval)
    qc.measure(q_eval, c_meas)

    sim = AerSimulator(method='matrix_product_state')
    t_reps = math.ceil((9/4) * math.log(2 / epsilon))
    
    counts = sim.run(transpile(qc, sim), shots=t_reps).result().get_counts()


    phases = []
    for bitstring, count in counts.items():
        bitstring = bitstring[::-1]
        phase_fraction = int(bitstring, 2) / (2**t_eval)
        if phase_fraction > 0.5: phase_fraction = 1.0 - phase_fraction
        theta = 2 * np.pi * phase_fraction
        if theta > 0.001: phases.extend([theta] * count)

    if phases:
        T_est = 1 / ( (alpha**2) * (np.sin(min(phases) / 2)**2) )
        if T_est > T_limit: return "contains more than T0 nodes"
        return T_est
    return "contains more than T0 nodes"


def Search_quantum_dynamic(root_state, graph, c_limit, T_bound, n_bound, K_shots=32, delta=0.5):
    T_limit = math.floor((1 + delta) * T_bound)
    status, G_sub = get_cached_tree(root_state, graph, c_limit, T_limit)
    
    if status == "overflow": return False
        
    target_nodes = [n for n, attr in G_sub.nodes(data=True) if is_solution(attr['state'], graph)]
    if len(G_sub.nodes()) <= 1: return 0 in target_nodes

    dim = 2 ** max(1, math.ceil(math.log2(T_limit)))
    distances = nx.single_source_shortest_path_length(G_sub, 0)
    VA = [node for node, d in distances.items() if d % 2 == 0]
    VB = [node for node, d in distances.items() if d % 2 != 0]

    total_proj_A = np.zeros((dim, dim), dtype=complex)
    for x in VA:
        if x >= dim: return False
        if x in target_nodes: continue
        children_x = list(G_sub.successors(x))
        if any(y >= dim for y in children_x): return False
        
        psi = np.zeros(dim, dtype=complex)
        if x == 0:
            norm = np.sqrt(1 + len(children_x) * n_bound)
            psi[x] = 1.0 / norm
            for y in children_x: psi[y] = np.sqrt(n_bound) / norm
        else:
            dx = len(children_x) + 1
            psi[x] = 1.0 / np.sqrt(dx)
            for y in children_x: psi[y] = 1.0 / np.sqrt(dx)
        total_proj_A += np.outer(psi, psi.conj())
    RA = np.eye(dim, dtype=complex) - 2 * total_proj_A

    total_proj_B = np.zeros((dim, dim), dtype=complex)
    for x in VB:
        if x >= dim: return False
        if x in target_nodes: continue
        children_x = list(G_sub.successors(x))
        if any(y >= dim for y in children_x): return False
        
        dx = len(children_x) + 1
        psi = np.zeros(dim, dtype=complex)
        psi[x] = 1.0 / np.sqrt(dx)
        for y in children_x: psi[y] = 1.0 / np.sqrt(dx)
        total_proj_B += np.outer(psi, psi.conj())
    RB = np.eye(dim, dtype=complex) - 2 * total_proj_B

    U = RB @ RA
    
    t_eval = min(5, max(3, math.ceil(-math.log2(1.0 / math.sqrt(max(1, T_bound * n_bound)))) + 1))
    
    q_eval = QuantumRegister(t_eval, 'eval')
    q_state = QuantumRegister(math.ceil(math.log2(dim)), 'state') 
    c_meas = ClassicalRegister(t_eval, 'meas')
    qc = QuantumCircuit(q_eval, q_state, c_meas)

    qc.h(q_eval)
    U_power = U
    for j in range(t_eval):
        cU_gate = UnitaryGate(U_power, label=f"U^{2**j}").control(1)
        qc.append(cU_gate, [q_eval[j]] + list(q_state))
        U_power = U_power @ U_power

    qc.append(QFT(num_qubits=t_eval, inverse=True, do_swaps=False).to_gate(), q_eval)
    qc.measure(q_eval, c_meas)

    sim = AerSimulator(method='matrix_product_state')
    counts = sim.run(transpile(qc, sim), shots=K_shots).result().get_counts()
    return counts.get('0' * t_eval, 0) >= (3*K_shots / 8)


def Find_marked_state_dynamic(root_state, graph, c_limit, T_bound, n_bound, delta=0.5):
    current_state = root_state.copy()
    while not is_solution(current_state, graph):
        enfants = branch(current_state, graph)
        found_good_branch = False
        for enfant in enfants:
            if cost(enfant, graph) <= c_limit:
                if Search_quantum_dynamic(enfant, graph, c_limit, T_bound, n_bound, delta=delta):
                    current_state = enfant
                    found_good_branch = True
                    break
        if not found_good_branch: break
    return current_state


def Montanaro_BB_MVC(graph):
    """Montanaro's algorithm
    slow so only possible to run on small graphs
    """
    global _TREE_CACHE
    _TREE_CACHE.clear()

    N = graph.number_of_nodes()
    n_depth = N  
    c_max = 2 ** math.ceil(math.log2(N)) 
    T_max = (2 ** (N + 1)) - 1
    T = 1
    c_old = 0
    root_state = {}
    
    print(f"\nStarting Montanaro_BB_MVC | Nodes: {N}")
    
    while T <= T_max:
        print(f"\nTree size limit T = {T}")
        if T > T_max / 2: 
            c_new = c_max
        else:
            c_new = 0
            for i in range(1, int(math.log2(c_max)) + 1):
                test_c = c_new + c_max / (2**i)
                if Count_quantum_dynamic(root_state, graph, test_c, T, n_depth) != "contains more than T0 nodes":
                    c_new = test_c
        print(f"Estimated cost (c_new): {c_new}")

        if Search_quantum_dynamic(root_state, graph, c_new, T, n_depth):
            low = math.floor(c_old)
            high = math.ceil(c_new)
            print(f"Solution detected. Starting binary search in [{low}, {high}]")
            while low < high: 
                mid = (low + high) // 2 
                print(f"Testing cost mid = {mid}... ", end="")
                if Search_quantum_dynamic(root_state, graph, mid, T, n_depth): 
                    high = mid
                    print("Success")
                else: 
                    low = mid + 1
                    print("Failed")
            
            print(f"Optimal MVC cost found: {low}. Recovering state...")
            etat_final = Find_marked_state_dynamic(root_state, graph, low, T, n_depth)
            return low, etat_final
            
        print("No solution found. Doubling T.")
        T = 2 * T
        c_old = c_new
    return "no solution", None


#version 2 : classical implementation of the two subroutines to validate the logic on larger graphs
def Count_fast(root_state, graph, c_limit, T0, delta=0.5):
    T_limit = math.floor((1 + delta) * T0)
    status, G_tree = build_oracle_tree(root_state, graph, c_limit, T_limit)
    if status == "overflow": return "contains more than T0 nodes"
    return G_tree.number_of_nodes()

def Search_fast(root_state, graph, c_limit, T_bound, delta=0.5):
    T_limit = math.floor((1 + delta) * T_bound)
    status, G_sub = build_oracle_tree(root_state, graph, c_limit, T_limit)
    if status == "overflow": return False
    return any(is_solution(attr['state'], graph) for n, attr in G_sub.nodes(data=True))

def Find_marked_state_fast(root_state, graph, c_limit, T_bound, delta=0.5):
    current_state = root_state.copy()
    while not is_solution(current_state, graph):
        enfants = branch(current_state, graph)
        found_good_branch = False
        for enfant in enfants:
            if cost(enfant, graph) <= c_limit:
                if Search_fast(enfant, graph, c_limit, T_bound, delta):
                    current_state = enfant
                    found_good_branch = True
                    break
        if not found_good_branch: break
    return current_state

def Montanaro_BB_MVC_Fast(graph):
    N = graph.number_of_nodes()
    c_max = 2 ** math.ceil(math.log2(N)) 
    T_max = (2 ** (N + 1)) - 1
    T = 1
    c_old = 0
    root_state = {}
    
    print(f"\nStarting Montanaro_BB_MVC_Fast | Nodes: {N}")
    
    while T <= T_max:
        print(f"\nTree size limit T = {T}")
        if T > T_max / 2: 
            c_new = c_max
        else:
            c_new = 0
            for i in range(1, int(math.log2(c_max)) + 1):
                test_c = c_new + c_max / (2**i)
                if Count_fast(root_state, graph, test_c, T) != "contains more than T0 nodes":
                    c_new = test_c
        print(f"Estimated cost (c_new): {c_new}")

        if Search_fast(root_state, graph, c_new, T):
            low = math.floor(c_old)
            high = math.ceil(c_new)
            print(f"Solution detected. Starting binary search in [{low}, {high}]")
            while low < high: 
                mid = (low + high) // 2 
                print(f"Testing cost mid = {mid}... ", end="")
                if Search_fast(root_state, graph, mid, T): 
                    high = mid
                    print("Success")
                else: 
                    low = mid + 1
                    print("Failed")
            
            print(f"Optimal MVC cost found: {low}. Recovering state...")
            etat_final = Find_marked_state_fast(root_state, graph, low, T)
            return low, etat_final
            
        print("No solution found. Doubling T.")
        T = 2 * T
        c_old = c_new
    return "no solution", None