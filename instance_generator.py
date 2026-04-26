"""
instance_generator.py
Generates graphs for Minimum Vertex Cover.
"""

import networkx as nx
import random

def generate_mvc_instance(model="erdos_renyi", n=10, seed=42, p=0.4, m=2, k=4, d=3):
    """
    Generates a graph instance for the MVC problem
    
    Parameters:
    - model : type of graph (erdos_renyi, barabasi_albert, watts_strogatz, regular, toy_5, toy_8)
    - n     : number of nodes (except for toy graphs)
    - seed  : random seed for reproducibility
    - p, m, k, d : parameters specific to certain graph models (probability for erdos_renyi, etc)
    """
    random.seed(seed)
    
    #create the graph
    if model == "erdos_renyi":
        G = nx.erdos_renyi_graph(n, p, seed=seed)
        
    elif model == "barabasi_albert":
        G = nx.barabasi_albert_graph(n, m, seed=seed)
        
    elif model == "watts_strogatz":
        G = nx.watts_strogatz_graph(n, k, p, seed=seed)
        
    elif model == "regular":
        G = nx.random_regular_graph(d, n, seed=seed)
        
    elif model == "toy_5":
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)])
        n = 5
        
    elif model == "toy_8":
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (3, 6), (4, 6), (5, 7), (6, 7), (2, 7)])
        n = 8
        
    elif model == "toy_9_mvc_2":
        G = nx.Graph()
        G.add_edges_from([
            (0, 1), 
            (0, 2), (0, 3), (0, 4), (0, 5), 
            (1, 6), (1, 7), (1, 8), 
        ])
        n = 9
        
    elif model == "toy_11_mvc_3":
        G = nx.Graph()
        G.add_edges_from([
            (0, 1), 
            (0, 2), (0, 3), (0, 4), (0, 5), 
            (1, 6), (1, 7), (1, 8), (2,3), (2,9), (2,10)
        ])
        n = 11
        
    elif model == "toy_15_star":
        G = nx.Graph()
        G.add_nodes_from(range(15))
        G.add_edges_from([
            (0, 5), (1, 5), (2, 5), (3, 5), (4, 5),
            (5, 6), (5, 7), (5, 8), (5, 9), (5, 10),
            (5, 11), (5, 12), (5, 13), (5, 14)
        ])
        n = 15
        
    elif model == "greedy_trap_30":
        G = nx.Graph()
        A = list(range(11))
        B_degrees = [11, 10, 9, 8, 7, 6, 5, 5, 4, 4, 3, 3, 3, 2, 2, 2, 2, 2, 1]
        
        G.add_nodes_from(A, bipartite=0)
        
        B_start = 11
        A_index = 0
        
        for i, deg in enumerate(B_degrees):
            b_node = B_start + i
            G.add_node(b_node, bipartite=1)
            for _ in range(deg):
                G.add_edge(b_node, A[A_index % 11])
                A_index += 1
        
        n = 30

    #nodes are ordrered 0,1,2,3... etc
    G = nx.convert_node_labels_to_integers(G)
    
    #cost 1 for all nodes (we can easily modify this later if we want to test with different costs)
    nx.set_node_attributes(G, 1, name="cost")


    instance = {
        'graph': G,
        'metadata': {
            'problem': 'Minimum_Vertex_Cover',
            'model': model,
            'n_nodes': n,
            'seed': seed
        }
    }
    
    return instance


def get_exact_mvc_solution(graph):
    """
    finds the exact solution for the MVC problem using the fact that the complement of a vertex cover is an independent set
    so that we can verify our results
    """
    G_comp = nx.complement(graph)
    cliques = list(nx.find_cliques(G_comp))
    max_indep_set = max(cliques, key=len)
    vrai_mvc = set(graph.nodes()) - set(max_indep_set)
    return sorted(list(vrai_mvc))