"""
problem_encoding.py

"""

def cost(state, graph):
    """
    Calculates the lower bound of the cost for a partial state
    
    - state (dict): a dictionary {node: 1 (included) or 0 (excluded)}.
    - graph (nx.Graph): the graph of the instance.
        
    returns the evaluated cost
    """

    #if an edge has both nodes excluded then the vertex cover is impossible
    for u, v in graph.edges():
        if state.get(u) == 0 and state.get(v) == 0:
            return float('inf') 
    
    #the lower bound is the number of nodes already included in the vertex cover
    return sum(1 for status in state.values() if status == 1)

def branch(state, graph):
    """
    generates the children of a node in the tree by assigning the next available vertex

    - state (dict): a dictionary {node: 1 (included) or 0 (excluded)}.
    - graph (nx.Graph): the graph of the instance.
        
    returns a list of new states
    
    """
    #we look for the first vertex in the graph that has not yet been assigned 0 or 1
    unassigned_nodes = [n for n in graph.nodes() if n not in state]
    
    #if all nodes have been assigned, we are at a leaf so we don't branch further
    if not unassigned_nodes:
        return []
        
    next_node = unassigned_nodes[0]
    
    #create two new states for the two branches of the tree (one where we include the next node in the vertex cover, and one where we exclude it)
    state_in = state.copy()
    state_in[next_node] = 1  
    
    state_out = state.copy()
    state_out[next_node] = 0 
    
    return [state_in, state_out]


def is_solution(state, graph):
    """
    verifies if a state is a valid solution leaf of the problem.
    """
    #a solution is valid if all nodes have been assigned and the cost is not infinite (no edge is left uncovered)
    
    return len(state) == graph.number_of_nodes() and cost(state, graph) != float('inf')

def decode_solution(state):
    """
    transforms the binary state dictionary into a list of vertices in the vertex cover
    """
    if state is None or state == "no solution":
        return []
    return [node for node, status in state.items() if status == 1]