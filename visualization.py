"""
visualization.py
generates plots (instances, solutions, etc)
"""

import matplotlib.pyplot as plt
import networkx as nx

def plot_graph_instance(graph, title="Graph instance"):
    """
   plots the graph instance
    """
    plt.figure(figsize=(10, 8))
    
 
    pos = nx.circular_layout(graph)
    
    nx.draw(graph, pos, 
            with_labels=True, 
            node_color='lightgray', 
            node_size=800,
            font_weight='bold', 
            edge_color='dimgray')
            
    plt.title(title, fontsize=14, fontweight='bold')
    
    plt.axis('off')
    plt.show()

def plot_mvc_solution(graph, mvc_nodes, title="solution to the MVC problem"):
    """
    plots the graph instance with the vertex cover
    """
    plt.figure(figsize=(10, 8))
    

    pos = nx.circular_layout(graph)
    
  
    nodes_in_cover = [n for n in graph.nodes() if n in mvc_nodes]
    nodes_outside = [n for n in graph.nodes() if n not in mvc_nodes]
    
    nx.draw_networkx_edges(graph, pos, edge_color='dimgray')
    nx.draw_networkx_nodes(graph, pos, nodelist=nodes_outside, 
                           node_color='lightgray', node_size=800, label="Not in the MVC")
                           
    
    nx.draw_networkx_nodes(graph, pos, nodelist=nodes_in_cover, 
                           node_color='tomato', node_size=900,
                           edgecolors='darkred', linewidths=2, label="In the MVC")
                           
    nx.draw_networkx_labels(graph, pos, font_weight='bold')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="upper right")
    plt.axis('off')
    plt.show()

