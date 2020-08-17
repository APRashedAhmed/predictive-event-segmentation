"""Script for generating the graph used in Schapiro et al 2013."""
import logging

import networkx as nx

logger = logging.getLogger(__name__)

def schapiro_pentagon(label='A', weight=1):
    """Creates a pentagon of nodes and edges like in the paper.

    Nodes have a common label a unique number from 0 to 4. Nodes zero and four
    are the "border" nodes, and have one less edge than the rest.

    Parameters
    ----------
    label : str (optional)
    	Label to assign the collection nodes

    weight : int (optional)
    	Weight of edges.

    Returns
    -------
    G : nx.Graph
    	Graph that contains the nodes and edges for the pentagon
    """
    G = nx.Graph()
    G.add_nodes_from(zip([label]*5, range(5)))

    nodes_list = list(G.nodes)
    border_nodes = [nodes_list[0], nodes_list[-1]]
    for i, node_1 in enumerate(nodes_list[:-1]):
        for node_2 in nodes_list[i+1:]:
            if node_1 in border_nodes and node_2 in border_nodes:
                continue
            G.add_edge(node_1, node_2, weight=weight)
    return G

def schapiro_graph(labels='ABC', weight=1):
    """Creates the full schapiro graph that is a collection of pentagon nodes.

    Each community receives a unique label, and is connected to neighboring
    communities.

    Parameters
    ----------
    labels : str (optional)
    	Labels to assign each community

    weight : int (optional)
    	Weight of edges.

    Returns
    -------
    G : nx.Graph
    	Graph that contains the nodes and edges for the full schapiro graph
    """    
    G = nx.Graph()

    # Add each community to the full graph
    for label in labels:
        pentagon = graph_pentagon(label)
        G.add_nodes_from(pentagon)
        G.add_edges_from(pentagon.edges, weight=weight)

    # Connect each community together
    for i, label in enumerate(labels[:-1]):
        G.add_edge((label,4), (labels[i+1],0), weight=weight)
        G.add_edge((label,0), (labels[i-1],4), weight=weight)

    return G
