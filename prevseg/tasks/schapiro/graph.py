"""Script for generating the graph used in Schapiro et al 2013."""
import logging

import networkx as nx

logger = logging.getLogger(__name__)

def schapiro_pentagon(offset=0, weight=1):
    """Creates a pentagon of nodes and edges like in the paper.

    Nodes are numbered from ``offset*5`` to ``offset*5 + 5``.
    
    Parameters
    ----------
    offset : int (optional)
    	Offset to start the numbering of the nodes

    weight : int (optional)
    	Weight of edges.

    Returns
    -------
    G : nx.Graph
    	Graph that contains the nodes and edges for the pentagon
    """
    G = nx.Graph()
    G.add_nodes_from(range(offset, offset+5))

    nodes_list = list(G.nodes)
    border_nodes = [nodes_list[0], nodes_list[-1]]
    for i, node_1 in enumerate(nodes_list[:-1]):
        for node_2 in nodes_list[i+1:]:
            if node_1 in border_nodes and node_2 in border_nodes:
                continue
            G.add_edge(node_1, node_2, weight=weight)
    return G

def schapiro_graph(n_pentagons=3, weight=1):
    """Creates the full schapiro graph that is a collection of pentagon nodes.

    Nodes are numberd in groups of five from ``0`` to ``n_pentagons * 5``.

    Parameters
    ----------
    n_pentagons : int (optional)
    	Number of pentagons to include in the graph

    weight : int (optional)
    	Weight of edges.

    Returns
    -------
    G : nx.Graph
    	Graph that contains the nodes and edges for the full schapiro graph
    """    
    G = nx.Graph()

    # Add each community to the full graph
    pents = [schapiro_pentagon(offset=i*5) for i in range(n_pentagons)]
    [G.add_edges_from(pent.edges, weight=weight) for pent in pents]

    # Connect each community together
    for i in range(n_pentagons-1):
        G.add_edge(list(pents[i])[0], list(pents[i-1])[-1], weight=weight)
        G.add_edge(list(pents[i])[4], list(pents[i+1])[0], weight=weight)

    return G
