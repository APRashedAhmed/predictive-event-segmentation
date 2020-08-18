"""Script that has the walking algorithms in the schapiro et al 2013 task"""
import logging

import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

@nx.utils.py_random_state(1)
def weighted_sample(node, seed=None):
    """Slightly modified version of ``nx.utils.weighted_choice`` to handle views

    Parameters
    ----------
    node : AtlasView
        A view of a node, containing all its neighbors
        
    Returns
    -------
    step : tuple
        Identifier for the sampled node
    """
    # use roulette method
    rnd = seed.random() * sum([n['weight'] for n in node.values()])
    for k, w in node.items():
        rnd -= w['weight']
        if rnd < 0:
            return k

def walk_random(G, source=None, steps=None):
    """Iterator that yields the start and final position after each step

    Parameters
    ----------
    G : nx.Graph
    	Graph to be walked through

    source : hashable (optional)
    	Hashable type that identifies the start node to use in G. Randomly
    	chooses a node if None

    steps : int or None (optional)
    	Number of steps to take before terminating. Yields indefinitely if None

    Yields
    ------
    (source, position) : (hashable, hashable)
    	The start and end node identifiers after each step
    """
    # Choose a start position
    source = source or list(G.nodes)[np.random.choice(len(G.nodes))]
    assert source in list(G.nodes)

    # Keep taking steps indefinitely or until steps reaches zero
    while steps != 0:
        position = weighted_sample(G[source])
        yield source, position
        source = position
        if isinstance(steps, (int, float)):
            steps -= 1

def walk_euclid(G, source=None):
    """Iterator that yields the start and final position after each step on a
    euclidean walk (visit every edge once).

    Parameters
    ----------
    G : nx.Graph
    	Graph to be walked through

    source : hashable (optional)
    	Hashable type that identifies the start node to use in G. Randomly
    	chooses a node if None

    Yields
    ------
    (source, position) : (hashable, hashable)
    	The start and end node identifiers after each step
    """    
    source = source or list(G.nodes)[np.random.choice(len(G.nodes))]
    assert source in list(G.nodes)
    
    return nx.algorithms.euler.eulerian_path(G, source=source)

def walk_hamiltonian(G, source=None):
    """Iterator that yields a start and final position after each step on a
    hamiltonian walk (visit every node once).

    Discovering hamiltonian paths is a NP-complete problem so beware using this
    on larger graphs as this is a brute force search implementation.

    Searches through all paths to find the first hamiltonian path it encounters.
    Once it finds it, the path is reversed 50% of the time, and rotated to start
    at ``source`` if passed.

    Parameters
    ----------
    G : nx.Graph
    	Graph to be walked through

    source : hashable (optional)
    	Hashable type that identifies the start node to use in G. Randomly
    	chooses a node if None

    Yields
    ------
    (source, position) : (hashable, hashable)
    	The start and end node identifiers after each step
    """
    F = [(G, [list(G.nodes)[np.random.choice(len(G.nodes))]])]
    n = G.number_of_nodes()
    while F:
        graph, path = F.pop()
        confs = []
        neighbors = (node for node in graph.neighbors(path[-1]) 
                     if node != path[-1]) #exclude self loops
        for neighbor in neighbors:
            conf_p = path[:]
            conf_p.append(neighbor)
            conf_g = nx.Graph(graph)
            conf_g.remove_node(path[-1])
            confs.append((conf_g, conf_p))
            
        for g, p in confs:
            if len(p) == n:
                # Reverse the list 50% of the time
                if np.random.choice(2):
                    p.reverse()
                
                # If a source was specified, rotate the list until source is
                # the start elem
                if source:
                    start = p[0]
                    while start != source:
                        p.append(p.pop(0))
                        start = p[0]
                
                # Create shifted list
                p_shift = list(p)
                p_shift.append(p_shift.pop(0))
                return iter(zip(p, p_shift))
            else:
                F.append((g,p))
    return None
