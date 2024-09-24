from collections import Counter
from math import log

import networkx as nx
import numpy as np


def girth_average(g: nx.Graph) -> float:
    """
    Calculate the girth average of a Tanner graph g
    """
    girths = []
    for n in g.nodes():
        if n.endswith("1"):
            l = min_cycle_length(g, n)
            if l: girths.append(l)


    girth_counts = Counter(girths)

    girth_avg = sum([k * v for k, v in girth_counts.items()]) / sum(girth_counts.values())

    return girth_avg

def girth_average_log(g: nx.Graph) -> float:
    """
    Calculate the adapted girth average of a Tanner graph g
    """
    girths = []

    for n in g.nodes():
        if n.endswith("1"):
            l = min_cycle_length(g, n)
            if l: girths.append(l)

    girth_counts = {k:v for k,v in Counter(girths).items()}

    girth_avg = sum([log(k) * v for k, v in girth_counts.items()]) / sum([log(g) for g in girth_counts.values()])

    return girth_avg

def girth_median(g: nx.Graph) -> float:
    """
    Calculate the mean girth of a Tanner graph g
    """
    girths = []

    for n in g.nodes():
        if n.endswith("1"):
            l = min_cycle_length(g, n)
            if l: girths.append(l)


    return np.median(girths).astype(float)

def girth_min_node_fraction(g: nx.Graph) -> float:
    """
    Calculate the fraction of nodes that are in the smallest cycle in a Tanner graph g
    """
    girths = []

    for n in g.nodes():
        if n.endswith("1"):
            l = min_cycle_length(g, n)
            if l: girths.append(l)

    girth_counts = {k:v for k,v in Counter(girths).items()}

    return girth_counts[min(girth_counts.keys())] / sum(girth_counts.values())


def min_cycle_length(tg, source_node) -> int | None:
    """
    Calculate the length of the shortest cycle in a Tanner graph tg starting from source_node.
    If there is no cycle, return None.
    """
    leaves = [(None, source_node)]

    all_nodes = {source_node}
    k = 0
    while True:
        new_leaves = []
        for parent, n in leaves:

            for nn in tg.neighbors(n):
                if nn != parent:
                    if nn in all_nodes:
                        return 2 * k
                    else:
                        all_nodes.add(nn)
                        new_leaves.append((n, nn))
        k += 1
        if leaves == new_leaves:
            return None

        leaves = new_leaves


def create_tanner_graph(H: np.ndarray) -> nx.Graph:
    """
    Create a Tanner graph from a parity check matrix H
    """
    tg = nx.Graph()
    for i in range(H.shape[0]):
        tg.add_node(f"{i}_0")

    for i in range(H.shape[1]):
        tg.add_node(f"{i}_1")

    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if H[i, j] == 1:
                tg.add_edge(f"{i}_0", f"{j}_1")

    return tg
