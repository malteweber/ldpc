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
            girths.append(min_cycle_length(g, n))

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
            girths.append(min_cycle_length(g, n))

    girths = [g for g in girths if g != np.inf]

    girth_counts = Counter(girths)


    girth_avg = sum([log(k) * v for k, v in girth_counts.items()]) / sum([log(g) for g in girth_counts.values()])

    return girth_avg


def min_cycle_length(tg, source_node) -> int:
    """
    Calculate the length of the shortest cycle in a Tanner graph tg starting from source_node.
    If there is not cycle, return np.inf.
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
            return np.inf

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
