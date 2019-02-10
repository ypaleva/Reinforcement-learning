import random

import networkx as nx
import matplotlib.pyplot as plt
from past.builtins import raw_input


def display_graph(G, i, ne):
    pos = nx.circular_layout(G)
    if i == '' and ne == '':
        new_node = []
        rest_nodes = G.nodes()
        new_edges = []
        rest_edges = G.edges()
    else:
        new_node = [i]
        rest_nodes = list(set(G.nodes()) - set(new_node))
        new_edges = ne
        rest_edges = list(set(G.edges()) - set(new_edges) - set([(b, a) for (a, b) in new_edges]))
    nx.draw_networkx_nodes(G, pos, nodelist=new_node, node_color='g')
    nx.draw_networkx_nodes(G, pos, nodelist=rest_nodes, node_color='r')
    nx.draw_networkx_edges(G, pos, edgelist=new_edges, edge_color='g', style='dashdot')
    nx.draw_networkx_edges(G, pos, edgelist=rest_edges, edge_color='r')
    plt.show()


def main():
    n = int(raw_input('Enter value of n: '))
    m0 = random.randint(2, n / 5)
    G = nx.path_graph(m0)
    display_graph(G, '', '')
    G = add_nodes_barabasi(G, n, m0)


def add_nodes_barabasi(G, n, m0):
    m = m0 - 1
    for i in range(m0 + 1, n + 1):
        G.add_node(i)
        degrees = nx.degree(G)
        node_probs = {}

        for each in G.nodes():
            node_probs[each] = (float)(degrees[each] / sum([x[1] for x in degrees]))

        node_probs_cum = []
        prev = 0
        for n, p in node_probs.items():
            temp = [n, prev + p]
            node_probs_cum.append(temp)
            prev = prev + p

        new_edges = []
        num_edges_added = 0
        target_nodes = []

        while (num_edges_added < m):
            prev_cum = 0
            r = random.random()
            k = 0
            while (not (r > prev_cum and r <= node_probs_cum[k][1])):
                prev_cum = node_probs_cum[k][1]
                k += 1

            target_node = node_probs_cum[k][0]
            if target_node in target_nodes:
                continue
            else:
                target_nodes.append(target_node)
            G.add_edge(i, target_node)
            num_edges_added += 1
            new_edges.append((i, target_node))

        print(num_edges_added, ' edges added...')
        display_graph(G, i, new_edges)
    return G


main()