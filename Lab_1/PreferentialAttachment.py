import random
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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


path_lengths = {}


def main():
    n = int(raw_input('Enter value of n: '))
    m0 = random.randint(2, n / 5)
    G = nx.path_graph(m0)
    # display_graph(G, '', '')
    G = add_nodes_barabasi(G, n, m0)
    # plot_deg_distribution(G)

    print('Edges: ', G.edges())

    for i in G.nodes():
        for j in G.nodes():
            if i != j:
                edge = str(i) + str(j)
                path = nx.dijkstra_path(G, i, j)
                pathL = path.__len__()-1
                # print('Shortest path from node ', i, ' to ', j, ' : ', path)
                if not pathL in path_lengths:
                    path_lengths[pathL] = 1
                else:
                    path_lengths[pathL] += 1
    print(path_lengths)
    s = OrderedDict(sorted(path_lengths.items(), key=lambda t: t[0]))
    names = list(s.keys())
    values = list(s.values())
    plt.bar(range(len(s)), values, tick_label=names)
    plt.xlabel('Path length')
    plt.ylabel('Frequency')
    plt.show()


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
            # print(r)
            k = 0
            while (not (r > prev_cum and r <= node_probs_cum[k][1])):
                prev_cum = node_probs_cum[k][1]
                k += 1

            target_node = node_probs_cum[k][0]
            if target_node in target_nodes:
                continue
            else:
                target_nodes.append(target_node)
            G.add_edge(i, target_node, weight=1)
            num_edges_added += 1
            new_edges.append((i, target_node))

        # print(num_edges_added, ' edges added...')
        # display_graph(G, i, new_edges)
    return G


def plot_deg_distribution(G):
    degrees = nx.degree(G)
    all_degrees = [x[1] for x in degrees]
    unique_degrees = list(set(all_degrees))
    unique_degrees.sort()
    count_of_degrees = []

    for i in unique_degrees:
        c = all_degrees.count(i)
        count_of_degrees.append(c)

    print(unique_degrees)
    print(count_of_degrees)

    plt.plot(unique_degrees, count_of_degrees, 'ro-')
    plt.xlabel('Degrees')
    plt.ylabel('Number of nodes')
    plt.title('Degree Distribution')
    plt.show()


main()
