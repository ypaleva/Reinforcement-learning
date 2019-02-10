import sys
import numpy as np

# graph = {0: {1: 1, 4: 1},
#          1: {2, 1},
#          2: {3: 1, 4: 1},
#          3: {0: 1},
#          4: {3: 1}}

graph = {0: {1: 2, 4: 4},
         1: {2: 3},
         2: {3: 5, 4: 1},
         3: {0: 8},
         4: {3: 3}}

print(graph)


def allPairsShortestPath(g):

    dist = np.zeros((5, 5))
    pred = np.zeros((5, 5))

    for u in g:
        for v in g:
            dist[u][v] = sys.maxsize
            pred[u][v] = None

        dist[u][u] = 0
        pred[u][u] = None

        for v in g[u]:
            dist[u][v] = g[u][v]
            pred[u][v] = u

    for mid in g:
        for u in g:
            for v in g:
                newlen = dist[u][mid] + dist[mid][v]
                if newlen < dist[u][v]:
                    dist[u][v] = newlen
                    pred[u][v] = pred[mid][v]
    return (dist, pred)


def constructShortestPath(s, t, pred):
    path = [t]
    while t != s:
        tmp = pred[s][t]
        t = tmp.astype(int)
        if t is None:
            return None
        path.insert(0, t)
    return path


dist, pred = allPairsShortestPath(graph)
print(dist)
path03 = constructShortestPath(0, 3, pred)
print(path03)
