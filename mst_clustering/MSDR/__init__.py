import networkx as nx
import numpy as np
from sklearn.metrics import pairwise_distances
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn import datasets


def f(X):
    def total_sd(T):
        sigma = 0
        for c in nx.connected_components(T):
            t = T.subgraph(c)
            sigma += sd(t) * t.number_of_nodes()
        return sigma/n

    def sd(T):
        l = list(nx.get_edge_attributes(T, "weight").values())
        return np.std(l) if len(l) > 0 else 0

    T = _generate_full_mst(X)
    eps = 1e-4
    n = T.number_of_nodes()
    deltas = [0]
    cur_sigma = total_sd(T)
    total_sigmas = [cur_sigma]
    while True:
        sigmas = []
        for e in T.edges:
            temp = nx.restricted_view(T, [], [e])
            sigmas.append(total_sd(temp))
        ax = np.argmin(sigmas)
        delta = cur_sigma - sigmas[ax]
        deltas.append(delta)
        cur_sigma -= delta
        total_sigmas.append(cur_sigma)
        edge_to_remove = list(T.edges)[ax]
        T.remove_edge(edge_to_remove[0], edge_to_remove[1])
        if abs(deltas[-1]-deltas[-2]) < eps*abs(deltas[-1]+1):
            break

    plt.scatter(np.arange(len(deltas)), deltas)
    plt.show()






def _generate_full_mst(X):
    n = X.shape[0]
    G = nx.complete_graph(n)

    distance_matrix = pairwise_distances(X)
    weights = {c: distance_matrix[c[0], c[1]] for c in combinations(range(n), 2)}
    # It works nicely because 'combinations' generates increasing sequences only

    nx.set_edge_attributes(G, weights, "weight")

    T = nx.Graph(G.edge_subgraph(nx.minimum_spanning_edges(G, data=False)))
    return T

def lattice_graph():
    x = np.arange(10)
    x[4:] += 1
    x = np.tile(x, 10)
    x = x + 0.3*np.random.rand(len(x))
    y = np.arange(10)
    y[4:] += 1
    y = np.repeat(y, 10)
    y = y + 0.3*np.random.rand(len(y))

    X = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
    return X

if __name__ == '__main__':
    X = lattice_graph()
    f(X)