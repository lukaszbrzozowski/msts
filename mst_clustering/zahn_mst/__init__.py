import networkx as nx
import numpy as np
from sklearn.metrics import pairwise_distances
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn import datasets


def zahn_mst(X,
              d=3,
              sigma_T=3.0,
              f_T=2.0,
             verbose=True):
    """
    :param X: data sample, ndarray of shape (n_samples_X, n_features)
    :param d: local subtree depth, integer >= 1
    :param sigma_T: standard deviation threshold
    :param f_T: edge length threshold
    :return:
    """

    T = _generate_full_mst(X)

    # Removing outliers
    _generate_node_attributes(T, d)
    _remove_inconsistent_edges(T, 3, 2)

    _generate_node_attributes(T, d)
    _remove_inconsistent_edges(T, sigma_T, f_T)

    if verbose:
        nc = len(list(nx.connected_components(T)))
        ie = nc - 1
        print(f"The algorithm found {nc} cluster(s) by removing {ie} inconsistent edge(s).")


    # Uncomment for matrix output
    # output = np.array([np.array([int(e[0]), int(e[1]), T.edges[e[0], e[1]]["weight"]]) for e in T.edges])
    # return output

    return T


def _generate_full_mst(X):
    n = X.shape[0]
    G = nx.complete_graph(n)

    distance_matrix = pairwise_distances(X)
    weights = {c: distance_matrix[c[0], c[1]] for c in combinations(range(n), 2)}
    # It works nicely because 'combinations' generates increasing sequences only

    nx.set_edge_attributes(G, weights, "weight")

    T = nx.Graph(G.edge_subgraph(nx.minimum_spanning_edges(G, data=False)))
    return T


def _generate_node_attributes(G, d):

    avgs = [0] * G.number_of_nodes()
    sds = [0] * G.number_of_nodes()

    for v in G.nodes:
        weights = list(nx.get_edge_attributes(G.edge_subgraph(nx.dfs_edges(G, v, d)), "weight").values())
        avgs[v] = np.average(weights)
        sds[v] = np.std(weights)

    avg_attributes, sd_attributes = dict(zip(G.nodes, avgs)), dict(zip(G.nodes, sds))

    nx.set_node_attributes(G, avg_attributes, "alew")  # Average Local Edge Weight
    nx.set_node_attributes(G, sd_attributes, "sd")


def _remove_inconsistent_edges(G, sigma, f):

    def __is_inconsistent(e):

        ew = G.edges[e[0], e[1]]["weight"]
        n1_alew, n1_sd = G.nodes[e[0]]["alew"], G.nodes[e[0]]["sd"]
        n2_alew, n2_sd = G.nodes[e[1]]["alew"], G.nodes[e[1]]["sd"]

        return ew > sigma * n1_sd + n1_alew and ew > sigma * n2_sd + n2_alew and ew > f * n1_alew and ew > f * n2_alew

    inconsistent_edges = [e for e in G.edges if __is_inconsistent(e)]
    G.remove_edges_from(inconsistent_edges)


def iris():
    iris = datasets.load_iris().data[:, np.array([3, 0])]
    return iris

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

def main():
    # np.random.seed(0)
    X = lattice_graph()
    # # X = np.random.rand(400, 5)
    #
    # T = zahn_mst(X, d=3, sigma_T=2, f_T=1.3)
    # plt.scatter(X[:, 0], X[:, 1])
    #
    # nx.draw(T, pos={v: X[v, :] for v in T.nodes})
    # plt.show()
    # X = iris()
    T = zahn_mst(X, d=4, sigma_T=1.6, f_T=1.3)
    nx.draw(T, pos={v: X[v, :] for v in T.nodes})
    plt.show()
    # print(T)


if __name__ == '__main__':
    main()
