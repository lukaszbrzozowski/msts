from sklearn.metrics import pairwise_distances
from itertools import combinations
import networkx as nx


def generate_full_emst(X):
    n = X.shape[0]
    G = nx.complete_graph(n)

    distance_matrix = pairwise_distances(X)
    weights = {c: distance_matrix[c[0], c[1]] for c in combinations(range(n), 2)}
    # It works nicely because 'combinations' generates increasing sequences only

    T = nx.Graph(G.edge_subgraph(nx.minimum_spanning_edges(G, data=False)))
    new_weights = {k: weights[k] for k in T.edges()}
    return T, new_weights

