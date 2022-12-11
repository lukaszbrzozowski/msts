import networkx as nx
from scipy.spatial.distance import pdist, squareform

# def generate_full_emst(X):
#     n = X.shape[0]
#     G = nx.complete_graph(n)
#
#     distance_matrix = pairwise_distances(X)
#     weights = {c: distance_matrix[c[0], c[1]] for c in combinations(range(n), 2)}
#     # It works nicely because 'combinations' generates increasing sequences only
#
#     T = nx.Graph(G.edge_subgraph(nx.minimum_spanning_edges(G, data=False)))
#     new_weights = {k: weights[k] for k in T.edges()}
#     return T, new_weights

def generate_full_emst(data):
    n = data.shape[0]
    G = nx.complete_graph(n)
    pd = squareform(pdist(data))
    edge_weights = {(i, j): pd[i, j] for i in range(n) for j in range(i + 1, n)}
    nx.set_edge_attributes(G, edge_weights, name="weight")
    mst = nx.minimum_spanning_tree(G)
    new_weights = {k: edge_weights[k] for k in mst.edges()}
    return mst, new_weights