from mst_clustering.utils.EMST import generate_full_emst
import numpy as np
import networkx as nx

class HEMST:

    def __init__(self, k: int):
        """
        Constructor of the HEMST method.

        :param k: number of clusters
        """
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")

        self.k = k
        self.n_c = 1
        self.emst = None
        self.mappings = None
        self.d = None
        self.data_type = None
        self.data = None


    def _pre_fit(self,
                 data: np.ndarray):
        self.data = data
        self.emst, self.weights = generate_full_emst(data)
        self.d = data.shape[1]
        self.data_type = type(data[0, :])
        weights_values = list(self.weights.values())
        average_weight = np.average(weights_values)
        weights_std = np.std(weights_values)
        for e in self.emst.edges():
            if self.weights[e] > average_weight + weights_std:
                self.n_c += 1
                self.emst.remove_edge(e[0], e[1])

    def _remove_longest_edges(self):
        difference = self.n_c - self.k
        longest_edges = sorted(self.weights.keys(), key=lambda x: self.weights[x], reverse=True)
        self.emst.remove_edges_from(longest_edges[:difference])
        self.n_c += difference

    def _map_vertices(self):
        connected_components = nx.connected_components(self.emst)
        current_mapping = {}

        S = np.empty(shape=(nx.number_connected_components(self.emst), self.d), dtype=self.data_type)
        for i, c in enumerate(connected_components):
            nodes = np.array(list(c))
            current_data = self.data[nodes, :]
            current_centroid = np.mean(current_data, axis=0)
            centroid_id = np.argmin(np.linalg.norm(current_centroid - current_data, axis=1))
            centroid = current_data[centroid_id, :]
            starting_ix = nodes[centroid_id]
            S[i, :] = centroid
            for n in nodes:
                current_mapping[n] = starting_ix
        print(current_mapping)
        print(S)
    def fit(self,
            data: np.ndarray):
        S = data
        self._pre_fit(S)

        while self.n_c != self.k:

            if self.n_c < self.k:
                self._remove_longest_edges()

            if self.n_c > self.k:
                self._map_vertices()
            break
        # return weights

if __name__ == '__main__':
    data = np.random.normal(size=(20, 5))
    hemst = HEMST(k=3)
    hemst.fit(data)