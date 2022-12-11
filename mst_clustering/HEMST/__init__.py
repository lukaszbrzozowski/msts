from msts.mst_clustering.utils.EMST import generate_full_emst
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class HEMST:
    class RecHEMST:
        def __init__(self, K: int,
                     data,
                     depth=0):
            self.desired_k = K
            self.n_c = 1
            self.emst = None
            self.data = data
            self.labels = None
            self.weights = None
            self.data_type = None
            self.depth = depth

        def _pre_fit(self):
            data = self.data
            self.emst, self.weights = generate_full_emst(data)
            self.d = data.shape[1]
            self.data_type = type(data[0, 0])
            weights_values = list(self.weights.values())
            average_weight = np.average(weights_values)
            weights_std = np.std(weights_values)
            for e in self.emst.edges():
                if self.weights[e] > average_weight + weights_std:
                    self.n_c += 1
                    self.emst.remove_edge(e[0], e[1])
                    del self.weights[e]
            self.labels = np.empty((self.data.shape[0], 2), dtype=int)
            for i, component in enumerate(nx.connected_components(self.emst)):
                nodes = np.array(list(component))
                print(nodes)
                for n in nodes:
                    self.labels[n, 0] = n
                    self.labels[n, 1] = i

        def _remove_longest_edges(self):
            difference = self.desired_k - self.n_c
            longest_edges = sorted(self.weights.keys(), key=lambda x: self.weights[x], reverse=True)
            self.emst.remove_edges_from(longest_edges[:difference])
            self.n_c += difference
            for i, component in enumerate(nx.connected_components(self.emst)):
                nodes = np.array(list(component))
                for n in nodes:
                    self.labels[n, 1] = i
            assert self.n_c == self.desired_k
            assert self.n_c == nx.number_connected_components(self.emst)

        def _map_to_representants(self):
            while self.n_c > self.desired_k:
                components = [[self.labels[i, 0] for i in range(self.labels.shape[0]) if self.labels[i, 1] == j] for j
                              in np.unique(self.labels[:, 1])]
                current_mapping = {}

                S = np.empty(shape=(len(components), self.d), dtype=self.data_type)
                for i, c in enumerate(components):
                    nodes = c

                    current_data = self.data[nodes, :]
                    current_centroid = np.mean(current_data, axis=0)
                    centroid_id = np.argmin(np.linalg.norm(current_centroid - current_data, axis=1))

                    centroid = current_data[centroid_id, :]
                    S[i, :] = centroid
                    for n in nodes:
                        current_mapping[n] = i
                nh = HEMST.RecHEMST(K=self.desired_k,
                               data=S, depth=self.depth + 1)
                nh.fit()
                new_labels = nh.get_labels()
                self.n_c = len(np.unique(new_labels[:, 1]))
                for i in range(self.labels.shape[0]):
                    self.labels[i, 1] = new_labels[current_mapping[i], 1]

        def fit(self):
            while self.n_c != self.desired_k:
                print("Prefit", self.n_c)
                self._pre_fit()
                print(self.labels)
                print("postprefit", self.n_c)
                if self.n_c < self.desired_k:
                    print("preremove", self.n_c)
                    self._remove_longest_edges()
                    print("postremove", self.n_c)
                if self.n_c > self.desired_k:
                    print("premap", self.n_c)
                    self._map_to_representants()
                    print("postmap", self.n_c)

        def get_labels(self):
            return self.labels

    def __init__(self, K: int):
        self.K = K

    def fit_transform(self, data):
        nh = self.RecHEMST(self.K, data)
        nh.fit()
        return nh.get_labels()

if __name__ == '__main__':
    data1 = np.random.multivariate_normal(np.array([0, 0]), np.diag(np.array([10, 10])), size=10)
    data2 = np.random.multivariate_normal(np.array([30, 30]), np.diag(np.array([10, 10])), size=10)
    data = np.vstack((data1, data2))
    plt.scatter(list(data[:, 0]), list(data[:, 1]))
    for i in range(20):
        plt.annotate(str(i), (data[i, 0], data[i, 1]))
    plt.show()
    hemst = HEMST(K=2)
    labs = hemst.fit_transform(data)
    print(np.unique(labs[:, 1]))
    print(labs)