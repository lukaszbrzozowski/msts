#!/usr/bin/env -S python3 -W ignore::FutureWarning


"""
Apply a range of clustering algorithms on the benchmark datasets
described in: Gagolewski M., A framework for benchmarking clustering algorithms,
SoftwareX 20, 2022, 101270, https://clustering-benchmarks.gagolewski.com,
DOI: 10.1016/j.softx.2022.101270.


Copyright (C) 2023, Marek Gagolewski <https://www.gagolewski.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import clustbench  # https://pypi.org/project/clustering-benchmarks/
import os.path
import numpy as np


## TODO: change me: -------------------------------------------------------

# see <https://github.com/gagolews/clustering-data-v1>:
data_path = os.path.join("..", "clustering-data-v1")

results_path_base = os.path.join(".", "results")

max_n = 1000
max_k = 16

skip_batteries = ["h2mg", "g2mg", "mnist"]

import sklearn.cluster
algorithms = {
    "KMeans": sklearn.cluster.KMeans(),
}


## ------------------------------------------------------------------------


batteries = clustbench.get_battery_names(path=data_path)
batteries = sorted(set(batteries) - set(skip_batteries))

for battery in batteries:
    results_path = os.path.join(results_path_base, battery)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    datasets = clustbench.get_dataset_names(battery, path=data_path)

    for dataset in datasets:
        b = clustbench.load_dataset(battery, dataset, path=data_path)

        print("%s/%s [n=%d, d=%d]: " % (
            b.battery, b.dataset, b.data.shape[0], b.data.shape[1]
        ), end="")

        if b.data.shape[0] > max_n:
            print("**skipping (n>max_n)**")
            continue

        ks = np.arange(2, max(max(b.n_clusters), max_k)+1)
        res = clustbench.load_results(
            results_path, ".", dataset, ks
        )
        res = clustbench.transpose_results(res)

        ks = list(sorted(set(ks) - set(res.keys())))
        if len(ks) == 0:
            print("**skipping (all done)**")
            continue

        print("k=%s... " % (", ".join([str(k) for k in ks])), end="")
        res = dict()
        for alg in algorithms:
            res[alg] = clustbench.fit_predict_many(
                algorithms[alg], b.data, ks
            )

        res = clustbench.transpose_results(res)
        for k in res:
            clustbench.save_results(
                os.path.join(results_path, "%s.result%d.gz" % (dataset, k)),
                res[k]
            )

        print("done.")
