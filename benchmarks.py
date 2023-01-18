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
import pandas as pd
import sys
import gc


## TODO: change me: -------------------------------------------------------

# see <https://github.com/gagolews/clustering-data-v1>:
data_path = os.path.join("..", "..", "clustering-data-v1")

results_path_base = os.path.join(".", "results")

max_n = 9999
max_k = np.Inf
all_k = False

skip_batteries = ["h2mg", "g2mg", "mnist"]

import mst_clustering.HEMST
import mst_clustering.CTCEHC
algorithms = {
    "HEMST": mst_clustering.HEMST.HEMST,
    "CTCEHC": mst_clustering.CTCEHC.CTCEHC,
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
        ), end="", file=sys.stderr)

        if b.data.shape[0] > max_n:
            print("**skipping (n>max_n)**", file=sys.stderr)
            continue

        if all_k:
            ks = np.arange(2, max(max(b.n_clusters), max_k)+1)
        else:
            ks = np.unique(b.n_clusters)

        ks = ks[ks <= max_k]

        res = clustbench.load_results(
            results_path, ".", dataset, ks
        )
        res = clustbench.transpose_results(res)

        ks = list(sorted(set(ks) - set(res.keys())))
        if len(ks) == 0:
            print("**skipping (all done)**", file=sys.stderr)
            continue

        print("k=%s... " % (", ".join([str(k) for k in ks])),
              end="", file=sys.stderr)
        res = dict()
        for alg in algorithms:
            print("%s... " % alg, end="", file=sys.stderr)
            sys.stderr.flush()
            gc.collect()
            res[alg] = clustbench.fit_predict_many(
                algorithms[alg](), b.data, ks
            )

        res = clustbench.transpose_results(res)
        for k in res:
            try:
                x = pd.DataFrame(res[k])
                if not np.all(x.min().isin([0, 1])):
                    raise ValueError("Minimal label neither 0 nor 1.")

                mx = x.max()
                if not mx[0] >= 1:
                    raise ValueError("At least 1 cluster is necessary.")

                if not np.all(mx == mx[0]):
                    raise ValueError("All partitions should be of the same cardinality.")

                if not np.all(x.apply(np.bincount).iloc[1:, :] > 0):
                    raise ValueError("Denormalised label vector: Cluster IDs should be consecutive integers.")

                clustbench.save_results(
                    os.path.join(results_path, "%s.result%d.gz" % (dataset, k)),
                    res[k]
                )
            except Exception as e:
                print("**k=%d: %s** " % (k, str(e)), file=sys.stderr)

        print("done.", file=sys.stderr)

