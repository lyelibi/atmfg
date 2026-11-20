# a-TMFG

**Approximate Triangulated Maximally Filtered Graphs for large datasets.**

The [TMFG](https://doi.org/10.1093/comnet/cnw015) builds a sparse planar graph (3N − 6 edges) that keeps the strongest correlations between N variables. It needs the dense N × N correlation matrix, which limits it to tens of thousands of nodes. a-TMFG builds the same kind of graph from an approximate k-nearest-neighbour graph (HNSW), without ever forming the dense matrix. Memory and runtime grow close to linearly in N.

Paper: L. Yelibia, *a-TMFG: Scalable Triangulated Maximally Filtered Graphs via Approximate Nearest Neighbors*, [arXiv:2603.09564](https://arxiv.org/abs/2603.09564).

## Install

```bash
pip install git+https://github.com/lyelibi/atmfg
```

Requires Python 3.10+, with NumPy, SciPy, numba and hnswlib. `hnswlib` is distributed as source, so pip compiles it: this works out of the box on Linux and macOS. On Windows, install the [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first, or get a prebuilt hnswlib with `conda install -c conda-forge hnswlib`.

Optional: `tqdm` for a progress bar, `networkx` for `res.to_networkx()`.

## Quickstart

```python
import numpy as np
from atmfg import atmfg

X = np.random.default_rng(0).standard_normal((50_000, 100))  # rows are the nodes
res = atmfg(X, k=50)

res.edges        # (3N-6, 2) int64, u < v, in insertion order
res.weights      # (3N-6,)  correlation of each edge's endpoints
res.cliques      # (N-3, 4) the seed clique, then (new node, face) per insertion
res.separators   # (N-4, 3) the face each node was inserted into
res.adjacency()  # scipy.sparse.csr_matrix with correlation weights
```

Similarity is the Pearson correlation between rows. See [`examples/quickstart.py`](examples/quickstart.py).

The first call takes a few seconds longer while numba compiles the engine. The result is cached, and later runs start in about a second.

## Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `k` | 50 | Neighbours per node in the kNN graph that drives the local search. Larger is more accurate and slower; capped at N − 1. |
| `clique_size` | 4 | Size of the cliques; faces have `clique_size − 1` nodes. 4 gives the TMFG (planar). Other sizes give the same construction on larger or smaller faces: chordal, not planar, with (c − 1)·N − c(c − 1)/2 edges. |
| `universe_size` | `None` | Maximum number of active faces kept; when full, the oldest half is pruned. `None` means 25% of all faces created, about N/2 for the TMFG. |
| `score` | `"sum"` | `"sum"`: the TMFG gain, the sum of the node's correlations with the face's nodes. `"centroid"`: correlation with the face's standardized centroid. |
| `standardize_features` | `True` | Z-score each column before standardizing the rows. |
| `hnsw_M`, `ef_construction`, `ef` | 64, 200, `None` | HNSW graph degree and beam widths (`ef=None` means max(100, k + 1)). Index building is most of the runtime; lower values are faster and slightly less accurate. |
| `rescue_k` | 5 | Candidates fetched per face during a global rescue. |
| `n_jobs` | −1 | Threads used by hnswlib. Use 1 for bit-for-bit reproducible results. |
| `seed` | `None` | Seed for hnswlib's level generator. |
| `verbose` | `False` | Progress bar and rescue messages. |

## How it works

1. **Preprocessing.** Rows are standardized, so the cosine between rows equals their correlation. An HNSW index gives a symmetrized kNN graph.
2. **Seed.** The construction starts from a clique: the endpoint of the strongest kNN edge and its three strongest neighbours.
3. **Local expansion.** Each face's candidates are the not-yet-added kNN neighbours of its nodes, scored by the TMFG gain. A lazy max-heap repeatedly inserts the best (face, node) pair, which splits the face into three new ones.
4. **Bounded universe.** At most `universe_size` faces are kept; when full, the oldest half is pruned.
5. **Global rescue.** When no face has a candidate left (a disconnected kNN graph, exhausted neighbourhoods, or pruned faces), every active face queries the HNSW index for the nearest nodes not yet in the graph. Nodes already added are marked deleted in the index.

With `k = N − 1` and no pruning, a-TMFG makes exactly the same choices as the exact greedy construction. The test suite checks this for clique sizes 3–6.

## Changes since the arXiv version

The figures in arXiv:2603.09564 (v1) were produced with an earlier implementation. Version 1.0.0 fixes bugs that changed results, most of them improving accuracy, and replaces the Cython engine with numba. See [CHANGELOG.md](CHANGELOG.md).

## Tests and benchmarks

```bash
pip install -e ".[dev]"
pytest                               # ~30 s
python benchmarks/run_benchmark.py   # accuracy vs. ground truth and exact TMFG, several minutes
```

## Citation

```bibtex
@article{yelibia2026atmfg,
  title   = {a-TMFG: Scalable Triangulated Maximally Filtered Graphs via Approximate Nearest Neighbors},
  author  = {Yelibia, Lionel},
  journal = {arXiv preprint arXiv:2603.09564},
  year    = {2026}
}
```

## License

MIT. See [LICENSE](LICENSE).
