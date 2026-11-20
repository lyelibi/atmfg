# Changelog

## 1.0.0 (2026-11-20)

First public release. The figures in arXiv:2603.09564 (v1) were produced with an
earlier, unreleased implementation; this version differs from it as follows.

### Fixes that change results
- **Candidate truncation.** Each face stores at most `(clique_size - 1) * k` candidates.
  The old code kept the first ones by node ID instead of the best ones by score.
  On GMRF data (N=5000, k=10) this raises true-edge recovery from 0.69 to 0.78 Jaccard.
- **Stale heap entries.** When the face universe was pruned, a deleted face's heap entry
  could later be applied to a new face reusing its slot, attaching a node with the old
  face's score. Heap entries are now tagged with the face's creation time. Affects runs
  where pruning occurs (e.g. N=5000, k=50, universe 1000: 0.794 -> 0.812 Jaccard).
- **Global rescue.** Rescued candidates were attached to the wrong faces (slots were
  reassigned on re-insertion), and rescue scores were on a different scale from local
  scores. Candidates are now updated in place, on the same scale.
- **Parent face deletion.** A newly created face could be deleted when pruning reused
  its parent's slot.
- **Score.** The default score is the TMFG gain (sum of correlations with the face's
  nodes), as described in the paper. The previous code used the correlation with the
  face's standardized centroid, which ranks faces differently; available as
  `score="centroid"`.
- **Preprocessing** is done in float32 without scikit-learn; on near-duplicate data
  this can flip near-ties relative to earlier versions.

### Robustness
- Raises `ValueError` on NaN/inf input or invalid parameters, and `RuntimeError` if the
  graph cannot be completed (previously returned an incomplete graph silently).
- Works for N <= k (k is capped at N - 1) and very small N.

### New
- `atmfg()` returns an `ATMFGResult` with edges, edge weights, cliques and separators.
- `clique_size` (TMFG convention, default 4): any size >= 3, faces of `clique_size - 1` nodes.
- Exposed HNSW and rescue settings (`hnsw_M`, `ef_construction`, `ef`, `rescue_k`, `n_jobs`, `seed`).
- `universe_size` defaults to 25% of the faces created over the run instead of a fixed 2500.
- `standardize_features` switch.

### Internals
- The Cython engine is replaced by numba (`@njit(cache=True)`): no build step for this package.
- About a third less working memory (single float32 copy of the data; no cached face vectors).
