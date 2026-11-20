"""Preprocessing, HNSW kNN graph and global-rescue queries."""
import hnswlib
import numpy as np
import scipy.sparse

_ROW_CHUNK = 8192


def standardize(data, standardize_features=True):
    """
    Returns float32 X whose rows have mean 0 and variance 1 over features, so that
    x_u . x_v / D is the Pearson correlation of rows u and v.

    standardize_features=True first z-scores each column across rows (as in the
    paper's experiments). Constant columns/rows are left at 0 (scale 1).
    Arithmetic is float64 per chunk of rows; only one float32 copy is kept.
    """
    data = np.asarray(data)
    N, D = data.shape
    if standardize_features:
        mu = data.mean(axis=0, dtype=np.float64)
        sd = data.std(axis=0, dtype=np.float64)
        sd[sd == 0.0] = 1.0
    X = np.empty((N, D), dtype=np.float32)
    for start in range(0, N, _ROW_CHUNK):
        Z = np.asarray(data[start:start + _ROW_CHUNK], dtype=np.float64)
        if standardize_features:
            Z = (Z - mu) / sd
        m = Z.mean(axis=1, keepdims=True)
        s = Z.std(axis=1, keepdims=True)
        s[s == 0.0] = 1.0
        X[start:start + _ROW_CHUNK] = (Z - m) / s
    return X


def unit_rows(X):
    """Rows scaled to unit L2 norm (zero rows stay zero)."""
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def build_knn_graph(X_unit, k, M=64, ef_construction=200, ef=None, n_jobs=-1, seed=None):
    """
    Symmetrized kNN graph from an HNSW index over unit-norm rows.

    For unit vectors the squared L2 distance d^2 and the cosine satisfy
    cos = 1 - d^2 / 2; with standardized rows cosine = correlation.

    Returns
    -------
    graph : scipy.sparse.csr_matrix   cosine weights, max over the two directions
    index : hnswlib.Index             kept for the global rescue
    """
    N, D = X_unit.shape
    index = hnswlib.Index(space='l2', dim=D)
    index.init_index(max_elements=N, ef_construction=ef_construction, M=M,
                     random_seed=100 if seed is None else int(seed))
    index.add_items(X_unit, num_threads=n_jobs)
    index.set_ef(max(100, k + 1) if ef is None else max(int(ef), k + 1))

    # k+1 because a node is normally returned as its own first neighbour
    nbrs, dist = index.knn_query(X_unit, k=k + 1, num_threads=n_jobs)
    sims = (1.0 - dist / 2.0).ravel()
    rows = np.repeat(np.arange(N), k + 1)
    cols = nbrs.ravel()
    keep = rows != cols

    # +1 shift so zero/negative cosines survive eliminate_zeros(); only the ORDER of
    # the weights is used (seed choice, symmetrization).
    g = scipy.sparse.coo_matrix((sims[keep] + 1.0, (rows[keep], cols[keep])),
                                shape=(N, N), dtype=np.float32).tocsr()
    g = g.maximum(g.transpose()).tocsr()
    g.setdiag(0)
    g.eliminate_zeros()
    g.data -= 1.0
    return g, index


def query_alive(index, X_unit, is_dead, queries, k, n_jobs=-1, fresh=False):
    """
    The k nearest nodes not yet in the graph, for each (unit-norm) query.

    Uses the main index (nodes in the graph are marked deleted), or builds a fresh
    index over the remaining nodes when fresh=True or the main index cannot return
    k results. Returns (labels, squared distances, used_fresh_index).
    """
    alive = np.flatnonzero(is_dead == 0)
    k = min(k, len(alive))
    if not fresh:
        try:
            lab, dist = index.knn_query(queries, k=k, num_threads=n_jobs)
            return lab, dist, False
        except RuntimeError:
            pass
    tmp = hnswlib.Index(space='l2', dim=X_unit.shape[1])
    tmp.init_index(max_elements=max(len(alive), 50), ef_construction=50, M=16)
    tmp.add_items(X_unit[alive], alive, num_threads=n_jobs)   # global IDs as labels
    tmp.set_ef(max(50, k))
    lab, dist = tmp.knn_query(queries, k=k, num_threads=n_jobs)
    return lab, dist, True
