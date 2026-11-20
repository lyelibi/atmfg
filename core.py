"""a-TMFG driver: compiled local expansion with a global HNSW rescue."""
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import scipy.sparse

from . import engine as E
from .knn import build_knn_graph, query_alive, standardize, unit_rows

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:          # tqdm is optional
    _tqdm = None

# Node insertions per call into the compiled loop between progress updates.
_CHUNK = 20_000
# Rebuild a fresh HNSW index over the remaining nodes once this fraction is in the graph.
_FRESH_INDEX_RATIO = 0.95


@dataclass
class ATMFGResult:
    """
    Output of :func:`atmfg`.

    Attributes
    ----------
    edges : ndarray (E, 2) int64
        Edges as (u, v) with u < v, in insertion order. E = s*N - s(s+1)/2 with
        s = clique_size - 1 (3N - 6 for the TMFG).
    weights : ndarray (E,) float32
        Correlation of each edge's endpoints (after preprocessing).
    cliques : ndarray (N - s, clique_size) int64
        The seed clique, then one clique per inserted node: (new node, face nodes).
    separators : ndarray (N - s - 1, s) int64
        The face each node was inserted into; separators[i] joins cliques[i + 1]
        to the graph built so far.
    n_rescues : int
        Number of global HNSW rescues triggered.
    """
    edges: np.ndarray
    weights: np.ndarray
    cliques: np.ndarray
    separators: np.ndarray
    n_rescues: int
    n_nodes: int
    params: dict = field(default_factory=dict)

    def adjacency(self):
        """Weighted symmetric adjacency matrix (scipy.sparse.csr_matrix)."""
        u, v = self.edges[:, 0], self.edges[:, 1]
        w = self.weights
        A = scipy.sparse.coo_matrix((np.r_[w, w], (np.r_[u, v], np.r_[v, u])),
                                    shape=(self.n_nodes, self.n_nodes))
        return A.tocsr()

    def to_networkx(self):
        """networkx.Graph with a 'weight' attribute on each edge."""
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(self.n_nodes))
        G.add_weighted_edges_from((int(u), int(v), float(w))
                                  for (u, v), w in zip(self.edges, self.weights))
        return G


def _total_faces(N, s):
    """Faces created over a full run: s+1 seed faces, then s-1 net per insertion."""
    return (s + 1) + (N - s - 1) * (s - 1)


def atmfg(X, k=50, clique_size=4, universe_size=None, score="sum",
          standardize_features=True, hnsw_M=64, ef_construction=200, ef=None,
          rescue_k=5, n_jobs=-1, seed=None, verbose=False):
    """
    Approximate TMFG of the rows of X.

    Parameters
    ----------
    X : array-like (N, D)
        Rows are the nodes; similarity is the Pearson correlation between rows.
    k : int
        Neighbours per node in the HNSW kNN graph that drives the local search
        (capped at N - 1).
    clique_size : int
        Size of the cliques; faces have clique_size - 1 nodes. 4 gives the TMFG
        (planar, 3N - 6 edges). Other sizes give the same construction on
        (clique_size - 1)-node faces: chordal, not planar.
    universe_size : int or None
        Maximum number of active faces kept at once; when full, the oldest half is
        pruned. None: 25% of all faces created over the run (about N/2 for the TMFG).
    score : {"sum", "centroid"}
        "sum": TMFG gain, the sum of the node's correlations with the face nodes.
        "centroid": correlation with the face's standardized centroid.
    standardize_features : bool
        Z-score each column before standardizing rows.
    hnsw_M, ef_construction, ef : int
        HNSW graph degree, build-time and query-time beam widths. ef None:
        max(100, k + 1). Index building dominates runtime; lower values are faster
        and slightly less accurate.
    rescue_k : int
        Nodes requested per face from HNSW during a global rescue.
    n_jobs : int
        Threads for hnswlib (-1: all cores). Use 1 for bit-for-bit reproducibility.
    seed : int or None
        Seed for hnswlib's level generator (None: hnswlib's default, 100).
    verbose : bool
        Progress bar (if tqdm is installed) and rescue messages.

    Returns
    -------
    ATMFGResult

    Raises
    ------
    ValueError    invalid input or parameters
    RuntimeError  the graph could not be completed
    """
    # --- validation ---
    if score not in ("sum", "centroid"):
        raise ValueError("score must be 'sum' or 'centroid'.")
    clique_size = int(clique_size)
    if clique_size < 3:
        raise ValueError("clique_size must be at least 3.")
    s = clique_size - 1
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array (N nodes x D features).")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or infinite values.")
    N, D = X.shape
    if N < clique_size:
        raise ValueError(f"Need at least clique_size={clique_size} rows.")
    k = int(min(k, N - 1))
    if k < s:
        raise ValueError(f"k must be at least clique_size - 1 = {s}.")
    if universe_size is None:
        universe_size = max(int(0.25 * _total_faces(N, s)), 2 * clique_size)
    U = int(universe_size)
    if U < clique_size:
        raise ValueError(f"universe_size must be at least clique_size={clique_size}.")
    mode = E.SCORE_SUM if score == "sum" else E.SCORE_CENTROID
    shift = float(s)
    params = dict(k=k, clique_size=clique_size, universe_size=U, score=score,
                  standardize_features=standardize_features, hnsw_M=hnsw_M,
                  ef_construction=ef_construction, ef=ef, rescue_k=rescue_k,
                  n_jobs=n_jobs, seed=seed)
    log = print if verbose else (lambda *a, **kw: None)

    # --- preprocessing and kNN graph ---
    Xs = standardize(X, standardize_features)
    Xu = unit_rows(Xs)
    log(f"Building HNSW kNN graph (N={N}, D={D}, k={k})...")
    graph, index = build_knn_graph(Xu, k, M=hnsw_M, ef_construction=ef_construction,
                                   ef=ef, n_jobs=n_jobs, seed=seed)
    indptr = graph.indptr.astype(np.int64)
    indices = graph.indices.astype(np.int32)

    # --- engine state ---
    K = s * k
    faces = np.zeros((U, s), dtype=np.int32)
    scores_pool = np.zeros(U * K, dtype=np.float32)
    nbs_pool = np.zeros(U * K, dtype=np.int32)
    counts = np.zeros(U, dtype=np.int32)
    active = np.zeros(U, dtype=np.uint8)
    birth = np.zeros(U, dtype=np.int64)
    free_stack = np.arange(U, dtype=np.int32)
    is_dead = np.zeros(N, dtype=np.uint8)
    cap = 4 * N + 2 * U
    heap = [np.empty(cap, np.float32), np.empty(cap, np.int32),
            np.empty(cap, np.int32), np.empty(cap, np.int64)]
    meta = np.zeros(E.META_SIZE, dtype=np.int64)
    meta[E.STACK] = U
    n_edges = s * N - s * (s + 1) // 2
    edges = np.empty((n_edges, 2), dtype=np.int32)
    added = np.empty(N, dtype=np.int32)
    cliques = np.empty((N - s, clique_size), dtype=np.int32)
    separators = np.empty((max(N - s - 1, 0), s), dtype=np.int32)

    def grow_heap(min_size):
        while heap[0].shape[0] < min_size:
            for i in range(4):
                heap[i] = np.concatenate([heap[i], np.empty_like(heap[i])])

    # --- seed: strongest kNN edge's endpoint + its s strongest neighbours ---
    anchor = int(graph.indices[np.argmax(graph.data)])
    row = graph[anchor]
    nbs_sorted = row.indices[np.argsort(row.data)[::-1]]
    if len(nbs_sorted) < s:
        raise ValueError("kNN graph too sparse to form the seed clique; increase k.")
    seed_nodes = sorted([anchor] + [int(x) for x in nbs_sorted[:s]])
    for v in seed_nodes:
        is_dead[v] = 1
        index.mark_deleted(v)
    for i, (u, v) in enumerate(combinations(seed_nodes, 2)):
        edges[i] = (u, v)
    meta[E.NEDGE] = clique_size * s // 2
    cliques[0] = seed_nodes
    meta[E.NCLIQUE] = 1
    E.add_children(-1, np.array(list(combinations(seed_nodes, s)), dtype=np.int32),
                   Xs, indptr, indices, mode, shift, faces, scores_pool, nbs_pool,
                   counts, active, birth, free_stack, is_dead, *heap, meta)

    # --- main loop: compiled local expansion, Python global rescue ---
    marked, n_rescues, just_rescued = 0, 0, False
    bar = _tqdm(total=n_edges, initial=int(meta[E.NEDGE]), desc="a-TMFG", unit="edge",
                dynamic_ncols=True) if (verbose and _tqdm is not None) else None
    try:
        while True:
            before = int(meta[E.NEDGE])
            status, steps = E.expand(Xs, indptr, indices, mode, shift, _CHUNK, faces,
                                     scores_pool, nbs_pool, counts, active, birth,
                                     free_stack, is_dead, *heap, meta, edges, added,
                                     cliques, separators)
            if bar is not None:
                bar.update(int(meta[E.NEDGE]) - before)
            if steps > 0:
                just_rescued = False
            if status == E.STATUS_DONE:
                break
            if status == E.STATUS_CHUNK:
                continue
            if status == E.STATUS_GROW:
                grow_heap(2 * heap[0].shape[0])
                continue

            # STATUS_RESCUE: local search exhausted
            if just_rescued:
                break
            n_added = int(meta[E.NADDED])
            for v in added[marked:n_added]:
                index.mark_deleted(int(v))
            marked = n_added
            slots = np.flatnonzero(active).astype(np.int64)
            if len(slots) == 0:
                break
            q = E.face_vectors(slots, faces, Xs, mode)
            qnorm = np.linalg.norm(q, axis=1, keepdims=True)
            q /= np.maximum(qnorm, 1e-12)
            lab, dist, fresh = query_alive(index, Xu, is_dead, q, rescue_k, n_jobs,
                                           fresh=is_dead.sum() / N > _FRESH_INDEX_RATIO)
            # same scale as the local scores: cos = 1 - d^2/2 on unit vectors;
            # sum mode: f . x / D = cos * ||f|| / sqrt(D)
            cos = 1.0 - dist / 2.0
            sc = shift + (cos * (qnorm / np.sqrt(D)) if mode == E.SCORE_SUM else cos)
            grow_heap(int(meta[E.HSIZE]) + len(slots) + 8)
            E.rescue_reset(slots, np.ascontiguousarray(sc, dtype=np.float32),
                           np.ascontiguousarray(lab, dtype=np.int32), scores_pool,
                           nbs_pool, counts, active, birth, is_dead, *heap, meta)
            n_rescues += 1
            msg = (f"  global rescue {n_rescues} at {int(meta[E.NEDGE])} edges: "
                   f"{len(slots)} faces queried{' (fresh index)' if fresh else ''}")
            bar.write(msg) if bar is not None else log(msg)
            just_rescued = True
    finally:
        if bar is not None:
            bar.close()

    if int(meta[E.NEDGE]) != n_edges:
        raise RuntimeError(
            f"a-TMFG stopped at {int(meta[E.NEDGE])} of {n_edges} edges: the global "
            "rescue found no attachable node. Try a larger universe_size or k.")

    weights = E.edge_weights(edges, Xs)
    out_edges = np.sort(edges, axis=1).astype(np.int64)
    return ATMFGResult(edges=out_edges, weights=weights, cliques=cliques.astype(np.int64),
                       separators=separators.astype(np.int64), n_rescues=n_rescues,
                       n_nodes=N, params=params)
