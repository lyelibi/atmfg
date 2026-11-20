"""
Synthetic data and exact references used by the tests and benchmarks.

- random_apollonian: a random maximal planar graph built TMFG-style (each node
  inserted into a uniformly random face), used as ground truth.
- gmrf_data: Y = (I + alpha/2 A) X with X ~ N(0, I); rows are nodes (paper, Sec. 4.1).
- exact_greedy: the exact greedy construction on a dense weight matrix, with
  (clique_size - 1)-node faces each used once; clique_size = 4 is the TMFG.
- exact_tmfg: exact TMFG with the seed of Massara et al. (2017).
"""
import numpy as np
import scipy.sparse as sp
from numba import njit


def random_apollonian(N, rng):
    """Edges (u, v) of a random maximal planar graph on N nodes, labels shuffled."""
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for v in range(4, N):
        j = rng.integers(len(faces))
        a, b, c = faces[j]
        faces[j] = (a, b, v)
        faces.append((a, c, v))
        faces.append((b, c, v))
        edges += [(a, v), (b, v), (c, v)]
    perm = rng.permutation(N)
    return np.array([(perm[u], perm[v]) for u, v in edges], dtype=np.int64)


def gmrf_data(edges, N, D, alpha, rng):
    """Y = (I + alpha/2 A) X with X ~ N(0, I_{N x D}); rows are nodes."""
    u, v = np.asarray(edges).T
    A = sp.coo_matrix((np.ones(len(u)), (u, v)), shape=(N, N)).tocsr()
    A = A + A.T
    X = rng.standard_normal((N, D))
    return (X + (alpha / 2.0) * (A @ X)).astype(np.float32)


def blobs(N, D, n_centers, rng, spread=50.0, std=1.0):
    """Well-separated Gaussian blobs: a kNN graph with many components."""
    centers = rng.uniform(-spread, spread, size=(n_centers, D))
    labels = rng.integers(n_centers, size=N)
    return (centers[labels] + std * rng.standard_normal((N, D))).astype(np.float32)


def row_corr(Y):
    """Dense Pearson correlation between rows."""
    Z = Y - Y.mean(1, keepdims=True)
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    W = (Z @ Z.T).astype(np.float64)
    np.fill_diagonal(W, 0.0)
    return W


def edge_set(edges):
    return {(min(int(u), int(v)), max(int(u), int(v))) for u, v in edges}


def jaccard(a, b):
    A, B = edge_set(a), edge_set(b)
    return len(A & B) / len(A | B)


@njit(cache=True)
def _best(W, face, in_graph):
    bs, bv = -1e18, -1
    for v in range(W.shape[0]):
        if not in_graph[v]:
            g = 0.0
            for j in range(face.shape[0]):
                g += W[v, face[j]]
            if g > bs:
                bs, bv = g, v
    return bs, bv


@njit(cache=True)
def exact_greedy(W, s, seed):
    """Exact greedy construction with s-node faces used once (s=3: TMFG), from a
    seed clique of s+1 nodes. O(N^2) time; for validation on small N."""
    N = W.shape[0]
    in_graph = np.zeros(N, dtype=np.bool_)
    nf = (s + 1) + (N - s - 1) * s
    faces = np.zeros((nf, s), dtype=np.int64)
    alive = np.zeros(nf, dtype=np.bool_)
    gain = np.full(nf, -1e18)
    best = np.full(nf, -1, dtype=np.int64)
    edges = np.zeros((s * N - s * (s + 1) // 2, 2), dtype=np.int64)
    ne = 0
    for a in range(s + 1):
        in_graph[seed[a]] = True
        for b in range(a + 1, s + 1):
            edges[ne, 0] = seed[a]
            edges[ne, 1] = seed[b]
            ne += 1
    F = 0
    for drop in range(s + 1):
        c = 0
        for j in range(s + 1):
            if j != drop:
                faces[F, c] = seed[j]
                c += 1
        alive[F] = True
        F += 1
    for f in range(F):
        gain[f], best[f] = _best(W, faces[f], in_graph)
    for _ in range(N - s - 1):
        t, tg = -1, -1e18
        for f in range(F):
            if alive[f] and gain[f] > tg:
                tg, t = gain[f], f
        v = best[t]
        in_graph[v] = True
        for j in range(s):
            edges[ne, 0] = v
            edges[ne, 1] = faces[t, j]
            ne += 1
        alive[t] = False
        for drop in range(s):
            c = 0
            for j in range(s):
                if j != drop:
                    faces[F, c] = faces[t, j]
                    c += 1
            faces[F, s - 1] = v
            alive[F] = True
            gain[F], best[F] = _best(W, faces[F], in_graph)
            F += 1
        for f in range(F):
            if alive[f] and best[f] == v:
                gain[f], best[f] = _best(W, faces[f], in_graph)
    return edges


def massara_seed(W):
    """Seed of Massara et al. (2017): the 4 nodes with the largest sum of weights
    above the mean weight."""
    N = W.shape[0]
    off = ~np.eye(N, dtype=bool)
    m = W[off].mean()
    s = np.where((W > m) & off, W, 0.0).sum(1)
    return np.argsort(-s, kind="stable")[:4].astype(np.int64)


def exact_tmfg(W):
    """Exact TMFG edges on a dense weight matrix (Massara et al. seed)."""
    return exact_greedy(np.ascontiguousarray(W, dtype=np.float64), 3, massara_seed(W))
