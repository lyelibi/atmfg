"""Fast tests for atmfg (well under a minute after numba's first compile)."""
from itertools import combinations

import networkx as nx
import numpy as np
import pytest

from atmfg import ATMFGResult, atmfg
from atmfg.knn import standardize
from benchmarks.common import blobs, exact_greedy, gmrf_data, jaccard, random_apollonian


def _gmrf(N, D, seed=0, alpha=0.3):
    rng = np.random.default_rng(seed)
    return gmrf_data(random_apollonian(N, rng), N, D, alpha, rng)


def _graph(res):
    G = nx.Graph()
    G.add_nodes_from(range(res.n_nodes))
    G.add_edges_from(map(tuple, res.edges))
    return G


def _check_structure(res, N, clique_size):
    s = clique_size - 1
    G = _graph(res)
    assert len(res.edges) == s * N - s * (s + 1) // 2 == G.number_of_edges()
    assert nx.is_connected(G)
    assert nx.is_chordal(G)
    assert max(len(c) for c in nx.chordal_graph_cliques(G)) == clique_size
    assert res.cliques.shape == (N - s, clique_size)
    assert res.separators.shape == (N - s - 1, s)
    assert set(np.unique(res.cliques)) == set(range(N))
    assert all(G.has_edge(int(a), int(b)) for c in res.cliques for a, b in combinations(c, 2))
    assert all(set(p) <= set(c) for p, c in zip(res.separators, res.cliques[1:]))
    if clique_size <= 4:
        assert nx.check_planarity(G)[0]


@pytest.mark.parametrize("clique_size", [3, 4, 5, 6])
def test_limit_matches_exact_greedy(clique_size):
    """With every neighbour and no pruning, a-TMFG is the exact greedy construction."""
    for seed in range(2):
        Y = _gmrf(150, 500, seed)
        N = len(Y)
        res = atmfg(Y, k=N - 1, clique_size=clique_size, universe_size=10 * N)
        Xs = standardize(Y).astype(np.float64)
        W = Xs @ Xs.T / Xs.shape[1]
        np.fill_diagonal(W, 0.0)
        ref = exact_greedy(W, clique_size - 1, res.cliques[0].astype(np.int64))
        assert jaccard(res.edges, ref) == 1.0


@pytest.mark.parametrize("clique_size", [3, 4, 5, 6])
@pytest.mark.parametrize("k,universe", [(20, None), (8, 60)])
def test_valid_structure(clique_size, k, universe):
    Y = _gmrf(800, 100)
    _check_structure(atmfg(Y, k=k, clique_size=clique_size, universe_size=universe), 800, clique_size)


def test_global_rescue_completes_disconnected_knn_graph():
    Y = blobs(1500, 20, 25, np.random.default_rng(0))
    res = atmfg(Y, k=3, universe_size=300)
    assert res.n_rescues > 0
    _check_structure(res, 1500, 4)


def test_result_object():
    Y = _gmrf(600, 100)
    res = atmfg(Y, k=20)
    assert isinstance(res, ATMFGResult)
    assert (res.edges[:, 0] < res.edges[:, 1]).all()
    Z = (Y - Y.mean(0)) / Y.std(0)
    C = np.corrcoef(Z)
    assert np.allclose(res.weights, C[res.edges[:, 0], res.edges[:, 1]], atol=1e-4)
    A = res.adjacency()
    assert A.shape == (600, 600) and A.nnz == 2 * len(res.edges)
    assert abs(A - A.T).nnz == 0
    assert res.to_networkx().number_of_edges() == len(res.edges)
    assert res.params["universe_size"] == int(0.25 * (2 * 600 - 4))


def test_single_thread_is_reproducible():
    Y = _gmrf(800, 100)
    a = atmfg(Y, k=15, universe_size=200, n_jobs=1)
    b = atmfg(Y, k=15, universe_size=200, n_jobs=1)
    assert np.array_equal(a.edges, b.edges) and np.array_equal(a.cliques, b.cliques)


@pytest.mark.parametrize("score", ["sum", "centroid"])
def test_options(score):
    Y = _gmrf(500, 50).astype(np.float64)
    for kw in ({}, {"standardize_features": False}, {"hnsw_M": 16, "ef_construction": 50, "ef": 60}):
        _check_structure(atmfg(Y, k=10, score=score, **kw), 500, 4)


@pytest.mark.parametrize("N,clique_size", [(3, 3), (4, 4), (5, 4), (6, 5), (12, 4)])
def test_tiny_inputs(N, clique_size):
    Y = np.random.default_rng(1).standard_normal((N, 6))
    _check_structure(atmfg(Y, k=50, clique_size=clique_size), N, clique_size)


def test_duplicate_and_constant_rows():
    rng = np.random.default_rng(3)
    Y = np.repeat(rng.standard_normal((60, 10)), 5, axis=0)
    Y[:3] = 1.0                       # constant rows
    _check_structure(atmfg(Y, k=10), 300, 4)


@pytest.mark.parametrize("X,kw", [
    (np.where(np.arange(200).reshape(20, 10) == 7, np.nan, 1.0), {}),
    (np.ones((10, 5)), {"clique_size": 2}),
    (np.ones((10, 5)), {"k": 2}),
    (np.ones((3, 5)), {}),
    (np.ones(10), {}),
    (np.ones((10, 5)), {"score": "max"}),
    (np.ones((10, 5)), {"universe_size": 3}),
])
def test_invalid_inputs(X, kw):
    with pytest.raises(ValueError):
        atmfg(X, **kw)
