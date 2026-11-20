"""Build an a-TMFG on synthetic data and inspect the result."""
import numpy as np

from atmfg import atmfg

rng = np.random.default_rng(0)
N, D = 10_000, 200

# Rows are the nodes; similarity is the correlation between rows.
# Here: 50 latent groups, each row a noisy copy of its group's signal.
groups = rng.integers(50, size=N)
signal = rng.standard_normal((50, D))
X = signal[groups] + 2.0 * rng.standard_normal((N, D))

res = atmfg(X, k=50, verbose=True)

print(f"{len(res.edges)} edges (3N-6 = {3 * N - 6})")
print(f"{len(res.cliques)} cliques of size {res.cliques.shape[1]}, {res.n_rescues} global rescues")
same = groups[res.edges[:, 0]] == groups[res.edges[:, 1]]
print(f"edges within a group: {same.mean():.1%}; mean edge correlation {res.weights.mean():.3f}")

A = res.adjacency()               # scipy.sparse.csr_matrix with correlation weights
# G = res.to_networkx()           # needs networkx
