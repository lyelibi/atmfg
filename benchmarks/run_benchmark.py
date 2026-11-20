"""
Accuracy and runtime on synthetic GMRF data with a known planar ground truth.

    python benchmarks/run_benchmark.py                      # defaults below
    python benchmarks/run_benchmark.py --N 20000 --D 500 --no-exact

Reports, per (k, universe_size), the Jaccard similarity of the a-TMFG edges to the
ground-truth graph and, unless --no-exact, to the exact TMFG (dense O(N^2); keep
N <= ~10,000). A second table stresses the global rescue on separated blobs.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from atmfg import atmfg                                                    # noqa: E402
from benchmarks.common import (blobs, edge_set, exact_tmfg, gmrf_data,     # noqa: E402
                               jaccard, random_apollonian, row_corr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=5000)
    p.add_argument("--D", type=int, default=1000, help="features per node")
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--no-exact", action="store_true", help="skip the dense exact TMFG")
    a = p.parse_args()

    grid = [(50, None), (50, 1000), (10, None), (10, 500), (3, 300)]
    print(f"GMRF: N={a.N}, D={a.D}, alpha={a.alpha}, {a.seeds} seeds")
    runs = []
    for s in range(a.seeds):
        rng = np.random.default_rng(s)
        A = random_apollonian(a.N, rng)
        Y = gmrf_data(A, a.N, a.D, a.alpha, rng)
        ref = None if a.no_exact else exact_tmfg(row_corr(Y))
        runs.append((Y, A, ref))
    if not a.no_exact:
        print(f"  exact TMFG vs truth: J = {np.mean([jaccard(ref, A) for _, A, ref in runs]):.3f}")
    print(f"  {'k':>3} {'universe':>9} {'J(truth)':>9} {'J(exact)':>9} {'time (s)':>9}")
    for k, U in grid:
        jt, je, tt = [], [], []
        for Y, A, ref in runs:
            t = time.perf_counter()
            res = atmfg(Y, k=k, universe_size=U)
            tt.append(time.perf_counter() - t)
            jt.append(jaccard(res.edges, A))
            if ref is not None:
                je.append(jaccard(res.edges, ref))
        je_s = f"{np.mean(je):9.3f}" if je else f"{'-':>9}"
        print(f"  {k:>3} {str(U if U else 'auto'):>9} {np.mean(jt):9.3f} {je_s} {np.mean(tt):9.2f}")

    print("\nGlobal rescue: 40 separated blobs, N=4000, D=50")
    print(f"  {'k':>3} {'universe':>9} {'rescues':>8} {'weight vs exact':>16}")
    for k, U in [(3, 2500), (5, 1000)]:
        wr, rc = [], []
        for s in range(a.seeds):
            Y = blobs(4000, 50, 40, np.random.default_rng(s))
            W = row_corr(Y)
            ref = exact_tmfg(W)
            res = atmfg(Y, k=k, universe_size=U)
            ws = lambda E: sum(W[u, v] for u, v in edge_set(E))
            wr.append(ws(res.edges) / ws(ref))
            rc.append(res.n_rescues)
        print(f"  {k:>3} {U:>9} {np.mean(rc):8.1f} {np.mean(wr):16.4f}")


if __name__ == "__main__":
    main()
