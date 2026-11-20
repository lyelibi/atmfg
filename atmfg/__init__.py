"""
a-TMFG: Approximate Triangulated Maximally Filtered Graph.

    from atmfg import atmfg
    res = atmfg(X, k=50)            # rows of X are the nodes
    res.edges, res.weights, res.cliques, res.separators

Reference: arXiv:2603.09564.
"""
from .core import ATMFGResult, atmfg

__version__ = "1.0.0"
__all__ = ["atmfg", "ATMFGResult", "__version__"]
