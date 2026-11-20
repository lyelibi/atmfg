import os
import sys

# Make `benchmarks` (synthetic data + exact references) importable from the tests.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
