"""Machine-learning scaffolding (README 2, 5.6, 5.7, 7): spectral fingerprints,
synthetic training data, and amortized inference that is always verified by
the certified physics fit ("ML proposes, physics verifies")."""
from . import amortized, dataset, moments

__all__ = ["amortized", "dataset", "moments"]
