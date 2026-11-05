"""Machine-learning scaffolding (README 2, 5.6, 5.7, 7): spectral fingerprints,
synthetic training data, amortized inference (Bayesian-linear and neural ensembles)
and amortized structure proposals - always verified by the certified physics fit
("ML proposes, physics verifies")."""
from . import amortized, dataset, moments, neural, structure

__all__ = ["amortized", "dataset", "moments", "neural", "structure"]
