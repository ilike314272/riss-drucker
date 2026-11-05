"""Ab initio interface and multi-fidelity correction (README 5.8, M5).

Existing codes are used through a file boundary (geometries in, energies out:
JSON point sets, see io.py). Multi-fidelity: E_high(x) = E_low(x) + delta(x)
with delta a Gaussian process fitted to few high-level points (predictive
variance required). Posterior samples of the surface are propagated to levels
through the library's solvers. Ab initio training data are a prior, never the
target the spectroscopic posterior must match (no double counting)."""
from . import gp, io, multifidelity, torsion

__all__ = ["gp", "io", "multifidelity", "torsion"]
