"""
Spectral Indices Package
==============================

This package provides tools for calculating all the spectral indices in batch.
"""

from .utils import MissingBandsError, check_bands, compute_index, get_index_keys
from .job import SpectralIndicesJob, SpectralIndicesResult

__all__ = [
    "MissingBandsError",
    "check_bands",
    "compute_index",
    "get_index_keys",
    "SpectralIndicesJob",
    "SpectralIndicesResult",
]
