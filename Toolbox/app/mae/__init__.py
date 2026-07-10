"""
Masked Autoencoder detection
==============================

This package provides tools for calculating the probability of crop marks
presence over each neochannel.
"""

from .utils import tilling
from .utils import compute_mae_batch
from .utils import define_model
from .utils import clip_mae

__all__ = [
    "tilling",
    "compute_mae_batch",
    "define_model",
    "clip_mae"
]
