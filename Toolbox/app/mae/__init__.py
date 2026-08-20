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
from .utils import parse_mae_bands
from .utils import validate_source_bands
from .utils import raised_hann_window
from .stats import is_inside_stats
from .stats import compute_raster_stats
from .stats import percentile_rank
from .stats import update_csv

__all__ = [
    "tilling",
    "compute_mae_batch",
    "define_model",
    "clip_mae",
    "parse_mae_bands",
    "validate_source_bands",
    "raised_hann_window",
    "is_inside_stats",
    "compute_raster_stats",
    "percentile_rank",
    "update_csv"
]
