"""
Sensor image handlers
==============================

This package provides the different functions to retrieve the required image
properties to correct them using the tools inside atmo_correction.
"""

from .world_view import WV3
from .world_view import LG06

__all__ = [
    "WV3",
    "LG06"
]

# Optional: package metadata
__version__ = "0.1.0"
__author__ = "Cristian Iranzo"
__email__ = "c.iranzo@unizar.es"