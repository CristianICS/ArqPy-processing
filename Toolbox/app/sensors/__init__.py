"""
Sensor image handlers
==============================

This package provides the different functions to retrieve the required image
properties to correct them using the tools inside atmo_correction.
"""

from .world_view import WV3
from .world_view import LG06

# Single shared registry of supported sensors, keyed by the name used in
# every tool's "Sensor" dropdown. Replaces the hardcoded WV3/LEGION dicts
# that used to be duplicated independently in several GUI files.
SENSORS = {
    "WV3": WV3,
    "LEGION_06": LG06,
}

__all__ = [
    "WV3",
    "LG06",
    "SENSORS",
]

# Optional: package metadata
__version__ = "0.1.0"
__author__ = "Cristian Iranzo"
__email__ = "c.iranzo@unizar.es"
