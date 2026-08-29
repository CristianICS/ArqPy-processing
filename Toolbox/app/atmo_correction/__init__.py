"""
Atmospheric Correction Package
==============================

This package provides tools for performing atmospheric correction on
remote-sensing imagery. Modules include atmospheric correction formulas and
common utility functions.
"""

from .formulas import FORMULAS
from .streaming import OutputSpec
from .streaming import correct_raster_streaming
from .utils import SensorParams
from .job import AtmoCorrectionJob, AtmoCorrectionResult

__all__ = [
    "FORMULAS",
    "OutputSpec",
    "correct_raster_streaming",
    "SensorParams",
    "AtmoCorrectionJob",
    "AtmoCorrectionResult",
]

# Optional: package metadata
__version__ = "0.1.0"
__author__ = "Cristian Iranzo"
__email__ = "c.iranzo@unizar.es"
