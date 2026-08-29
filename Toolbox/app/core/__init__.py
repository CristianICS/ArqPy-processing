"""
Shared infrastructure used across the toolbox's processing packages.

This package holds domain-agnostic helpers (raster I/O, tiling, vector
clipping, progress reporting, environment configuration) with no
PySimpleGUI or tool-specific business logic, so every processing package
(``atmo_correction``, ``pca``, etc.) can depend on it without depending on
each other.
"""
