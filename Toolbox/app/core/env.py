"""One-shot process configuration: GDAL exception mode and OTB executable
resolution.

Both used to be set/resolved independently in several modules, which caused
two live bugs: GDAL's exception mode is process-global, so whichever module
happened to import last silently decided the behaviour for every other
module's GDAL calls too; and the OTB executable path used to be resolved as
a module-level constant at import time, raising a bare TypeError the moment
`CONDA_PREFIX` was unset, before the user did anything.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from osgeo import gdal


def configure_gdal(use_exceptions: bool = True) -> None:
    """Set GDAL's process-global configuration once. Call this exactly
    once, from a launcher's entrypoint, before importing/using any GUI or
    domain module that touches GDAL.

    Covers what used to be set independently per module: the exception
    mode (previously `UseExceptions()`/`DontUseExceptions()` calls that
    could silently override each other depending on import order) and the
    multi-threading hints `atmo_correction.streaming.correct_raster_streaming`
    used to re-set on every single call."""
    if use_exceptions:
        gdal.UseExceptions()
    else:
        gdal.DontUseExceptions()

    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
    gdal.SetConfigOption("GDAL_TIFF_OVR_BLOCKSIZE", "128")


class OTBNotFoundError(RuntimeError):
    """Raised lazily, only when OTB is actually needed, never at import time."""


@lru_cache(maxsize=1)
def find_otb_executable(otb_folder: str = "OTB-9.1.1-Win64") -> Path:
    """Resolve the OTB command-line launcher under the active conda
    environment's OTB installation. Cached after the first successful
    lookup, since it can't change within a single process's lifetime."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        raise OTBNotFoundError(
            "CONDA_PREFIX is not set; cannot locate the OTB installation. "
            "Run this tool from within its packaged environment."
        )
    otb_exe = Path(conda_prefix, otb_folder, "bin", "otbApplicationLauncherCommandLine.exe")
    if not otb_exe.exists():
        raise OTBNotFoundError(f"OTB executable not found at expected path: {otb_exe}")
    return otb_exe
