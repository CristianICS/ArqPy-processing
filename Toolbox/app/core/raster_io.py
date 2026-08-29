"""Shared GDAL raster creation-options and dtype-mapping helpers.

Every processing package used to hand-roll its own "TILED=YES,
COMPRESS=..." creation-options list and its own numpy<->GDAL dtype
mapping. This module is the single source of truth for both.

`osgeo.gdal` is imported lazily, inside the two functions that actually
call it (`numpy_dtype_to_gdal`, `create_gtiff_like`), rather than at
module level: tools with no GDAL in their conda env (e.g. `sam3_cropmarks`,
which only uses rasterio) still need the GDAL-independent helpers here
(`RasterProfile`, `gdal_creation_options`, `rasterio_profile_kwargs`, ...)
without failing to import this module at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np


@dataclass(frozen=True)
class RasterProfile:
    """GDAL GTiff creation-option settings shared by the processing tools."""
    compress: str = "DEFLATE"
    predictor: Optional[int] = 2
    tiled: bool = True
    blockxsize: Optional[int] = 512
    blockysize: Optional[int] = 512
    bigtiff: Optional[str] = "IF_SAFER"


def gdal_creation_options(profile: RasterProfile = RasterProfile()) -> list[str]:
    """Render a RasterProfile as a GDAL GTiff driver `options=[...]` list."""
    options = []
    if profile.compress:
        options.append(f"COMPRESS={profile.compress}")
    if profile.tiled:
        options.append("TILED=YES")
    if profile.blockxsize:
        options.append(f"BLOCKXSIZE={profile.blockxsize}")
    if profile.blockysize:
        options.append(f"BLOCKYSIZE={profile.blockysize}")
    if profile.bigtiff:
        options.append(f"BIGTIFF={profile.bigtiff}")
    if profile.predictor is not None:
        options.append(f"PREDICTOR={profile.predictor}")
    return options


def cog_creation_options(profile: RasterProfile = RasterProfile()) -> list[str]:
    """Render the subset of a RasterProfile accepted by GDAL's COG driver
    (COG has its own internal tiling/overview scheme, so TILED/BLOCKXSIZE/
    BLOCKYSIZE from `profile` are not passed through)."""
    options = []
    if profile.compress:
        options.append(f"COMPRESS={profile.compress}")
    if profile.predictor is not None:
        options.append(f"PREDICTOR={profile.predictor}")
    if profile.bigtiff:
        options.append(f"BIGTIFF={profile.bigtiff}")
    return options


def rasterio_profile_kwargs(profile: RasterProfile = RasterProfile()) -> dict:
    """Render a RasterProfile as rasterio profile-update kwargs for a tiled
    GeoTIFF (rasterio's counterpart to `gdal_creation_options`)."""
    kwargs: dict = {"driver": "GTiff"}
    if profile.tiled:
        kwargs["tiled"] = True
    if profile.blockxsize:
        kwargs["blockxsize"] = profile.blockxsize
    if profile.blockysize:
        kwargs["blockysize"] = profile.blockysize
    if profile.compress:
        kwargs["compress"] = profile.compress
    if profile.predictor is not None:
        kwargs["predictor"] = profile.predictor
    if profile.bigtiff:
        kwargs["bigtiff"] = profile.bigtiff
    return kwargs


def otb_creation_option_string(
    profile: RasterProfile = RasterProfile(),
    *, copy_src_overviews: bool = False, num_threads: bool = False,
) -> str:
    """Render a RasterProfile as an OTB `?&gdal:co:...` output-URI query
    string, for OTB CLI apps' `-out path?...` argument. `copy_src_overviews`
    and `num_threads` add the two GDAL creation options OTB pansharpening
    pipelines conventionally set that have no `RasterProfile` equivalent."""
    parts = [f"gdal:co:{opt}" for opt in gdal_creation_options(profile)]
    if copy_src_overviews:
        parts.append("gdal:co:COPY_SRC_OVERVIEWS=YES")
    if num_threads:
        parts.append("gdal:co:NUM_THREADS=ALL_CPUS")
    return "&" + "&".join(parts)


def numpy_dtype_to_gdal(dtype) -> int:
    """Map a numpy dtype to its GDAL type constant, defaulting to Float32."""
    from osgeo import gdal

    numpy_to_gdal = {
        np.dtype("uint8"): gdal.GDT_Byte,
        np.dtype("int16"): gdal.GDT_Int16,
        np.dtype("uint16"): gdal.GDT_UInt16,
        np.dtype("int32"): gdal.GDT_Int32,
        np.dtype("uint32"): gdal.GDT_UInt32,
        np.dtype("float32"): gdal.GDT_Float32,
        np.dtype("float64"): gdal.GDT_Float64,
    }
    return numpy_to_gdal.get(np.dtype(dtype), gdal.GDT_Float32)


def dtype_range(dtype) -> tuple[float, float]:
    """Return the (min, max) representable range for a numpy dtype."""
    dt = np.dtype(dtype)
    info = np.iinfo(dt) if np.issubdtype(dt, np.integer) else np.finfo(dt)
    return float(info.min), float(info.max)


def create_gtiff_like(
    out_path: Union[str, Path],
    xsize: int,
    ysize: int,
    band_count: int,
    gdal_dtype: int,
    *,
    geotransform=None,
    projection: Optional[str] = None,
    metadata: Optional[dict] = None,
    profile: RasterProfile = RasterProfile(),
) -> gdal.Dataset:
    """Create a new GTiff dataset with the given geo-referencing/metadata
    and the creation options from `profile`. Raises RuntimeError if GDAL
    fails to create it (e.g. an invalid output path)."""
    from osgeo import gdal

    driver = gdal.GetDriverByName("GTiff")
    dst = driver.Create(
        str(out_path), xsize, ysize, band_count, gdal_dtype,
        options=gdal_creation_options(profile),
    )
    if dst is None:
        raise RuntimeError(f"Could not create output raster: {out_path}")

    if geotransform is not None:
        dst.SetGeoTransform(geotransform)
    if projection:
        dst.SetProjection(projection)
    if metadata:
        dst.SetMetadata(metadata)
    return dst


def translate_to_cog(src_path: Union[str, Path], dst_path: Union[str, Path],
                      profile: RasterProfile = RasterProfile(),
                      *, nodata: Optional[float] = None) -> None:
    """Translate an existing raster to a Cloud Optimized GeoTIFF via
    `gdal_translate`. Raises CalledProcessError on failure."""
    import subprocess
    co_flags = []
    for opt in cog_creation_options(profile):
        co_flags.extend(["-co", opt])
    nodata_flags = ["-a_nodata", str(nodata)] if nodata is not None else []
    subprocess.run([
        "gdal_translate",
        str(src_path),
        str(dst_path),
        *nodata_flags,
        "-of", "COG",
        *co_flags,
    ], check=True)
