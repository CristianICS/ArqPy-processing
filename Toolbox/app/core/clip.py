"""Shared "clip raster to a vector cutline" helper.

Also home to the fix for a bug repeated across several GUI files: the
clip-vector field being left blank must silently mean "skip clipping",
never trigger a "vector folder does not exist" popup.

`osgeo.gdal` is imported lazily, inside `clip_raster_to_vector` only:
`resolve_clip_vector` is pure path logic, and `gui.framework` imports this
module unconditionally (for `resolve_clip_vector`) even for tools like
`sam3_cropmarks` whose conda env has no GDAL at all.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from .raster_io import RasterProfile, cog_creation_options, gdal_creation_options


def resolve_clip_vector(raw_value: str) -> Optional[Path]:
    """Interpret a GUI clip-vector field value.

    A blank value means "skip clipping" and returns None without raising
    or warning. A non-blank value that does not point to an existing file
    raises FileNotFoundError, so the caller aborts instead of silently
    continuing with a bad path.
    """
    if not raw_value:
        return None
    path = Path(raw_value)
    if not path.exists():
        raise FileNotFoundError(f"Clip vector not found: {path}")
    return path


def clip_raster_to_vector(
    src_path: Union[str, Path],
    vector_path: Union[str, Path],
    dst_path: Union[str, Path],
    *,
    nodata: Optional[float] = None,
    profile: RasterProfile = RasterProfile(),
    as_cog: bool = True,
) -> Path:
    """Warp `src_path` against `vector_path`'s cutline, writing straight to
    `dst_path` (as a COG when `as_cog=True`, or a plain tiled GeoTIFF
    otherwise). Raises RuntimeError on failure."""
    from osgeo import gdal

    creation_options = (
        cog_creation_options(profile) if as_cog else gdal_creation_options(profile)
    )
    warp_kwargs = dict(
        cutlineDSName=str(vector_path),
        cropToCutline=True,
        multithread=True,
        creationOptions=creation_options,
        warpOptions=["NUM_THREADS=ALL_CPUS"],
    )
    if nodata is not None:
        warp_kwargs["dstNodata"] = nodata
    if as_cog:
        warp_kwargs["format"] = "COG"

    result = gdal.Warp(str(dst_path), str(src_path), **warp_kwargs)
    if result is None:
        raise RuntimeError(f"Failed to clip raster to vector cutline: {src_path}")
    result = None
    return Path(dst_path)
