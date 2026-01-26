from dataclasses import dataclass
from dataclasses import replace
from typing import Callable, Optional, Sequence, Tuple, Union
from osgeo import gdal

import subprocess
import numpy as np

try:
    import numexpr as ne
    _USE_NUMEXPR = True
except Exception:
    _USE_NUMEXPR = False

from .utils import write_log
from .formulas import FORMULAS
from atmo_correction.utils import BandParams, CtxParams, SensorParams

@dataclass
class OutputSpec:
    """
    Output configuration for the corrected raster.

    Attributes
    ----------
    dtype : str
        Output dtype, e.g., "uint16" or "float32".
    nodata : Union[int, float]
        Output nodata value (use np.nan for float32 if desired).
    scale : Optional[float]
        If provided, multiply results by this factor (e.g., 10000 for scaled reflectance).
    clamp : Optional[Tuple[float, float]]
        (min, max) clipping range. If None, inferred from dtype if integer.
    compress : str
        GDAL compression (e.g., "ZSTD", "DEFLATE", "LZW").
    predictor : Optional[int]
        GDAL predictor (2 recommended for continuous data).
    tiled : bool
        Write tiled GeoTIFF.
    blocksize : Optional[int]
        Override block size (square) and update the creation options.
        If None, keep source blocks where possible.
    bigtiff : Optional[str]
        BIGTIFF creation option; e.g., "IF_SAFER".
    cog : Optional[bool]
        If create a Cloud Optimized Geotiff
    """
    dtype: str = "uint16"
    nodata: Union[int, float] = 0
    scale: Optional[float] = 10000.0
    clamp: Optional[Tuple[float, float]] = None
    compress: str = "DEFLATE"
    predictor: Optional[int] = 2
    tiled: bool = False
    blocksize: Optional[int] = None
    bigtiff: Optional[str] = "IF_SAFER"
    cog: Optional[bool] = True


BandFormula = Union[
    str,                                        # name in FORMULAS
    Callable[[np.ndarray, dict, dict], np.ndarray]
]

def _dtype_range(dtype: str) -> Tuple[float, float]:
    """Return (min, max) representable range for dtype."""
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.integer):
        info = np.iinfo(dt)
    else:
        info = np.finfo(dt)
    return float(info.min), float(info.max)

def _numpy_dtype_to_gdal(dtype) -> int:
    """Minimal numpy dtype -> GDAL type mapper."""
    dt = np.dtype(dtype)
    if dt == np.uint8:
        return gdal.GDT_Byte
    if dt == np.int16:
        return gdal.GDT_Int16
    if dt == np.uint16:
        return gdal.GDT_UInt16
    if dt == np.int32:
        return gdal.GDT_Int32
    if dt == np.uint32:
        return gdal.GDT_UInt32
    if dt == np.float32:
        return gdal.GDT_Float32
    if dt == np.float64:
        return gdal.GDT_Float64
    # Fallback: use Float32
    return gdal.GDT_Float32

def correct_raster_streaming(
    infile: str,
    outfile: str,
    band_formulas: Sequence[BandFormula],
    sensor: SensorParams,
    out: OutputSpec = OutputSpec(),
    build_overviews: bool = False,
    scale_to_int: bool = False,
    scale_factor: float = 10000.0
) -> None:
    """
    Apply per-band atmospheric (or radiometric) corrections efficiently
    using GDAL.

    Parameters
    ----------
    infile : str
        Path to input raster (single or multi-band).
    outfile : str
        Path to output corrected raster.
    band_formula : Sequence[BandFormula]
        For each band, either:
          - a string key in FORMULAS (e.g., "DOS3"), or
          - a callable f(band, params, ctx) -> float32 array.
        Length must equal the source band count.
    sensor : SensorParams
        Per-band parameter objects list and global parameters for the chosen sensor.
    out : OutputSpec
        Output configuration (dtype, scaling, compression, etc.).
    build_overviews : bool
        If True, builds overview levels (2,4,8,16) with average resampling.
    scale_to_int : bool, optional
        If True, multiply results by `scale_factor` and convert to int16.
        Useful for storing reflectance-like outputs as scaled integers.
    scale_factor : float, optional
        Constant factor used when `scale_to_int=True`. Default is 10000.

    Notes
    -----
    - Memory-safe for very large rasters: processes by native GDAL tile windows.
    - Nodata handling:
        * If the source defines `src.nodata`, those pixels are preserved as nodata.
        * Otherwise zeros are treated as nodata by default; adjust as needed by
          pre-setting a nodata in the source or customizing the mask logic below.
    - Scaling and clipping:
        * If `out.scale` is not None, results are multiplied and then clipped.
        * If `out.clamp` is None and `out.dtype` is integer, clip to dtype range.
    """

    # Resolve callable formulas (strings -> functions)
    funcs: Tuple[Callable, ...] = tuple(
        FORMULAS[f] if isinstance(f, str) else f
        for f in band_formulas
    )

    # Store applied functions to compute the logs
    str_formulas = {}

    # Multi-threading hints for GDAL (similar to rasterio.Env options)
    gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
    gdal.SetConfigOption("GDAL_TIFF_OVR_BLOCKSIZE", "128")

    src = gdal.Open(infile, gdal.GA_ReadOnly)
    if src is None:
        raise RuntimeError(f"Could not open input raster: {infile}")

    xsize = src.RasterXSize
    ysize = src.RasterYSize
    band_count = src.RasterCount

    if band_count != len(funcs) or band_count != len(sensor.bands):
        src = None
        raise ValueError("The number of bands, formulas, and band params must have equal lengths")

    # Get basic geo-info
    geotransform = src.GetGeoTransform()
    projection = src.GetProjection()
    metadata = src.GetMetadata()

    # Determine output dtype (numpy + GDAL)
    if scale_to_int:
        np_out_dtype = np.dtype("int16")
    else:
        np_out_dtype = np.dtype(out.dtype)

    gdal_dtype = _numpy_dtype_to_gdal(np_out_dtype)

    # Determine clipping range
    if out.clamp is not None:
        clamp_min, clamp_max = map(float, out.clamp)
    else:
        if np.issubdtype(np_out_dtype, np.integer):
            clamp_min, clamp_max = _dtype_range(np_out_dtype)
        else:
            clamp_min, clamp_max = -np.inf, np.inf

    # Build any convenient derived ctx once
    ctx = replace(sensor.ctx)  # shallow copy

    # Choose driver
    driver = gdal.GetDriverByName("GTiff")
    # Creation options
    creation_options = []

    # Compression
    if out.compress:
        # e.g. "DEFLATE", "LZW", etc.
        creation_options.append(f"COMPRESS={out.compress}")

    # Tiling
    if out.tiled or out.blocksize:
        creation_options.append("TILED=YES")

    # Block size / tile size
    if out.blocksize:
        blocksize = int(out.blocksize)
        # GTiff uses BLOCKXSIZE/BLOCKYSIZE
        creation_options.append(f"BLOCKXSIZE={blocksize}")
        creation_options.append(f"BLOCKYSIZE={blocksize}")
    else:
        # Try to preserve native tiling from the first band
        band1 = src.GetRasterBand(1)
        bx, by = band1.GetBlockSize()
        # If not full raster scanline, assume it is tiled and preserve it
        if not (by == ysize and bx == xsize):
            creation_options.append(f"BLOCKXSIZE={bx}")
            creation_options.append(f"BLOCKYSIZE={by}")

    # BIGTIFF
    if getattr(out, "bigtiff", None):
        # out.bigtiff might be "YES", "NO", "IF_NEEDED", etc.
        creation_options.append(f"BIGTIFF={out.bigtiff}")

    # Predictor
    if getattr(out, "predictor", None) is not None:
        creation_options.append(f"PREDICTOR={out.predictor}")

    # Create output dataset paths
    log_path = outfile.with_suffix(".log")
    if getattr(out, "cog", False):
        outfile_cog = outfile
        outfile = outfile.parent / outfile.name.replace(".tif", "_temp.tif")

    dst = driver.Create(
        str(outfile),
        xsize,
        ysize,
        band_count,
        gdal_dtype,
        options=creation_options
    )
    if dst is None:
        src = None
        raise RuntimeError(f"Could not create output raster: {outfile}")

    # Set geo-info
    if geotransform is not None:
        dst.SetGeoTransform(geotransform)
    if projection:
        dst.SetProjection(projection)
    if metadata:
        dst.SetMetadata(metadata)

    # Set per-band nodata on output bands
    for bidx in range(1, band_count + 1):
        dst_band = dst.GetRasterBand(bidx)
        if out.nodata is not None:
            dst_band.SetNoDataValue(float(out.nodata))

    # Determine processing block size (from output, same as input band1)
    band1 = src.GetRasterBand(1)
    blockx, blocky = band1.GetBlockSize()
    if blockx <= 0 or blocky <= 0:
        # Fallback to full-width scanline
        blockx, blocky = xsize, 1

    # Main windowed processing loop
    for yoff in range(0, ysize, blocky):
        win_ysize = min(blocky, ysize - yoff)

        for xoff in range(0, xsize, blockx):
            win_xsize = min(blockx, xsize - xoff)

            # Process each band for this block
            for bidx in range(1, band_count + 1):
                src_band = src.GetRasterBand(bidx)
                dst_band = dst.GetRasterBand(bidx)

                arr = src_band.ReadAsArray(
                    xoff, yoff, win_xsize, win_ysize
                )
                if arr is None:
                    raise RuntimeError(
                        f"Failed reading block at x={xoff}, y={yoff}, band={bidx}"
                    )

                arr = arr.astype("float32", copy=False)

                # Input nodata mask: prefer declared nodata,
                # else treat zeros as nodata
                src_nodata = src_band.GetNoDataValue()
                if src_nodata is not None and np.isfinite(src_nodata):
                    nodata_val = float(src_nodata)
                else:
                    nodata_val = 0
                mask = (arr == nodata_val)

                # Prepare output array filled with nodata
                out_arr = np.full(arr.shape, nodata_val, dtype=np.float32)

                # Indices of valid pixels
                valid = ~mask
                has_valid = np.any(valid)
                if has_valid:
                    # Run the band-specific formula only on valid pixels
                    # Note: funcs[bidx - 1] should be able to handle a 1D array
                    out_valid, str_func = funcs[bidx - 1](
                        arr[valid],              # only valid pixels (1D)
                        sensor.bands[bidx - 1],
                        sensor.ctx,
                        sensor.cal_func
                    )

                    # Put calibrated values back into full output array
                    out_arr[valid] = out_valid
                else:
                    # No valid data at all
                    str_func = None

                if bidx not in str_formulas:
                    str_formulas[bidx] = str_func

                # Optional scaling (out.scale)
                if has_valid and getattr(out, "scale", None) is not None:
                    scale_val = float(out.scale)
                    if _USE_NUMEXPR:
                        out_arr = ne.evaluate(
                            "out_arr * scale",
                            local_dict={
                                "out_arr": out_arr,
                                "scale": scale_val
                            })
                    else:
                        out_arr *= scale_val

                # Scale to int if requested (valid pixels will have been
                # changed already; nodata will be restored later)
                if has_valid and scale_to_int:
                    out_arr *= scale_factor
                    out_arr = np.clip(
                        out_arr,
                        0,
                        np.iinfo(np.int16).max
                    )
                    out_arr = out_arr.astype(np.int16, copy=False)

                # Clip
                if has_valid and (not np.isinf(clamp_min) or not np.isinf(clamp_max)):
                    out_arr = np.clip(out_arr, clamp_min, clamp_max)

                # Final cast to output dtype
                out_arr = out_arr.astype(np_out_dtype, copy=False)

                # Restore nodata on masked pixels
                if np.issubdtype(np_out_dtype, np.floating) and np.isnan(out.nodata):
                    out_arr[mask] = np.nan
                else:
                    out_arr[mask] = out.nodata

                # Write block
                dst_band.WriteArray(out_arr, xoff, yoff)

    # Overviews
    if build_overviews:
        factors = [2, 4, 8, 16]
        # "AVERAGE" resampling similar to rasterio.enums.Resampling.average
        dst.BuildOverviews("AVERAGE", factors)
        # Optionally, you can also set overview metadata if desired:
        ovr_md = dst.GetMetadata("OVERVIEWS")
        if ovr_md is None:
            ovr_md = {}
        ovr_md["RESAMPLING"] = "AVERAGE"
        dst.SetMetadata(ovr_md, "OVERVIEWS")

    # Flush to disk
    dst.FlushCache()
    dst = None
    src = None

    # Translate into COG
    if getattr(out, "cog", False):
        subprocess.run([
            "gdal_translate",
            str(outfile),
            str(outfile_cog),
            "-of", "COG",
            "-co", "COMPRESS=DEFLATE",
            "-co", "PREDICTOR=2",
            "-co", "BIGTIFF=IF_SAFER"
        ], check=True)
        # Remove the temp file
        outfile.unlink()
    
    # Write logs 
    for i in range(len(str_formulas)):
        write_log(
            sensor.bands[i],
            str_formulas[i + 1],
            i + 1,
            log_path
        )
