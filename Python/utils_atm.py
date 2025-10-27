
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple, Union
from datetime import datetime, timezone
from pathlib import Path
from Py6S import SixS, Geometry, AtmosProfile, AeroProfile, Wavelength

import ee
import math
import json
import numpy as np
import rasterio

try:
    import numexpr as ne
    _USE_NUMEXPR = True
except Exception:
    _USE_NUMEXPR = False

ee.Initialize()

def write_log(params, formula, band_id, log_file):
    """Store the logs of the computed image."""
    # Unique file name per run
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    with open(log_file, "a") as f:
        f.write(f"[{ts}] B{band_id} - PARAMS: {json.dumps(params)}\n")
        f.write(f"[{ts}] B{band_id} - FORM: {json.dumps(formula)}\n")

# ----------------------------- Formula Registry -----------------------------

"""
Each formula is a small function with signature:

    f(band: np.ndarray, params: dict, ctx: dict) -> np.ndarray

- `band`  : the input band window as float32
- `params`: per-band parameters (e.g., gain, offset, L_min, L_max, etc.)
- `ctx`   : shared context (e.g., sin_theta, earth_sun_distance, esun, etc.)

Return value MUST be a float32 array with the same shape as `band`.
"""

def dark_object_subtraction_v3(
    band: np.ndarray,
    params: dict,
    ctx: dict,
) -> tuple[np.ndarray, str]:
    """
    Dark Object Subtraction (DOS) atmospheric correction.

    Converts a radiometrically calibrated band (DNs) to surface reflectance
    using a Chavez-style DOS2/DOS3 refinement:
      - Estimate haze radiance from a 'dark object' DN
      - Subtract (L_haze - L_1%) from TOA radiance
      - Convert to reflectance using ESUN (Thuillier-based), Earth-Sun
        distance, solar zenith, and a simple Rayleigh transmittance term.

    Parameters
    ----------
    band : np.ndarray
        Input image band as DNs (any dtype); will be cast to float32.
    params : dict
        Required keys:
          - "gain", "offset": radiometric coefficients (DN->radiance)
          - "abs_factor", "effective_band_width": sensor radiometric terms
          - "e_thuillier": band ESUN (at 1 AU; same units as radiance)
          - "lamb_scatter": scale for DN of dark object -> haze DN (empirical)
          - "dark_object_dn": DN at which haze is estimated
          - "lowerBandEdge", "upperBandEdge": band edges in micrometers
    ctx : dict
        Required keys:
          - "capture_date": datetime.date or datetime.datetime
            (for Earth–Sun distance)
          - "sol_zen_rad": solar zenith angle in radians

    Returns
    -------
    (np.ndarray, str)
        Tuple of (reflectance array float32, stringified formula used).
    """

    # ---- 1) Parse inputs (fail loudly if something is missing) ----
    try:
        gain = float(params["gain"])
        offset = float(params["offset"])
        abs_factor = float(params["abs_factor"])
        effective_bwidth = float(params["effective_band_width"])
        e_thuillier = float(params["e_thuillier"])
        scale_coef = float(params["lamb_scatter"])
        L_haze_start_dn = float(params["dark_object_dn"])
        min_wav = float(params["lowerBandEdge"])
        max_wav = float(params["upperBandEdge"])

        capture_date = ctx["capture_date"]
        d = earth_sun_distance_au(capture_date)  # must return AU
        sun_zen_ang = float(ctx["sun_zen_ang"])
        # Extract solar zenith angle and convert it to radians.
        # Scientific Python math functions (e.g., math.cos) expect radians.
        sun_zen_ang = math.radians(sun_zen_ang)
    except KeyError as e:
        raise ValueError(f"Missing required parameter: {e}") from e
    except Exception as e:
        raise ValueError(f"Invalid parameter types/values: {e}") from e

    # Center wavelength (micrometers) for Rayleigh term
    cwav = (min_wav + max_wav) / 2.0

    # ---- 2) DN -> TOA radiance (ARC) ----
    # Avoid taking abs() here; radiance should be linear in DN with given
    # offset. Any negative radiance due to noise/offset is handled later
    # via clipping.
    band = band.astype("float32", copy=False)
    band_toa_rad = gain * band * (abs_factor / effective_bwidth) + offset

    # ---- 3) Haze estimation ----
    # Start with a DN for the dark object, scale it (empirical), and
    # convert to radiance.
    L_haze_dn = L_haze_start_dn * scale_coef
    L_haze = gain * L_haze_dn * (abs_factor / effective_bwidth) + offset  # radiance

    # Chavez L1% (radiance for 1% surface reflectance) to refine path radiance
    # L_1% = (0.01 * ESUN * cos(theta_s)) / (d^2 * pi)
    L_one = (0.01 * e_thuillier * abs(math.cos(sun_zen_ang))) / (d**2 * math.pi)

    # Path radiance (refined)
    L_haze = L_haze - L_one

    # ---- 4) Simple Rayleigh (molecular) transmittance term ----
    # Rayleigh optical thickness at wavelength (micrometers).
    tau_R = 0.008569 * (cwav ** -4) * (1.0 + 0.0113 * (cwav ** -2) + 0.00013 * (cwav ** -4))
    mu0 = abs(math.cos(sun_zen_ang))  # cosine of SZA
    # Downward Rayleigh transmittance (single path): Tz = exp(-tau_R / mu0)
    # (Some formulations use two-way; here it’s included once as
    # in many DOS recipes.)
    T_z = math.exp(-tau_R / max(mu0, 1e-6))  # guard against division by ~0 at very high SZA

    # ---- 5) Reflectance formula (Chavez-style DOS) ----
    # p = pi * ( (L_toa - L_haze) * d^2 ) / (ESUN * cos(theta_s) * T_z)
    # Build a string version for logging:
    str_formula = (
        f"(pi * ((band_toa_rad - {L_haze}) * {d}**2) / "
        f"({e_thuillier} * abs(cos({sun_zen_ang})) * {T_z}))"
    )

    # Compute reflectance
    # Note: computing numerator and denominator separately for clarity.
    numerator = math.pi * (band_toa_rad - L_haze) * (d**2)
    denominator = e_thuillier * mu0 * T_z
    # Avoid division by zero
    if denominator <= 0:
        # If mu0 ~ 0 (SZA ~ 90) or T_z <= 0, reflectance isn't defined;
        # return zeros.
        refl = np.zeros_like(band_toa_rad, dtype="float32")
        return refl, str_formula

    band_dos = numerator / denominator

    # ---- 6) Clip negatives (physically meaningless) ----
    if _USE_NUMEXPR:
        # Use numexpr for speed on large arrays; MUST assign the result
        band_dos = ne.evaluate(
            "where(band_dos > 0, band_dos, 0)", 
            local_dict={"band_dos": band_dos}
        )
    else:
        band_dos = np.where(band_dos > 0, band_dos, 0.0)

    return band_dos.astype("float32", copy=False), str_formula

def second_simulation_solar_spectrum(
    band: np.ndarray,
    params: dict,
    ctx: dict,
) -> tuple[np.ndarray, str]:
    """
    Surface reflectance from TOA radiance using the 6S radiative transfer
    model (Py6S).

    Pipeline:
      1) DN -> TOA radiance using sensor radiometric terms
      2) Build 6S scene (geometry, atmosphere, aerosols, altitudes, wavelength)
      3) Run 6S to obtain:
           - Lp (atmospheric intrinsic/path radiance)
           - Edir, Edif (direct/diffuse solar irradiance at surface)
           - Transmittances (down/up) to undo atmospheric effects
      4) Invert 6S forward model:
           p = pi * (L_toa - Lp) / [ T_up * (Edir * T_down + Edif) ]

    Parameters
    ----------
    band : np.ndarray
        Input band as DNs (any dtype); will be cast to float32.
    params : dict
        Required:
          - "gain", "offset" : radiometric coefficients (DN -> radiance)
          - "abs_factor", "effective_band_width" : sensor radiometric terms
          - "lowerBandEdge", "upperBandEdge" : band edges in micrometers
    ctx : dict
        Required:
          - "bbox" : [minLon, minLat, maxLon, maxLat] (EPSG:4326)
          - "capture_date" : ISO UTC string, e.g. "YYYY-MM-DDTHH:MM:SS.sssZ"
          - "meanSatAz" : sensor azimuth (deg)
          - "sol_zen_ang" : sun elevation (deg)
          - "meanSunAz" : sun azimuth (deg)

    Returns
    -------
    (np.ndarray, str)
        (surface reflectance float32, human-readable formula string)
    """
    # ---- 1) Parse inputs (fail loudly if something is missing) ----
    try:
        gain = float(params["gain"])
        offset = float(params["offset"])
        abs_factor = float(params["abs_factor"])
        effective_bwidth = float(params["effective_band_width"])
        min_wav = float(params["lowerBandEdge"])
        max_wav = float(params["upperBandEdge"])

        geom_bbox = ctx["bbox"]  # EPSG:4326 [minLon, minLat, maxLon, maxLat]
        capture_date_txt = ctx["capture_date"] # UTC ISO8601 string
        mean_sat_az = float(ctx["meanSatAz"])  # degrees
        mean_sun_az = float(ctx["meanSunAz"])  # degrees
        sun_zen_ang = float(ctx["sun_zen_ang"])  # degrees
    except KeyError as e:
        raise ValueError(f"Missing required parameter: {e}") from e
    except Exception as e:
        raise ValueError(f"Invalid parameter types/values: {e}") from e
    
    # ---- 2) DN -> TOA radiance (ARC) ----
    # Avoid taking abs() here; radiance should be linear in DN with given
    # offset. Any negative radiance due to noise/offset is handled later
    # via clipping.
    band = band.astype("float32", copy=False)
    band_toa_rad = gain * band * (abs_factor / effective_bwidth) + offset

    # ---- 3) Build 6S scene ----
    # The backbone of Py6S is the 6S (i.e. SixS) class.
    # It allows to define the input parameters, to run the radiative transfer
    # code and to access transfer model outputs.
    s = SixS()

    # 3.1 Geometry (Py6S expects DEGREES)
    s.geometry = Geometry.User()
    # Parse UTC capture date
    capture_dt = datetime.strptime(capture_date_txt, "%Y-%m-%dT%H:%M:%S.%fZ")
    # Data used for Earth-Sun distance in 6S
    s.geometry.month = capture_dt.month
    s.geometry.day = capture_dt.day

    # Sensor angles
    # If the acquisition is near-nadir, view_z = 0 is OK.
    # If off-nadir, set view_z accordingly.
    s.geometry.view_z = 0.0
    # Sensor azimuth angle
    s.geometry.view_a = mean_sat_az

    # Solar angles (degrees). 6S requires solar ZENITH, not elevation
    s.geometry.solar_z = sun_zen_ang
    # Solar azimuth angle
    s.geometry.solar_a = mean_sun_az

    
    # 3.2 Atmospheric profile & aerosols
    # There are Earth Engine-based retrievals here:
    # If you don't want EE, replace with your own inputs.
    # --- BEGIN Earth Engine-dependent block ---
    footprint = ee.Geometry.Rectangle(geom_bbox)
    geom = footprint.centroid()
    date_ee = ee.Date(f"{capture_dt.year}-{capture_dt.month}-{capture_dt.day}")
    h2o = Atmospheric.water(geom, date_ee).getInfo()  # g/cm^2 or cm of precipitable water (check units)
    o3  = Atmospheric.ozone(geom, date_ee).getInfo()  # atm-cm
    aot = Atmospheric.aerosol(geom, date_ee).getInfo()  # AOT at 550 nm

    # Get target elevation
    # Altitude - Shuttle Radar Topography mission (covers *most* of the Earth)
    srtm_col = ee.Image('CGIAR/SRTM90_V4') 
    alt = srtm_col.reduceRegion(
        reducer = ee.Reducer.mean(),
        geometry = geom.centroid()
    ).get('elevation').getInfo()
    km = alt/1000 # i.e. Py6S uses units of kilometers
    # --- END Earth Engine-dependent block ---
    # If you're not using EE, pass user-provided values instead:
    # h2o = ctx["column_water_vapor"]   # e.g., g/cm^2
    # o3  = ctx["ozone"]                # atm-cm
    # aot = ctx["aot550"]               # unitless at 550 nm

    # Atmospheric constituents
    s.atmos_profile = AtmosProfile.UserWaterAndOzone(h2o, o3)
    # Choose an aerosol model appropriate to your scene;
    # Continental is a safer default than Desert
    s.aero_profile = AeroProfile.Continental
    # s.aero_profile = AeroProfile.Desert
    s.aot550 = aot

    # 3.3 Altitudes
    # Target altitude (terrain) in km a.s.l. If you don't have a DEM,
    # 0 km is a reasonable fallback.
    s.altitudes.set_target_custom_altitude(km)
    # IMPORTANT: set SENSOR altitude in km a.s.l.
    # If you don't know platform altitude, use "satellite level" which places
    # the sensor above atmosphere.
    # Note: meanSatEl is in angles, not kilometers
    s.altitudes.set_sensor_satellite_level()

    # 3.4 Wavelength
    # Using a flat spectral response between [min_wav, max_wav] (micrometers).
    # For highest fidelity, integrate actual sensor RSR if available.
    # Wavelength function: 
    # https://github.com/robintw/Py6S/blob/master/Py6S/Params/wavelength.py
    s.wavelength = Wavelength(min_wav, max_wav)

    # ---- 4) Run 6S ----
    s.run()

    # ---- 5) Extract outputs & invert the 6S forward model ----
    # Path radiance (Lp) [same units as radiance]
    Lp = s.outputs.atmospheric_intrinsic_radiance

    # Direct and diffuse solar irradiance at the surface (Edir, Edif)
    # NOTE: These already include Earth–Sun distance effects inside 6S.
    Edir = s.outputs.direct_solar_irradiance
    Edif = s.outputs.diffuse_solar_irradiance


    # Transmittances:
    # Use downwelling and upwelling TOTAL transmittances
    # (gas * scattering * aerosol).
    # Py6S exposes components; a robust way is to use 'total' if available,
    # otherwise multiply gas and scattering (and aerosol) components.
    trans = s.outputs.trans  # dict-like

    # Downward path (sun->surface)
    try:
        T_down = trans['total'].downward
        T_up   = trans['total'].upward
    except KeyError:
        # Fallback if 'total' is not present:
        # multiply gas and total scattering.
        T_down = trans['global_gas'].downward * trans['total_scattering'].downward
        T_up   = trans['global_gas'].upward   * trans['total_scattering'].upward

    # Inversion:
    # L_toa = Lp + (p/pi) * T_up * (Edir * T_down + Edif)
    #  =>  p = pi * (L_toa - Lp) / [ T_up * (Edir * T_down + Edif) ]
    numerator = math.pi * (band_toa_rad - Lp)
    irradiance_term = (Edir * T_down) + Edif
    denominator = T_up * irradiance_term

    # Build a readable formula for logs (using scalar placeholders)
    str_formula = (
        f"pi*(band_toa_rad - {Lp}) / "
        f"({T_up} * ({Edir}*{T_down} + {Edif}))"
    )

    # Denominator can be scalar; if it is, broadcast works fine.
    # Guard against zero/negative (can happen in extreme conditions).
    # If denominator is an array (unlikely here), use elementwise guard.
    if np.isscalar(denominator):
        if denominator <= 0:
            refl = np.zeros_like(band_toa_rad, dtype="float32")
            return refl, str_formula
    else:
        # Avoid division-by-zero pixel-wise
        denominator = np.where(denominator > 0, denominator, np.nan)

    band_6s = numerator / denominator

    # ---- 6) Clip negatives (not physically meaningful) & NaNs ----
    if _USE_NUMEXPR:
        band_6s = ne.evaluate("where(band_6s > 0, band_6s, 0)")
        # Replace any NaNs from denominator fixes
        band_6s = ne.evaluate("where(isnan(band_6s), 0, band_6s)")
    else:
        band_6s = np.where(band_6s > 0, band_6s, 0.0)
        band_6s = np.nan_to_num(band_6s, nan=0.0, posinf=0.0, neginf=0.0)

    return band_6s.astype("float32", copy=False), str_formula

# Registry mapping short names to callables
FORMULAS: Dict[str, Callable[[np.ndarray, dict, dict], np.ndarray]] = {
    "DOS3": dark_object_subtraction_v3,
    "S6": second_simulation_solar_spectrum
}


# ----------------------------- I/O Orchestration -----------------------------

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
    cog: Optional[bool] = False


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


def correct_raster_streaming(
    infile: str,
    outfile: str,
    band_formulas: Sequence[BandFormula],
    band_params: Sequence[dict],
    ctx_params: dict,
    out: OutputSpec = OutputSpec(),
    build_overviews: bool = False,
    scale_to_int: bool = False,
    scale_factor: float = 10000.0
) -> None:
    """
    Apply per-band atmospheric (or radiometric) corrections efficiently
    using windowed I/O, with a **different formula per band if desired**.

    Parameters
    ----------
    infile : str
        Path to input raster (single or multi-band).
    outfile : str
        Path to output corrected raster.
    band_formulas : Sequence[BandFormula]
        For each band, either:
          - a string key in FORMULAS (e.g., "DOS3"), or
          - a callable f(band, params, ctx) -> float32 array.
        Length must equal the source band count.
    band_params : Sequence[dict]
        Per-band parameter dicts for the chosen formulas
        (same length as bands).
    ctx_params : dict
        Shared context parameters available to all formulas, such as:
          - "sin_theta": float (sin of solar elevation)
          - "cos_theta": float (cos of solar zenith)
          - "d": float Earth–Sun distance in AU
          - any other values your custom formulas require.
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

    with rasterio.Env(
        GDAL_NUM_THREADS="ALL_CPUS",
        NUM_THREADS="ALL_CPUS"
    ), rasterio.open(infile) as src:

        if src.count != len(funcs) or src.count != len(band_params):
            raise ValueError("bands, band_formulas, and band_params must have equal lengths")

        profile = src.profile.copy()
        profile.update(
            dtype=out.dtype,
            nodata=out.nodata,
            compress=out.compress,
            tiled=out.tiled,
        )
        if out.predictor is not None:
            profile.update(predictor=out.predictor)
        if out.cog:
            profile.update(driver="COG")
        if out.blocksize:
            profile.update(blockxsize=out.blocksize, blockysize=out.blocksize)
        else:
            # Preserve native tiling if the source is tiled
            try:
                by, bx = src.block_shapes[0]
                if (by, bx) != (src.height, src.width):
                    profile.update(blockxsize=bx, blockysize=by)
            except Exception:
                pass
        if out.bigtiff:
            profile.update(BIGTIFF=out.bigtiff)

        if scale_to_int:
            profile.update(dtype="int16")

        # Determine clipping range
        if out.clamp is not None:
            clamp_min, clamp_max = map(float, out.clamp)
        else:
            if np.issubdtype(np.dtype(out.dtype), np.integer):
                clamp_min, clamp_max = _dtype_range(out.dtype)
            else:
                # For float outputs, default to no clamp
                clamp_min, clamp_max = -np.inf, np.inf

        # Build any convenient derived ctx once
        ctx = dict(ctx_params)  # shallow copy

        with rasterio.open(outfile, "w", **profile) as dst:
            for _, window in dst.block_windows(1):
                # Pre-compute mask per band then apply formula
                for bidx in range(1, src.count + 1):
                    arr = (
                        src.read(bidx, window=window)
                        .astype("float32", copy=False)
                    )

                    # Input nodata mask: prefer declared nodata,
                    # else treat zeros as nodata
                    if src.nodata is not None and np.isfinite(src.nodata):
                        mask = (arr == float(src.nodata))
                    else:
                        mask = (arr == 0)

                    # Apply band-specific formula
                    out_arr, str_func = funcs[bidx - 1](arr, band_params[bidx - 1], ctx)

                    if bidx not in str_formulas:
                        str_formulas[bidx] = str_func

                    # Optional scaling
                    if out.scale is not None:
                        if _USE_NUMEXPR:
                            out_arr = ne.evaluate(
                                "out_arr * scale",
                                local_dict={
                                    "out_arr": out_arr, 
                                    "scale": float(out.scale)
                                })
                        else:
                            out_arr *= float(out.scale)
                    
                    # Scale to int if requested
                    if scale_to_int:
                        out_arr *= scale_factor
                        out_arr = np.clip(out_arr, 0, np.iinfo(np.int16).max)
                        out_arr = out_arr.astype(np.int16)
                        # Force output dtype update only once
                        if dst.profile["dtype"] != "int16":
                            dst.profile.update(dtype="int16")
                    
                    # Clip then cast
                    if not np.isinf(clamp_min) or not np.isinf(clamp_max):
                        out_arr = np.clip(out_arr, clamp_min, clamp_max)

                    out_arr = out_arr.astype(out.dtype, copy=False)

                    # Restore nodata
                    if np.issubdtype(np.dtype(out.dtype), np.floating) and np.isnan(out.nodata):
                        out_arr[mask] = np.nan
                    else:
                        out_arr[mask] = out.nodata

                    dst.write(out_arr, bidx, window=window)
                    

            if build_overviews:
                factors = [2, 4, 8, 16]
                dst.build_overviews(factors, rasterio.enums.Resampling.average)
                dst.update_tags(ns="rio_overview", resampling="average")

    for i in range(len(str_formulas)):
        write_log(
            band_params[i],
            str_formulas[i+1],
            i + 1,
            outfile.with_suffix(".log"))

def _parse_utc(utc: Union[str, datetime]) -> datetime:
    """
    Return a timezone-aware datetime in UTC.
    Accepts:
      - ISO 8601 strings like 'YYYY-MM-DDTHH:MM:SSZ' or with '.%f'
      - datetime (naive -> assumed UTC; aware -> converted to UTC)
    """
    if isinstance(utc, datetime):
        dt = utc
    else:
        s = utc.strip()
        # Normalize trailing Z (UTC) to a proper offset if present.
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        # Try fromisoformat first (fast path, Python 3.11+ handles +00:00).
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            # Fallback to common IMD formats with/without fractional seconds.
            for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ",
                        "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
                try:
                    dt = datetime.strptime(utc, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError(f"Unsupported UTC datetime format: {utc!r}")

    # Ensure tz-aware in UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def julian_day(utc: Union[str, datetime]) -> float:
    """
    Compute the astronomical Julian Day (JD) for a UTC instant.

    Implementation uses the Unix-epoch relation:
        JD = (unix_seconds / 86400) + 2440587.5
    where Unix epoch (1970-01-01T00:00:00Z) corresponds to JD 2440587.5.
    """
    dt = _parse_utc(utc)
    unix_seconds = dt.timestamp()  # seconds since 1970-01-01T00:00:00Z
    return unix_seconds / 86400.0 + 2440587.5


def earth_sun_distance_au(utc: Union[str, datetime]) -> float:
    """
    Approximate Earth–Sun distance in astronomical units for a UTC instant.

    Uses the low-precision USNO series
    (sufficient for remote-sensing irradiance):
        D = JD - 2451545.0
        g = 357.529° + 0.98560028° * D   (mean anomaly of the Sun)
        r = 1.00014 - 0.01671*cos(g) - 0.00014*cos(2g)
    Returns:
        r (float): distance in AU (Astronomical Units)
    """
    jd = julian_day(utc)
    D = jd - 2451545.0
    g_deg = 357.529 + 0.98560028 * D
    g = math.radians(g_deg)
    return 1.00014 - 0.01671 * math.cos(g) - 0.00014 * math.cos(2 * g)


class Atmospheric:
    """
    from "atmospheric.py", Sam Murphy (2016-10-26)

    Atmospheric water vapour, ozone and AOT from GEE

    Usage
    H2O = Atmospheric.water(geom,date)
    O3 = Atmospheric.ozone(geom,date)
    AOT = Atmospheric.aerosol(geom,date)
    """
    def round_date(date,xhour):
        """
        rounds a date of to the closest 'x' hours
        """
        y = date.get('year')
        m = date.get('month')
        d = date.get('day')
        H = date.get('hour')
        HH = H.divide(xhour).round().multiply(xhour)
        return date.fromYMD(y,m,d).advance(HH,'hour')

    def round_month(date):
        """
        round date to closest month
        """
        # start of THIS month
        m1 = date.fromYMD(date.get('year'),date.get('month'),ee.Number(1))

        # start of NEXT month
        m2 = m1.advance(1,'month')

        # difference from date
        d1 = ee.Number(date.difference(m1,'day')).abs()
        d2 = ee.Number(date.difference(m2,'day')).abs()

        # return closest start of month
        return ee.Date(ee.Algorithms.If(d2.gt(d1),m1,m2))

    def water(geom,date):
        """
        Water vapour column above target at time of image aquisition.

        (Kalnay et al., 1996, The NCEP/NCAR 40-Year Reanalysis Project. Bull.
        Amer. Meteor. Soc., 77, 437-471)
        """

        # Point geometry required
        centroid = geom.centroid(10)

        # H2O datetime is in 6 hour intervals
        H2O_date = Atmospheric.round_date(date,6)

        # filtered water collection
        water_ic = ee.ImageCollection('NCEP_RE/surface_wv').filterDate(H2O_date, H2O_date.advance(1,'month'))

        # water image
        water_img = ee.Image(water_ic.first())

        # water_vapour at target
        water = water_img.reduceRegion(reducer=ee.Reducer.mean(), geometry=centroid).get('pr_wtr')

        # convert to Py6S units (Google = kg/m^2, Py6S = g/cm^2)
        water_Py6S_units = ee.Number(water).divide(10)

        return water_Py6S_units

    def ozone(geom,date):
        """
        returns ozone measurement from merged TOMS/OMI dataset

        OR

        uses our fill value (which is mean value for that latlon and
        day-of-year)
        """
        # Point geometry required
        centroid = geom.centroid(10)

        def ozone_measurement(centroid,O3_date):

            # filtered ozone collection
            ozone_ic = ee.ImageCollection('TOMS/MERGED').filterDate(O3_date, O3_date.advance(1,'month'))

            # ozone image
            ozone_img = ee.Image(ozone_ic.first())

            # ozone value IF TOMS/OMI image exists ELSE use fill value
            ozone = ee.Algorithms.If(ozone_img,\
            ozone_img.reduceRegion(reducer=ee.Reducer.mean(), geometry=centroid).get('ozone'),\
            ozone_fill(centroid,O3_date))

            return ozone

        def ozone_fill(centroid,O3_date):
            """
            Gets our ozone fill value (i.e. mean value for that doy and latlon)
            you can see it
            1) compared to LEDAPS: https://code.earthengine.google.com/8e62a5a66e4920e701813e43c0ecb83e
            2) as a video: https://www.youtube.com/watch?v=rgqwvMRVguI&feature=youtu.be
            """

            # ozone fills (i.e. one band per doy)
            ozone_fills = ee.ImageCollection(
               'users/samsammurphy/public/ozone_fill').toList(366)

            # day of year index
            jan01 = ee.Date.fromYMD(O3_date.get('year'),1,1)
            doy_index = date.difference(jan01,'day').toInt() # (NB. index is one less than doy, so no need to +1)

            # day of year image
            fill_image = ee.Image(ozone_fills.get(doy_index))

            # return scalar fill value
            return (fill_image
                    .reduceRegion(reducer=ee.Reducer.mean(), geometry=centroid)
                    .get('ozone'))

        # O3 datetime in 24 hour intervals
        O3_date = Atmospheric.round_date(date,24)

        # TOMS temporal gap
        TOMS_gap = ee.DateRange('1994-11-01','1996-08-01')

        # avoid TOMS gap entirely
        ozone = ee.Algorithms.If(
           TOMS_gap.contains(O3_date),
           ozone_fill(centroid,O3_date),
           ozone_measurement(centroid,O3_date))

        # fix other data gaps (e.g. spatial, missing images, etc..)
        ozone = ee.Algorithms.If(ozone,ozone,ozone_fill(centroid,O3_date))

        #convert to Py6S units
        ozone_Py6S_units = ee.Number(ozone).divide(1000) # (i.e. Dobson units are milli-atm-cm )

        return ozone_Py6S_units


    def aerosol(geom,date):
        """
        Aerosol Optical Thickness.

        try:
            MODIS Aerosol Product (monthly)
        except:
            fill value
        """

        def aerosol_fill(date):
            """
            MODIS AOT fill value for this month (i.e. no data gaps)
            """
            return (
                ee.Image('users/samsammurphy/public/AOT_stack')
                .select([ee.String('AOT_').cat(date.format('M'))])
                .rename(['AOT_550'])
            )


        def aerosol_this_month(date):
            """
            MODIS AOT original data product for this month
            (i.e. some data gaps)
            """
            # image for this month
            img =  ee.Image(
                (ee.ImageCollection('MODIS/061/MOD08_M3')
                .filterDate(Atmospheric.round_month(date))
                .first())
            )

            # fill missing month (?)
            img = ee.Algorithms.If(
                img,
                # all good
                (img.select(['Aerosol_Optical_Depth_Land_Ocean_Mean_Mean'])
                    .divide(1000)
                    .rename(['AOT_550'])),
                # missing month
                aerosol_fill(date))

            return img


        def get_AOT(AOT_band,geom):
            """
            AOT scalar value for target
            """
            return (ee.Image(AOT_band)
                    .reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=geom.centroid(10)
                    )
                    .get('AOT_550'))


        after_modis_start = date.difference(
            ee.Date('2000-03-01'),'month').gt(0)

        AOT_band = ee.Algorithms.If(
            after_modis_start,
            aerosol_this_month(date), 
            aerosol_fill(date)
        )

        AOT = get_AOT(AOT_band,geom)

        # Check reduce region worked (else force fill value)
        AOT = ee.Algorithms.If(AOT,AOT,get_AOT(aerosol_fill(date),geom))
        return AOT