import math
import csv
from datetime import datetime
from typing import Tuple, Dict, List, Optional

import numpy as np

try:
    import numexpr as ne
    _USE_NUMEXPR = True
except Exception:
    _USE_NUMEXPR = False

# Py6S
from Py6S import SixS, Geometry, AtmosProfile, AeroProfile, Wavelength
from Py6S.Params.wavelength import SpectralResponseFunction


# ---------------------------
# RSR utilities (WV3 PAN/MS)
# ---------------------------

def load_rsr_csv(csv_path: str) -> Tuple[List[float], List[float]]:
    """
    Load a band Relative Spectral Response (RSR) from a CSV file.
    CSV format expected: wavelength_um,response (headers allowed or not).
    Returns two lists (wavelength_um, response), response normalized to peak=1.
    """
    wv, rsr = [], []
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            # skip header if present
            try:
                w = float(row[0])
                r = float(row[1])
            except Exception:
                continue
            wv.append(w)
            rsr.append(r)

    if len(wv) == 0 or len(rsr) == 0:
        raise ValueError(f"Empty/invalid RSR file: {csv_path}")

    # Normalize to peak=1 so only shape matters
    rsr = np.array(rsr, dtype=float)
    peak = np.max(rsr)
    if peak <= 0:
        raise ValueError("RSR has non-positive peak.")
    rsr = (rsr / peak).tolist()
    return wv, rsr


def rsr_to_py6s_wavelength(wavelength_um: List[float], response: List[float]) -> Wavelength:
    """
    Build a Py6S Wavelength object from an RSR vector.
    """
    if len(wavelength_um) != len(response):
        raise ValueError("RSR wavelength and response must have same length.")
    srf = SpectralResponseFunction(wavelength_um, response)
    return Wavelength(srf)


# ------------------------------------
# Atmospheric parameter acquisition
# ------------------------------------

def get_atmo_from_aeronet(
    date_utc: datetime,
    lon: float,
    lat: float,
    radius_km: float = 100.0,
    prefer_level: str = "L2.0"
) -> Tuple[float, float, float]:
    """
    Fetch atmospheric parameters from nearest AERONET site:
      - AOT at 550 nm (interpolated from AOT_500)
      - Column water vapor (cm or g/cm^2 depending on product; we convert)
      - Ozone (not from AERONET; we’ll return NaN here to be filled from CAMS)
    Implementation detail:
      - In production, call NASA AERONET web services (e.g., AERONET AOD, PWV).
      - Here we raise to indicate you must plug your fetcher, or use CAMS/manual.

    Returns: (aot550, h2o, ozone_atm_cm)
    """
    raise NotImplementedError("Implement AERONET fetch (nearest site, AOD & PWV) or use CAMS/manual.")


def get_atmo_from_cams(
    date_utc: datetime,
    lon: float,
    lat: float
) -> Tuple[float, float, float]:
    """
    Fetch atmospheric parameters from Copernicus CAMS (CAMS global reanalysis/forecasts):
      - AOT at 550 nm
      - Total column water vapor (kg/m^2 -> convert to cm of precipitable water)
      - Total column ozone (Dobson Units -> atm-cm)
    Implementation detail:
      - In production, pull via CDS API (cdsapi) or Atmosphere Data Store REST.
      - Here we raise to indicate you must plug your fetcher.

    Returns: (aot550, h2o, ozone_atm_cm)
    """
    raise NotImplementedError("Implement CAMS fetch via cdsapi/ADS, or pass manual values.")


def get_atmo_params(
    bbox: List[float],
    capture_dt: datetime,
    source: str,
    manual: Optional[Dict[str, float]] = None
) -> Tuple[float, float, float]:
    """
    Unified entry-point. Returns (aot550, column_water_vapor_cm, ozone_atm_cm).
    - source: "cams", "aeronet", or "manual"
    - bbox: [minLon, minLat, maxLon, maxLat] (we take center point)
    - manual: dict with keys {"aot550", "h2o_cm", "ozone_atm_cm"} if source="manual"
    """
    minlon, minlat, maxlon, maxlat = bbox
    lon = 0.5 * (minlon + maxlon)
    lat = 0.5 * (minlat + maxlat)

    if source.lower() == "manual":
        if not manual:
            raise ValueError("Manual source selected but no 'manual' dict provided.")
        aot = float(manual["aot550"])
        h2o_cm = float(manual["h2o_cm"])
        ozone_atm_cm = float(manual["ozone_atm_cm"])
        return aot, h2o_cm, ozone_atm_cm

    if source.lower() == "aeronet":
        # You can combine AERONET (AOT & PWV) + CAMS (ozone) in practice.
        return get_atmo_from_aeronet(capture_dt, lon, lat)

    if source.lower() == "cams":
        return get_atmo_from_cams(capture_dt, lon, lat)

    raise ValueError("Unknown source. Use 'cams', 'aeronet', or 'manual'.")


# ------------------------------------
# 6S correction (with RSR + ext. atmos)
# ------------------------------------

def surface_reflectance_6s_with_rsr(
    band_dn: np.ndarray,
    params: Dict,
    ctx: Dict,
    rsr_csv_path: str,
    atmos_source: str = "manual",
    manual_atmos: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, str]:
    """
    Atmospheric correction using Py6S with:
      - Real WV3 band RSR (loaded from CSV)
      - Atmospheric parameters from AERONET or CAMS (or manual)

    Inputs
    ------
    band_dn : np.ndarray
        Band digital numbers.
    params : dict
        Sensor radiometry & band metadata:
          - "gain", "offset"
          - "abs_factor", "effective_band_width"
          - "lowerBandEdge", "upperBandEdge"  (μm; not used for RSR but logged/validated)
    ctx : dict
        Scene context:
          - "bbox": [minLon, minLat, maxLon, maxLat]  (EPSG:4326)
          - "capture_date": "YYYY-MM-DDTHH:MM:SS.sssZ" (UTC)
          - "meanSatAz": degrees
          - "meanSunEl": degrees
          - "meanSunAz": degrees
          - Optional: "target_alt_km" (float, default 0.0)
    rsr_csv_path : str
        Path to CSV with (wavelength_um, response) for the target band (WV3 PAN or MS).
    atmos_source : str
        "cams" | "aeronet" | "manual"
    manual_atmos : dict or None
        Required if atmos_source="manual": {"aot550", "h2o_cm", "ozone_atm_cm"}

    Returns
    -------
    (reflectance_float32, formula_string)
    """
    # ---- Parse required inputs ----
    try:
        gain = float(params["gain"])
        offset = float(params["offset"])
        abs_factor = float(params["abs_factor"])
        effective_bwidth = float(params["effective_band_width"])
        _ = float(params["lowerBandEdge"])
        _ = float(params["upperBandEdge"])

        bbox = ctx["bbox"]
        capture_date_txt = ctx["capture_date"]
        mean_sat_az = float(ctx["meanSatAz"])
        mean_sun_el = float(ctx["meanSunEl"])
        mean_sun_az = float(ctx["meanSunAz"])
        target_alt_km = float(ctx.get("target_alt_km", 0.0))
    except KeyError as e:
        raise ValueError(f"Missing required parameter: {e}") from e

    # ---- 1) DN -> TOA radiance ----
    band_dn = band_dn.astype("float32", copy=False)
    L_toa = gain * band_dn * (abs_factor / effective_bwidth) + offset

    # ---- 2) Atmospheric parameters (AERONET/CAMS/manual) ----
    capture_dt = datetime.strptime(capture_date_txt, "%Y-%m-%dT%H:%M:%S.%fZ")
    aot550, h2o_cm, ozone_atm_cm = get_atmo_params(
        bbox=bbox,
        capture_dt=capture_dt,
        source=atmos_source,
        manual=manual_atmos,
    )

    # ---- 3) Build & run 6S with RSR ----
    s = SixS()
    s.geometry = Geometry.User()

    # Geometry (Py6S expects DEGREES)
    s.geometry.month = capture_dt.month
    s.geometry.day = capture_dt.day
    s.geometry.view_z = 0.0                     # near-nadir; override if you know off-nadir
    s.geometry.view_a = mean_sat_az
    s.geometry.solar_z = float(90.0 - mean_sun_el)
    s.geometry.solar_a = mean_sun_az

    # Atmosphere
    s.atmos_profile = AtmosProfile.UserWaterAndOzone(h2o_cm, ozone_atm_cm)
    s.aero_profile = AeroProfile.Continental
    s.aot550 = aot550

    # Altitudes
    s.altitudes.set_sensor_satellite_level()  # sensor above atmosphere (safe default)
    s.altitudes.set_target_custom_altitude(target_alt_km)

    # Wavelength via RSR (band-accurate)
    wv_um, rsp = load_rsr_csv(rsr_csv_path)
    s.wavelength = rsr_to_py6s_wavelength(wv_um, rsp)

    # Run 6S
    s.run()

    # ---- 4) Invert 6S forward model to get surface reflectance ----
    Lp = s.outputs.atmospheric_intrinsic_radiance
    Edir = s.outputs.direct_solar_irradiance
    Edif = s.outputs.diffuse_solar_irradiance

    trans = s.outputs.trans
    try:
        T_down = trans['total'].downward
        T_up   = trans['total'].upward
    except KeyError:
        T_down = trans['global_gas'].downward * trans['total_scattering'].downward
        T_up   = trans['global_gas'].upward   * trans['total_scattering'].upward

    numerator = math.pi * (L_toa - Lp)
    irradiance_term = (Edir * T_down) + Edif
    denominator = T_up * irradiance_term

    str_formula = (
        f"rho = pi*(L_toa - {Lp}) / ({T_up} * ({Edir}*{T_down} + {Edif}))"
    )

    if np.isscalar(denominator):
        if denominator <= 0:
            rho = np.zeros_like(L_toa, dtype="float32")
            return rho, str_formula
    else:
        denominator = np.where(denominator > 0, denominator, np.nan)

    rho = numerator / denominator

    # ---- 5) Clean-up: negatives & NaNs ----
    if _USE_NUMEXPR:
        rho = ne.evaluate("where(rho > 0, rho, 0)")
        rho = ne.evaluate("where(isnan(rho), 0, rho)")
    else:
        rho = np.where(rho > 0, rho, 0.0)
        rho = np.nan_to_num(rho, nan=0.0, posinf=0.0, neginf=0.0)

    return rho.astype("float32", copy=False), str_formula
