from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from typing import Union, Callable, Sequence, Optional
import numpy as np
import json
import math

@dataclass
class BandParams:
    """
    Required parameters to perform the available atmospheric corrections.

    This class is band dependent.

    Attributes
    ----------
    name : str
        Band descriptor name or band key.
    gain : float
        Radiometric GAIN coefficient (perform ARC formula).
    offset : float
        Radiometric OFFSET coefficient (perform ARC formula).
    abs_factor : float
        Absolute calibrated factor (sensor radiometric term)
    effective_band_width : float
        Sensor current effective band width
    e_thuillier : float
        ESUN term (at 1 AU; same units as radiance)
    lamb_scatter : float
        Scale for DN (digital number) of dark object. Haze DN (empirical)
    dark_object_dn : float
        DN at which haze is estimated
    lower_band_edge : float
        Spectral minimum edge of the band (micrometers)
    upper_band_edge : float
        Spectral maximum edge of the band (micrometers)
    """
    name: Union[str, int]
    gain: float
    offset: float
    e_thuillier: float
    lamb_scatter: float
    dark_object_dn: float
    lower_band_edge: float
    upper_band_edge: float
    abs_factor: Optional[float] = None
    effective_band_width: Optional[float] = None

@dataclass
class CtxParams:
    """
    Required parameters describing the whole image.

    d : float
        Earth-Sun distance (AU units).
    sol_zen_ang : float
        Solar zenith angle (degrees).
    mean_sat_az: float
        Mean satellite azimuth (degrees).
    mean_sun_az: float
        Mean solar azimuth (degrees).
    bbox: list['ULX','ULY','LRX','LRY']
        Image bounding box in degrees (EPSG:4326)
    """
    d: float
    sol_zen_ang: float
    mean_sat_az: float
    mean_sun_az: float
    bbox: Sequence[float]

@dataclass
class SensorParams:
    """
    Define parameters to perform atmospheric corrections by sensor.
    
    cal_func : callable
        Function to perform radiometric calibration.
    """
    name: str
    bands: Sequence[BandParams]
    ctx: CtxParams
    cal_func: Optional[Callable[..., np.ndarray]] = None

def write_log(params: BandParams, formula, band_id, log_file):
    """Store the logs of the computed image."""
    # Unique file name per run
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")

    with open(log_file, "a") as f:
        f.write(f"[{ts}] B{band_id} - PARAMS: {json.dumps(asdict(params))}\n")
        f.write(f"[{ts}] B{band_id} - FORM: {json.dumps(formula)}\n")

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
    Approximate Earth-Sun distance in astronomical units for a UTC instant.

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