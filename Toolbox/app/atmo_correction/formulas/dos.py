from typing import Optional, Callable

from atmo_correction.utils import BandParams, CtxParams

import numpy as np
import math

try:
    import numexpr as ne
    _USE_NUMEXPR = True
except Exception:
    _USE_NUMEXPR = False

def dark_object_subtraction_v3(
    band: np.ndarray,
    params: BandParams,
    ctx: CtxParams,
    radiometric_cal: Callable
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
    params : BandParams
        Object with the required parameters to apply correction formula.
    ctx : CtxParans
        Parameters to apply correction formula, yet not band-dependent.
    radiometric_cal : callable
        Formula to apply radiometric calibration. It is mandatory to convert
        haze levels into Radiometric units instead DN.

    Returns
    -------
    (np.ndarray, str)
        Tuple of (reflectance array float32, stringified formula used).
    """

    # Parse inputs (fail loudly if something is missing) ----
    try:
        e_thuillier = float(params.e_thuillier)
        scale_coef = float(params.lamb_scatter)
        L_haze_start_dn = float(params.dark_object_dn)
        min_wav = float(params.lower_band_edge)
        max_wav = float(params.upper_band_edge)

        d = ctx.d
        # Scientific Python math functions (e.g., math.cos) expect radians.
        sun_zen_ang = math.radians(float(ctx.sol_zen_ang))

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
    band = radiometric_cal(band, params)

    # ---- 3) Haze estimation ----
    # Start with a DN for the dark object, scale it (empirical), and
    # convert to radiance.
    L_haze_dn = L_haze_start_dn * scale_coef
    L_haze = radiometric_cal(L_haze_dn, params)  # radiance

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
    numerator = math.pi * (band - L_haze) * (d**2)
    denominator = e_thuillier * mu0 * T_z
    # Avoid division by zero
    if denominator <= 0:
        # If mu0 ~ 0 (SZA ~ 90) or T_z <= 0, reflectance isn't defined;
        # return zeros.
        refl = np.zeros_like(band, dtype="float32")
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