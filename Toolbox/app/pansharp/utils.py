from pathlib import Path
import subprocess

from core.env import find_otb_executable
from core.raster_io import RasterProfile, otb_creation_option_string, translate_to_cog

# OTB output creation-options profile, matching this tool's original
# COG-oriented GTiff settings (PREDICTOR=3, unlike the core default of 2).
_OTB_PROFILE = RasterProfile(
    compress="DEFLATE", predictor=3, tiled=True,
    blockxsize=512, blockysize=512, bigtiff="IF_SAFER",
)


def _otb_output_co() -> str:
    return otb_creation_option_string(
        _OTB_PROFILE, copy_src_overviews=True, num_threads=True,
    )


class Sensor:
    def __init__(self, name: str, weights: list):
        self.name = name
        self.weights = weights

    def compute_brovey_formula(self):
        """For Brovey transformation, convert weights into OTB format."""
        n = len(self.weights)

        # Denominator: sum_i w_i * im2b(i+1)
        den = "+".join(
            f"{self.weights[i]}*im2b{i+1}" for i in range(n)
        )

        # Construct the RCS operation by band
        # Each of this operations will be divided by the above den
        # Numerators: im1b1 * w_i * im2b(i+1) / (den)
        nums = [
            f"(im1b1*{self.weights[i]}*im2b{i+1}) / ({den})"
            for i in range(n)
        ]

        # Final expression: { num_1, num_2, ..., num_n }
        # NOTE: double braces for scaping, OTB requires the expression
        # to be inside braces
        expr = f"{{ {','.join(nums)} }}"
        return expr


class WV3(Sensor):

    def lmnv(self, mul_img_path: Path, pan_img_path: Path):

        otb_exe = find_otb_executable()
        otb_co = _otb_output_co()

        # Extract ONLY band 8 (NIR2) from the MUL aligned on PAN grid
        # OTB bands are 1-based; WV3 VNIR band 8 = NIR2.
        nir2_path = mul_img_path.parent / "WV3_NIR2_temp.tif"
        nir2_pan_path = mul_img_path.parent / "WV3_NIR_PAN_temp.tif"

        try:
            subprocess.run([
                otb_exe,
                "ExtractROI",
                "-in", str(mul_img_path),
                "-cl", "Channel8",
                "-out", f"{nir2_path}?&{otb_co}"
            ], check=True)

            # Superimpose (warp) the MUL to the PAN grid
            subprocess.run([
                otb_exe,
                "Superimpose",
                "-inr", str(pan_img_path),
                "-inm", str(nir2_path),
                "-interpolator", "linear",
                "-out", f"{nir2_pan_path}?&{otb_co}"
            ], check=True)

            # Pansharpen ONLY the NIR2 band using LMVM
            # (local mean/variance with HPF PAN)
            out_name = mul_img_path.name.replace("DOS3.tif", "B08_NIR2_Plmvm.tif")
            out_path = mul_img_path.parent / out_name
            subprocess.run([
                otb_exe,
                "Pansharpening",
                # Panchromatic (0.31 m)
                "-inp", str(pan_img_path),
                # Single-band NIR2 (1.24 m)
                "-inxs", str(nir2_pan_path),
                # Use defaults window: 3x3
                "-method", "lmvm",
                "-out", f"{out_path}?{otb_co}"
            ], check=True)
        finally:
            # Remove temporal files
            nir2_path.unlink(missing_ok=True)
            nir2_pan_path.unlink(missing_ok=True)

        return out_path

    def merge_pansharpened_bands(self, brovey_path, lmnv_path):
        """Create VRT with the bands from Brovey and LMNV."""
        out_name = lmnv_path.name.replace(
            "B08_NIR2_Plmvm.tif", "HIGHRES_stack.vrt")

        subprocess.check_call([
            "gdalbuildvrt",
            "-separate",
            str(lmnv_path.parent / out_name),
            str(brovey_path),
            str(lmnv_path)
        ])


WV3 = WV3("WV3", [0.005, 0.142, 0.209, 0.144, 0.234, 0.157, 0.116])
LEGION = Sensor("LEGION", [])

SENSORS = {"WV3": WV3, "LEGION": LEGION}


def brovey(
    sensor: Sensor,
    mul_img_path,
    pan_img_path,
    out_path
):
    """Run weighted Brovey pansharpen operation"""
    otb_exe = find_otb_executable()
    otb_co = _otb_output_co()

    # Align the MUL image with PAN dimensions
    mul_temp_img_name = mul_img_path.name.replace(".tif", "_temp.tif")
    mul_temp_path = mul_img_path.parent / mul_temp_img_name

    out_path_temp = out_path.parent / out_path.name.replace(
        ".tif", "_btemp.tif")

    try:
        subprocess.run([
            otb_exe,
            "Superimpose",
            "-inr", str(pan_img_path),
            "-inm", str(mul_img_path),
            "-interpolator", "linear",
            "-out", f"{mul_temp_path}?{otb_co}"
        ], check=True)

        # Output the OTB expression
        expr = sensor.compute_brovey_formula()

        # Export the image with OTB, then convert to COG with GDAL
        subprocess.run([
            otb_exe,
            "BandMathX",
            "-il", str(pan_img_path), str(mul_temp_path),
            "-out", f"{out_path_temp}?{otb_co}",
            "-exp", expr,
        ], check=True)

        # Translate to COG
        translate_to_cog(
            out_path_temp, out_path,
            RasterProfile(compress="DEFLATE", predictor=2, bigtiff="IF_SAFER"),
        )
    finally:
        # Remove temp files
        mul_temp_path.unlink(missing_ok=True)
        out_path_temp.unlink(missing_ok=True)


def bayesian(mul_img_path, pan_img_path, out_path, bayes_lambda=0.995):
    """
    OTB CLI command for Bayesian pansharpening using the Pansharpening application
        """
    otb_exe = find_otb_executable()
    otb_co = _otb_output_co()

    # Align the MUL image with PAN dimensions
    mul_temp_img_name = mul_img_path.name.replace(".tif", "_temp.tif")
    mul_temp_path = mul_img_path.parent / mul_temp_img_name

    # Compute Bayesian pansharpening
    out_temp_name = out_path.name.replace(".tif", "_btemp.tif")
    out_temp_path = out_path.parent / out_temp_name

    try:
        subprocess.run([
            otb_exe,
            "Superimpose",
            "-inr", str(pan_img_path),
            "-inm", str(mul_img_path),
            "-interpolator", "nn",
            "-out", f"{mul_temp_path}?{otb_co}", "float"
        ], check=True)

        subprocess.run([
            otb_exe,
            "Pansharpening",
            "-inp", str(pan_img_path),
            "-inxs", str(mul_temp_path),
            "-out", str(out_temp_path), "float",
            "-method", "bayes",
            "-method.bayes.lambda", str(bayes_lambda),
            "-method.bayes.s", "1"
        ], check=True)

        # Translate to COG
        translate_to_cog(
            out_temp_path, out_path,
            RasterProfile(compress="DEFLATE", predictor=2, bigtiff="IF_SAFER"),
            # Match the OTB fillnodata value
            nodata=-9999,
        )
    finally:
        # Remove temp files
        mul_temp_path.unlink(missing_ok=True)
        out_temp_path.unlink(missing_ok=True)
