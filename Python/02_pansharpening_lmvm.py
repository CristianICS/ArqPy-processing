from pathlib import Path
import os

# Brute-force Orfeo ToolBox
OTB = r"C:\OTB-9.1.1-Win64"
py  = os.path.join(OTB, "lib", "otb", "python")
bin = os.path.join(OTB, "bin")
lib = os.path.join(OTB, "lib")

# Help the loader (even if PATH is correct)
os.add_dll_directory(bin)
os.add_dll_directory(lib)

import otbApplication as otb  # type: ignore

# Globar dir where WV3 images are looking for
ROOT = r"I:\2025_Nuevas_compras_imagenes\Kampir_Tepe\Kampir_WV3"
IMG_NAME = "050305158020_01"
images_dir = Path(ROOT, IMG_NAME, IMG_NAME + "_BOA")

# Search valid images
MUL = [i for i in images_dir.glob("*-M2AS-*_DOS3.tif")]
PAN = [i for i in images_dir.glob("*-P2AS-*_DOS3.tif")]

assert len(MUL) > 0, "No MUL image found"
assert len(PAN) > 0, "No PAN image found"

# -----------------------------------------------------------------------------
# 1) Extract ONLY band 8 (NIR2) from the MUL aligned on PAN grid
#    OTB bands are 1-based; WV3 VNIR band 8 = NIR2.
# -----------------------------------------------------------------------------
NIR2 = "C:/paula_uribe_processing_temp/WV3_NIR2.tif"

ext = otb.Registry.CreateApplication("ExtractROI")
ext.SetParameterString("in", str(MUL[0]))
# Select channel 8 (NIR2). Must be a list of strings.
ext.SetParameterStringList("cl", ["Channel8"])
# NOTE: The gdal:co approach is now failing.
# ext.SetParameterString("out", NIR2_ON_PAN + co)
ext.SetParameterString("out", NIR2)
# ext.SetParameterOutputImagePixelType("out", otb.ImagePixelType_float)
ext.SetParameterOutputImagePixelType("out", otb.ImagePixelType_int16)
ext.ExecuteAndWriteOutput()

# -----------------------------------------------------------------------------
# 1) Superimpose (warp) the MUL to the PAN grid (keeps reflectance as float32)
# -----------------------------------------------------------------------------
NIR2_ON_PAN = "C:/paula_uribe_processing_temp/WV3_NIR2_onPAN.tif"

sup = otb.Registry.CreateApplication("Superimpose")
sup.SetParameterString("inr", str(PAN[0])) # Reference: PAN grid
sup.SetParameterString("inm", NIR2)   # Moving: MUL (float)
sup.SetParameterString("interpolator", "linear")  # linear is safe for radiometry

# GeoTIFF creation options (can be toggled on if you want compression)
co = ("?gdal:co:COMPRESS=DEFLATE"
      "&gdal:co:TILED=YES"
      "&gdal:co:PREDICTOR=3"
      "&gdal:co:BIGTIFF=IF_SAFER")
# If you want them active, uncomment below:
# sup.SetParameterString("out", NIR2_ON_PAN + co)
sup.SetParameterString("out", NIR2_ON_PAN)
# sup.SetParameterOutputImagePixelType("out", otb.ImagePixelType_float)
ext.SetParameterOutputImagePixelType("out", otb.ImagePixelType_int16)
sup.ExecuteAndWriteOutput()

# -----------------------------------------------------------------------------
# 3) Pansharpen ONLY the NIR2 band using LMVM (local mean/variance with HPF PAN)
# -----------------------------------------------------------------------------
app = otb.Registry.CreateApplication("Pansharpening")

# Inputs: Raw WV3 PAN and single-band NIR2 (co-registered)
app.SetParameterString("inp",  str(PAN[0]))   # panchromatic (0.31 m)
app.SetParameterString("inxs", NIR2_ON_PAN)   # single-band NIR2 (1.24 m -> res target)

# Method: LMVM (good for bands with weak spectral overlap to PAN, e.g., NIR2)
app.SetParameterString("method", "lmvm")

# Optional tuning (uncomment to adjust local window; defaults ~3x3)
# app.SetParameterInt("method.lmvm.radiusx", 5)
# app.SetParameterInt("method.lmvm.radiusy", 5)

# --- Output as float32 to preserve reflectance units
out_name = MUL[0].name.replace("DOS3.tif", "B08_NIR2_Plmvm.tif")
out = f"{images_dir / out_name}"
# If you want GDAL options:
out_with_co = (
    f"{images_dir / out_name}"
    # "?gdal:driver=COG"
    "&gdal:co:COMPRESS=DEFLATE"
    # "&gdal:co:CLOUD_OPTIMIZED=YES"
    # When creating COG, the file must be tiled.
    # Some GDAL versions auto-tile, yet it’s safest to be explicit:
    "&gdal:co:TILED=YES&gdal:co:BLOCKXSIZE=512&gdal:co:BLOCKYSIZE=512"
    "&gdal:co:PREDICTOR=3"
    # "&gdal:co:BIGTIFF=IF_SAFER"
)

app.SetParameterString("out", out)
# app.SetParameterOutputImagePixelType("out", otb.ImagePixelType_float)
ext.SetParameterOutputImagePixelType("out", otb.ImagePixelType_int16)

# Optionally increase RAM (MB)
app.SetParameterInt("ram", 2048)

app.ExecuteAndWriteOutput()

# Print the recommended GDAL command to convert the pansharpened band to COG
gdal_trnslate_args = [
    "gdal_translate",
    out,
    str(out).replace(".tif", "_cog.tif"),
    "-of COG -co COMPRESS=DEFLATE -co PREDICTOR=2",
    "-co BIGTIFF=IF_SAFER -co NUM_THREADS=ALL_CPUS -co OVERVIEWS=AUTO"
]
print("GDAL translate:", (" ").join(gdal_trnslate_args))
# Print the GDAL command to construct a VRT with the two pansharpening methods
gdal_vrt_args = [
    "gdalbuildvrt -separate -overwrite",
    str(MUL[0]).replace(".tif", "_HIGHRES_stack.vrt"),
    str(MUL[0]).replace(".tif", "_wBrovey.tif"),
    str(out)
]
print("GDAL buildvrt:", (" ").join(gdal_vrt_args))