from pathlib import Path
import sys, subprocess
# from osgeo_utils.gdal_pansharpen import main as gdal_pansharpen

# Globar dir where WV3 images are looking for
ROOT = r"I:\2025_Nuevas_compras_imagenes\Kampir_Tepe\Kampir_WV3"
IMG_NAME = "050305158020_01"
images_dir = Path(ROOT, IMG_NAME, IMG_NAME + "_BOA")

# Search valid images
MUL = [i for i in images_dir.glob("*-M2AS-*_DOS3.tif")]
PAN = [i for i in images_dir.glob("*-P2AS-*_DOS3.tif")]

assert len(MUL) == 1, "No MUL image found"
assert len(PAN) == 1, "No PAN image found"

OUT = str(MUL[0]).replace(".tif", "_wBrovey.tif")

weights_wv3 = [0.005, 0.142, 0.209, 0.144, 0.234, 0.157, 0.116]
# Transform weights to introduce in gdal tool
bands = []
weights = []
for i, w in enumerate(weights_wv3, start=1):
    bands.extend(["-b", str(i)])
    weights.extend(["-w", str(w)])

inpts = [f'{str(MUL[0])},band={i}' for i in range(1, len(weights_wv3)+1)]

args = [
    *bands,
    *weights,
    "-of", "COG",
    "-r", "cubic",
    "-co", "COMPRESS=DEFLATE",
    "-co", "PREDICTOR=2",
    "-co", "NUM_THREADS=ALL_CPUS",
    "-co", "BIGTIFF=IF_SAFER",
    str(PAN[0]), *inpts, OUT
]
print("ARGS: gdal_pansharpen", (" ").join(args))
# run via module (avoids PATH issues)
# subprocess.run([sys.executable, "-m", "osgeo_utils.gdal_pansharpen", *args])
# gdal_pansharpen(args)