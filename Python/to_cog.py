"""
Transform to COG several TIF files. Run the script in OSGeo4Web cmd.
"""
from pathlib import Path
import subprocess

# Folder with the files to transform
ROOT = r"I:\2025_Nuevas_compras_imagenes\Kampir_Tepe\Kampir_WV3\050305158010_01\PCA_combinations\LMNV"
# Prefix to search the files
pref = "_temp.tif"

# Create the command template
cmd = "gdal_translate {ipt} {opt} -of COG -co COMPRESS=DEFLATE -co PREDICTOR=3 -co BIGTIFF=IF_SAFER -co NUM_THREADS=ALL_CPUS -co OVERVIEWS=AUTO"

# Run
for f in Path(ROOT).glob(f"*{pref}"):
    out_file = str(f).replace(pref, ".tif")
    print(cmd.format(ipt=f, opt=out_file))
    subprocess.run(cmd.format(ipt=f, opt=out_file))
    f.unlink()