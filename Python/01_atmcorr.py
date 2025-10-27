from osgeo import gdal
from pathlib import Path

from utils_wv3 import WV3
from utils_atm import OutputSpec
from utils_atm import correct_raster_streaming
gdal.UseExceptions()

ROOT = Path(r"I:\2025_Nuevas_compras_imagenes\Kampir_Tepe\Kampir_WV3")
img_id = "050305158020_01"
image = WV3(Path(ROOT, img_id))

# Use this to clip the corrected image to a specific roi
CLIP = r"I:\ArqPy-processing-iranzo\data\rois\aoi_kampyr_tepe.kml" # ROI path
# Create output dir
boa_folder = Path(ROOT, img_id, img_id + "_BOA")
boa_folder.mkdir(exist_ok=True)

# Perform atmospheric correction
for img_type, img_path in image.get_images().items():

    init_message = f"Correcting {img_path.stem}"
    print(init_message)
    print("-"*len(init_message))
    
    # if img_type == "mul":
    #     continue

    if len(CLIP) > 0:
        out_clip_path = Path(boa_folder, img_path.stem + "_temp.tif")
        gdal.Warp(
            out_clip_path,              # dst
            img_path,                   # src
            cutlineDSName=CLIP,
            cropToCutline=True,
            dstNodata=0,
            multithread=True,
            creationOptions=["TILED=YES","COMPRESS=DEFLATE","PREDICTOR=2"],
            warpOptions=["NUM_THREADS=ALL_CPUS"]
        )

    for atm in ["DOS3"]:

        if img_type == "mul" and atm == "DOS3":
            # Band to compute the min DN
            image.START_BAND = "C"
        elif img_type == "pan" and atm == "DOS3":
            image.START_BAND = "P"

        band_params = image.extract_params_per_band(img_path)
        ctx_params = image.extract_ctx_params(img_path)

        # Create output names
        outname = img_path.stem + f"_{atm}.tif"
        out_path = Path(boa_folder, img_path.stem + f"_{atm}.tif")

        if len(CLIP) > 0:
            img_path_init = img_path
            img_path = out_clip_path

        band_formulas = [atm] * len(band_params)
        correct_raster_streaming(
            infile=img_path,
            outfile=out_path,
            band_formulas=band_formulas,
            band_params=band_params,
            ctx_params=ctx_params,
            out=OutputSpec(
                dtype="float32",
                nodata=0,
                scale=None,
                compress="DEFLATE",
                predictor=2,
                blocksize=512,
                tiled=True,
                cog=True),
            build_overviews=True,
            scale_to_int=True
        )

        if len(CLIP) > 0:
            # Remove temp image
            out_clip_path.unlink()
            img_path = img_path_init