"""____________________________________________________________________________
Script Name:        02_pca.py
Description:        Compute PCA with all band's combinations.
Requirements:       GDAL version >= "3.1.4", Dask, numpy
Outputs:            Image with PCs as bands.
____________________________________________________________________________"""
from pathlib import Path
from utils_pca import PCA

IMGS_DIR = Path(r"I:\2025_Nuevas_compras_imagenes\Kampir_Tepe\Kampir_WV3")
img_id = "050305158010_01"
img_folder = Path(IMGS_DIR / img_id / f"{img_id}_BOA")

# Extract image to compute PCA
img_path_nir2 = [i for i in img_folder.glob("*HIGHRES_stack.vrt")]
img_path = [i for i in img_folder.glob("*wBrovey.tif")]
if len(img_path) != 1:
    raise ValueError(f"There is no a valid Brovey image inside {img_folder}")
if len(img_path_nir2) != 1:
    raise ValueError(f"There is no a valid stacked image inside {img_folder}")

# There are two PCA subdirs, one to store the NIR2 images obtained with LMNV
# method, and another one for the wBrovey pansharpening method.
OUT_DIR_NIR2 = Path(IMGS_DIR / img_id / "PCA_combinations", "LMNV")
OUT_DIR = Path(IMGS_DIR / img_id / "PCA_combinations")
# Create the directory (and all parents) if it doesn't exist
OUT_DIR_NIR2.mkdir(parents=True, exist_ok=True)

band_pos = [1, 2, 3, 4, 5, 6, 7, 8]
band_keys = ["C", "B", "G", "Y", "R", "RE1", "N", "N2"]

combis_dict = PCA.combis(
    band_keys, band_pos, Path(OUT_DIR, 'wv3_band_combinations.csv'))

for i, item in combis_dict.items():
    if "N2" in item["names"]:
        out_path = Path(
            OUT_DIR_NIR2,
            img_path[0].name.replace("wBrovey.tif", f"COMB{i}_temp.tif"))
        
        if out_path.exists():
            continue
        
        PCA.run(
            img_path_nir2[0], out_path, item["pos"], normalize=True, ram=2048)
    else:
        out_path = Path(
            OUT_DIR,
            img_path[0].name.replace("wBrovey.tif", f"_COMB{i}_temp.tif"))
        
        if out_path.exists():
            continue
        
        PCA.run(img_path[0], out_path, item["pos"], normalize=True, ram=2048)
