from pathlib import Path
from utils_indices import Indices

ROOT = Path(__file__).parent.parent

INDICES = Path(ROOT, 'data', 'indices.json')

IMGS_DIR = Path(r"I:\2025_Nuevas_compras_imagenes\Kampir_Tepe\Kampir_WV3")
img_id = "050305158010_01"
img_folder = Path(IMGS_DIR / img_id / f"{img_id}_BOA")
# Extract pansharpened image
img_path = [i for i in img_folder.glob("*wBrovey.tif")]
if len(img_path) != 1:
    raise ValueError(f"There is no a valid image inside {img_folder}")

OUT_DIR = Path(IMGS_DIR / img_id / "SPECTRAL_INDICES")
# Create the directory (and all parents) if it doesn't exist
OUT_DIR.mkdir(parents=True, exist_ok=True)

band_pos = [1, 2, 3, 4, 5, 6, 7, 8]
band_keys = ["C", "B", "G", "Y", "R", "RE1", "N", "N2"]
# NOTE: Band keys must be same as indices.json band keys.
# NOTE: Band keys must be ordered from first image's band to last band
# NOTE: Band keys must include all image bands.

for index_key in Indices.get_index_keys(INDICES):
    print(index_key)
    print('-' * 20)
    co = [
        "COMPRESS=DEFLATE",
        "PREDICTOR=2",
        "BIGTIFF=IF_SAFER",
        "NUM_THREADS=ALL_CPUS"
    ]

    try:
        Indices.compute_index(
            INDICES, index_key, img_path[0], band_pos, band_keys, OUT_DIR,
            creation_opts=co)
    except:
        continue