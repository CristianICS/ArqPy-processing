"""
Prepare the data to inspect the crop marks with MAE saliency values.
"""
from pathlib import Path

import pandas as pd


PRINT_FIRST_TABLE = True

ROOT = Path(__file__).resolve().parent.parent

MAE_values_path = ROOT / "results" / "MAE_extracted_values_cropmarks.csv"

mae_extracted_values = pd.read_csv(MAE_values_path)

# Directory containing the global MAE stats per derived product
IMG_PATH = Path(r"C:\IRANZO\uzbekistan\zar_tepe\Ortoimagenes\Satelite\WV3\015105921010_01")
indices_mae_path = IMG_PATH / "SPECTRAL_INDICES/mae_saliency/mae_stats.csv"
pca_mae_path = IMG_PATH / "PCA/mae_saliency/mae_stats.csv"
highpass_mae_path = IMG_PATH / "HIGHPASS/mae_saliency/mae_stats.csv"
pan_mae_path = IMG_PATH / "015105921010_01_BOA/mae_saliency/mae_stats.csv"

indices_mae = pd.read_csv(indices_mae_path)
pca_mae = pd.read_csv(pca_mae_path)
highpass_mae = pd.read_csv(highpass_mae_path)
pan_mae = pd.read_csv(pan_mae_path)

# Table: Mean and maximum MAE saliency values by derived image product group
# mean,median,min,p90,p95,max,std
indices_best = indices_mae.loc[indices_mae["mean"].idxmax(), "id"]
pca_best = pca_mae.loc[pca_mae["mean"].idxmax(), "id"]
highpass_best = highpass_mae.loc[highpass_mae["mean"].idxmax(), "id"]
pan_best = pan_mae.loc[pan_mae["mean"].idxmax(), "id"]

if PRINT_FIRST_TABLE:
    print("PCA mean saliency", pca_mae["mean"].mean())
    print("PCA max saliency", pca_mae["max"].max())
    print("Best PCA combination", pca_best)

    print("Spectral Indices mean saliency", indices_mae["mean"].mean())
    print("Spectral Indices max saliency", indices_mae["max"].max())
    print("Best spectral index", indices_best)

    print("Highpass mean saliency", highpass_mae["mean"].mean())
    print("Highpass max saliency", highpass_mae["max"].max())
    print("Best highpass product", highpass_best)

    print("PAN mean saliency", pan_mae["mean"].mean())
    print("PAN max saliency", pan_mae["max"].max())
    print("Best PAN product", pan_best)

# Table: Saliency values and percentile ranks grouped by derived image product
# and true or false cropmark. 
# id,image_name,MAE_value,pixel_count,cropmark_median,cropmark_min,cropmark_max,cropmark_std,percentile_rank,min_max_rank,layer_valid_pixels,layer_mean,layer_median,layer_min,layer_p90,layer_p95,layer_max,layer_std,stats_source,cropmark,replant_2024,is_detected_with_natural_color,comments,derived_product_name,replant_2025

product_group = mae_extracted_values["stats_source"].str.split("\\").str[-3]
mae_extracted_values["product_id"] = product_group

group_cols = ["product_id", "cropmark"]
value_cols = ["cropmark_median", "percentile_rank"]
print(mae_extracted_values.groupby(group_cols)[value_cols].mean())

# Table: which product has the best saliency value in cropmarks, and what value
# it has in false cropmarks
# Mean percentile rank for each image within product/crop group
group_cols = ["product_id", "cropmark", "image_name"]
image_means = (
    mae_extracted_values.groupby(group_cols, as_index=False)
      ["percentile_rank"]
      .mean()
)

# Image with highest mean percentile rank for each product/crop group
best_images = image_means.loc[
    image_means.groupby(group_cols[0:2])["percentile_rank"].idxmax()
]

print(best_images.to_string(index=False))