from pathlib import Path
from uuid import uuid4
import csv
import numpy as np
import torch
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import timm
import fiona
import rasterio
from rasterio.merge import merge
from rasterio.windows import Window
from rasterio.mask import mask
from rasterio.warp import transform_geom

def tilling(
    img_path: Path,
    target_width: int = 512,
    target_height: int = 512,
    compression: bool = False
):
    """
    Note: If the image source has rotation/shear (non-north-up geotransform),
    the tile_gt math below still handles it correctly because it uses Window.
    """
    # Create a temporal folder to store the chunks
    tiles_temp_folder = img_path.parent / (img_path.stem + "_temp_tiles")
    tiles_temp_folder.mkdir(exist_ok=True)

    # Store the tiles metadata in a dictionary
    tiles_meta = {}

    # Optional GeoTIFF creation tweaks
    GTIFF_CREATION_KWARGS = {
        "driver": "GTiff",
        "tiled": True,
        "compress": "deflate",
        "predictor": 2,
        "BIGTIFF": "IF_SAFER"
    }

    with rasterio.open(img_path) as src:
        img_width = src.width
        img_height = src.height

        base_profile = src.profile.copy()
        base_tags = src.tags()
        
        # Per-band nodata (rasterio has .nodata and .nodatavals);
        # keep per-band where possible
        nodatavals = src.nodatavals

        tile_id = 1
        for y in range(0, img_height, target_height):
            for x in range(0, img_width, target_width):
                w = min(target_width, img_width - x)
                h = min(target_height, img_height - y)

                window = Window(col_off=x, row_off=y, width=w, height=h)

                # Read tile. Keep original band count; shape is (count, h, w)
                chunk = src.read(window=window)

                out_name = f"tile_{y}_{x}.tif"
                tile_path = tiles_temp_folder / out_name

                tile_profile = base_profile.copy()
                tile_profile.update(
                    {
                        "width": chunk.shape[2],
                        "height": chunk.shape[1],
                        "transform": src.window_transform(window),
                    }
                )
                if compression:
                    tile_profile.update({**GTIFF_CREATION_KWARGS})

                # If source uses internal tiling, rasterio/GDAL may require
                # block sizes not to exceed tile dimensions.
                # Make block sizes <= output dims.
                # if tile_profile.get("tiled", False):
                #     bsx = min(256, tile_profile["width"])
                #     bsy = min(256, tile_profile["height"])
                #     tile_profile.update({"blockxsize": bsx, "blockysize": bsy})

                tiles_meta[tile_id] = {"y": y, "x": x, "path": tile_path}
                tile_id += 1

                if tile_path.exists():
                    continue
                
                with rasterio.open(tile_path, "w", **tile_profile) as dst:
                    dst.write(chunk)

                    # Preserve dataset-level tags (optional but often useful)
                    if base_tags:
                        dst.update_tags(**base_tags)

                    # Preserve per-band nodata values where present
                    for bidx, nd in enumerate(nodatavals, start=1):
                        if nd is not None:
                            dst.update_tags(bidx, nodata=nd)  # metadata tag

                    # Preserve per-band tags too
                    for bidx in range(1, src.count + 1):
                        band_tags = src.tags(bidx)
                        if band_tags:
                            dst.update_tags(bidx, **band_tags)

    
    return tiles_meta, tiles_temp_folder


def vit_tokens(model, x):
    """
    Return (B, 1+N, D) tokens after transformer blocks.
    
    In timm's ViT, forward_features() can return either:
        - tokens (CLS + patches)
        - a pooled feature vector
    """
    x = model.patch_embed(x)  # (B, N, D)

    # Add cls + pos embed using timm helper if present
    if hasattr(model, "_pos_embed"):
        x = model._pos_embed(x)
    else:
        cls = model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + model.pos_embed
        x = model.pos_drop(x)

    if hasattr(model, "norm_pre") and model.norm_pre is not None:
        x = model.norm_pre(x)

    for blk in model.blocks:
        x = blk(x)

    x = model.norm(x)
    return x  # (B, 1+N, D)


def to_3ch(img):  # img: (C,H,W)
    C, H, W = img.shape
    if C == 3:
        return img
    if C == 1:
        return np.repeat(img, 3, axis=0)
    if C == 2:
        return np.concatenate([img, img[:1]], axis=0)  # add a copy of band0
    # C > 3
    return img[:3]


def compute_mae(img_path, model, device: str):
    """
    Note: This function does not a MAE training / reconstruction. It's just
    MAE-pretrained weights used for feature maps.

    Caveat: MAE-pretrained weights were learned on natural RGB images.
    For other type of images, the features might still be useful, but the
    "pretraining prior" isn't perfectly matched.
    """
    with rasterio.open(img_path) as src:
        img = src.read() # (C, H, W)
        meta = src.meta.copy()
        nodata = src.nodata
    
    # Export the heatmap and its metadata to save it as GeoTIFF later
    out_meta = meta.copy()
    out_meta.update({"count": 1, "dtype": "float32"})

    is_all_nodata = np.all(img == nodata)
    if is_all_nodata:
        return img[0], out_meta

    # Transform the image into a 3 channels one
    img = to_3ch(img)
    C, H, W = img.shape

    # Normalize to [0, 1] per band
    img = img.astype(np.float32)
    mins = img.reshape(C, -1).min(axis=1)[:, None, None]
    maxs = img.reshape(C, -1).max(axis=1)[:, None, None]
    img = (img - mins) / np.maximum(maxs - mins, 1e-6)
    
    # Convert to torch (B, C, H, W) and resize to 224
    # Note: B = batch
    x = torch.from_numpy(img).unsqueeze(0) # (1, C, H, W)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = x.to(device)
    
    with torch.no_grad():
        tokens = vit_tokens(model, x.float())  # (1, 1+N, D)
        # drop CLS, L2 norm per patch
        patch_tokens = tokens[:, 1:, :]  # (1, N, D)
        # Compute L2 norm for each patch
        norms = torch.norm(patch_tokens, dim=-1)  # (1, N)

    # infer patch grid from N (224 with patch16 => N=196 => 14x14)
    N = norms.shape[1]
    g = int(np.sqrt(N))
    if g * g != N:
        raise RuntimeError(
            f"Patch token count {N} is not a square; can't reshape to grid.")
    heat = norms.reshape(1, 1, g, g)
    
    # upsample to original raster size
    heat_up = F.interpolate(
        heat,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )
    heat_up = heat_up[0, 0].detach().cpu().numpy().astype(np.float32)

    return heat_up, out_meta

def update_csv(csv_path, file_name, mean_value, max_value):
    """
    Store mean MAE values by image.

    Add (img_name, embedding_norms_mean) to a CSV file.
    Creates the file if it does not exist.
    Avoids duplicates based on img_name.
    """
    fieldnames = ["img_name", "embedding_norms_mean", "embedding_norms_max"]
    new_row = {
        "img_name": file_name,
        "embedding_norms_mean": round(mean_value, 2),
        "embedding_norms_max": round(max_value, 2)
    }

    # If file does not exist, create it
    if not Path(csv_path).exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(new_row)
        return

    # If file exists, read and check
    rows = []
    img_names = set()

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)
            img_names.add(row["img_name"])

    # If file_name already exists, do nothing
    if file_name in img_names:
        return

    # Otherwise, append and rewrite file
    rows.append(new_row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def define_model(
    required_bands: int = 3,
    model_name = "vit_base_patch16_224.mae"
):
    """Retrieve target model once."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load MAE Vision Transformer (ViT) model by default
    model = timm.create_model(
        model_name,
        pretrained=True,
        # Note: always three bands (or the first three bands).
        in_chans=required_bands
    ).to(device).eval()

    return model, device


def compute_mae_batch(
    tiles_meta: dict,
    out_path: str,
    stats_csv_path,
    model,
    device: str,
    out_suffix = "_mae.tif"
):
    """
    Apply MAE ViT to all the image tiles.
    
    :required_bands: Number of bands inside target images.
    """
    for tile_id, props in tiles_meta.items():

        tile_path = Path(props["path"])
        
        out_name = tile_path.stem + out_suffix
        out_tile_path = tile_path.parent / out_name
        
        if out_tile_path.exists():
            continue

        heatmap, out_meta = compute_mae(tile_path, model, device)

        with rasterio.open(out_tile_path, "w", **out_meta) as dst:
            dst.write(heatmap, 1)

    merge_mae_tiles(tile_path.parent, out_suffix, out_path)
    mae_mean, mae_max = compute_mae_stats(out_path)
    update_csv(stats_csv_path, Path(out_path).stem, mae_mean, mae_max)


def compute_mae_stats(img_path):
    """
    Retrieve the Embedding Norms mean value

    Note: it avoids loading the whole raster into memory.
    """
    total_sum = 0.0
    total_count = 0
    max_value = None

    with rasterio.open(img_path) as src:
        nodata = src.nodata

        # Iterate over blocks of band 1
        for _, window in src.block_windows(1):
            data = src.read(1, window=window, masked=True)

            # Skip fully masked blocks
            if data.count() == 0:
                continue
            
            # Update sum and count
            total_sum += data.sum()
            total_count += data.count()

            # Update max
            block_max = data.max()

            if max_value is None or block_max > max_value:
                max_value = block_max

    mean_value = total_sum / total_count if total_count > 0 else None
    return mean_value, max_value

def merge_mae_tiles(folder, mae_suffix, out_path):
    suffix = f"*{mae_suffix}"
    inputs = sorted([*folder.glob(suffix), *folder.glob(suffix)])
    srcs = [rasterio.open(p) for p in inputs]
    try:
        mosaic, transform = merge(srcs)
        out_meta = srcs[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": transform,
                "compress": "deflate",
                "predictor": 2,
                "tiled": True,
                "bigtiff": "if_safer",
            }
        )

        with rasterio.open(out_path, "w", **out_meta) as dst:
            dst.write(mosaic)
    finally:
        for s in srcs:
            s.close()


def clip_mae(inp_path: Path, clip_layer_path: Path) -> None:
    """Clip a raster in place using a vector cutline."""
    temp_path = inp_path.with_stem(
        f"{inp_path.stem}_temp_{uuid4().hex}"
    )

    try:
        with rasterio.open(inp_path) as src:
            with fiona.open(clip_layer_path) as clip_layer:
                geometries = [
                    transform_geom(
                        clip_layer.crs,
                        src.crs,
                        feature["geometry"],
                    )
                    for feature in clip_layer
                ]

            clipped, transform = mask(
                src,
                geometries,
                crop=True,
            )

            profile = src.profile.copy()
            profile.update(
                driver="COG",
                height=clipped.shape[1],
                width=clipped.shape[2],
                transform=transform,
                compress="DEFLATE",
                predictor=3,
                BIGTIFF="IF_SAFER",
            )

            with rasterio.open(temp_path, "w", **profile) as dst:
                dst.write(clipped)

        temp_path.replace(inp_path)

    finally:
        temp_path.unlink(missing_ok=True)
