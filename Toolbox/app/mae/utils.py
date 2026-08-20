from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torch.nn.functional as F
import timm
import fiona
import rasterio
from affine import Affine
from rasterio.windows import Window
from rasterio.mask import mask
from rasterio.warp import transform_geom

DEFAULT_TILE_OVERLAP = 128
DEFAULT_HANN_MIN_WEIGHT = 1e-3
GTIFF_BLOCK_SIZE = 256


def _set_tiled_gtiff_profile(profile: dict, predictor: int) -> None:
    """Set valid tiled-GeoTIFF options without inherited block dimensions."""
    profile.pop("blockxsize", None)
    profile.pop("blockysize", None)
    profile.update(
        driver="GTiff",
        tiled=True,
        blockxsize=GTIFF_BLOCK_SIZE,
        blockysize=GTIFF_BLOCK_SIZE,
        compress="deflate",
        predictor=predictor,
        bigtiff="if_safer",
    )


def parse_mae_bands(value: str) -> tuple[int, int, int]:
    """Parse one to three one-based source bands into three model channels."""
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError(
            "Bands must contain one to three comma-separated integers."
        )
    if len(parts) > 3:
        raise ValueError("A maximum of three bands can be selected.")

    try:
        bands = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("Band numbers must be integers.") from exc

    if any(band < 1 for band in bands):
        raise ValueError("Band numbers must be positive and one-based.")

    if len(bands) == 1:
        bands *= 3
    elif len(bands) == 2:
        bands.append(bands[0])

    return tuple(bands)


def validate_source_bands(
    img_path: Path,
    bands: tuple[int, int, int],
) -> None:
    """Ensure that every selected one-based band exists in a raster."""
    with rasterio.open(img_path) as src:
        band_count = src.count

    unavailable = sorted({band for band in bands if band > band_count})
    if unavailable:
        listed = ", ".join(map(str, unavailable))
        raise ValueError(
            f"{Path(img_path).name} has {band_count} band(s), but the "
            f"selection requests unavailable band(s): {listed}."
        )


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    """Return tile origins that cover an axis with the requested overlap."""
    if tile_size < 1:
        raise ValueError("Tile dimensions must be positive.")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError(
            "Tile overlap must be non-negative and smaller than the tile size."
        )
    if length <= tile_size:
        return [0]

    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def raised_hann_window(
    height: int,
    width: int,
    minimum_weight: float = DEFAULT_HANN_MIN_WEIGHT,
) -> np.ndarray:
    """Create a separable 2D Hann window with a nonzero weight floor.

    The floor makes normalization stable at raster margins and in any area
    covered by only one prediction. The returned window is normalized to a
    maximum weight of one.
    """
    if height < 1 or width < 1:
        raise ValueError("Hann-window dimensions must be positive.")
    if not 0 < minimum_weight <= 1:
        raise ValueError("minimum_weight must be greater than 0 and at most 1.")

    row_window = np.hanning(height) if height > 1 else np.ones(1)
    col_window = np.hanning(width) if width > 1 else np.ones(1)
    weights = np.outer(row_window, col_window)
    maximum = weights.max()
    if maximum > 0:
        weights /= maximum
    weights = minimum_weight + (1.0 - minimum_weight) * weights
    return weights.astype(np.float32)


def tilling(
    img_path: Path,
    target_width: int = 512,
    target_height: int = 512,
    overlap: int = DEFAULT_TILE_OVERLAP,
    compression: bool = False
):
    """Split a raster into overlapping tiles with full source coverage."""
    if overlap >= min(target_width, target_height):
        raise ValueError(
            "Tile overlap must be smaller than both tile dimensions."
        )

    # Create a temporal folder to store the chunks
    tiles_temp_folder = img_path.parent / (img_path.stem + "_temp_tiles")
    tiles_temp_folder.mkdir(exist_ok=True)

    # Store the tiles metadata in a dictionary
    tiles_meta = {}

    with rasterio.open(img_path) as src:
        img_width = src.width
        img_height = src.height

        base_profile = src.profile.copy()
        base_tags = src.tags()
        
        # Per-band nodata (rasterio has .nodata and .nodatavals);
        # keep per-band where possible
        nodatavals = src.nodatavals

        tile_id = 1
        y_starts = _tile_starts(img_height, target_height, overlap)
        x_starts = _tile_starts(img_width, target_width, overlap)
        for y in y_starts:
            for x in x_starts:
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
                    _set_tiled_gtiff_profile(tile_profile, predictor=2)

                tiles_meta[tile_id] = {
                    "y": y,
                    "x": x,
                    "width": w,
                    "height": h,
                    "overlap": overlap,
                    "path": tile_path,
                }
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


def compute_mae(
    img_path,
    model,
    device: str,
    bands: tuple[int, int, int] = (1, 1, 1),
):
    """
    Note: This function does not a MAE training / reconstruction. It's just
    MAE-pretrained weights used for feature maps.

    Caveat: MAE-pretrained weights were learned on natural RGB images.
    For other type of images, the features might still be useful, but the
    "pretraining prior" isn't perfectly matched.
    """
    validate_source_bands(img_path, bands)
    with rasterio.open(img_path) as src:
        img = src.read(list(bands))  # (3, H, W), in the selected order
        meta = src.meta.copy()
        nodata = src.nodata
    
    # Export the heatmap and its metadata to save it as GeoTIFF later
    out_meta = meta.copy()
    out_meta.update({"count": 1, "dtype": "float32"})

    is_all_nodata = np.all(img == nodata)
    if is_all_nodata:
        return img[0], out_meta

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


def define_model(
    required_bands: int = 3,
    model_name: str = "vit_base_patch16_224.mae",
    checkpoint_path: str | Path | None = None,
):
    """Load the MAE model from a local checkpoint or the remote default."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = Path(checkpoint_path).resolve() if checkpoint_path else None
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"MAE checkpoint does not exist: {checkpoint}")

    pretrained_overlay = (
        {"file": str(checkpoint)} if checkpoint is not None else None
    )
    model = timm.create_model(
        model_name,
        pretrained=True,
        pretrained_cfg_overlay=pretrained_overlay,
        # Note: always three bands (or the first three bands).
        in_chans=required_bands
    ).to(device).eval()

    return model, device


def _model_source(model) -> str:
    """Return a concise, reproducible identifier for the loaded weights."""
    config = getattr(model, "pretrained_cfg", {}) or {}
    local_file = config.get("file")
    if local_file:
        return Path(local_file).name
    return config.get("hf_hub_id") or config.get("url") or "pretrained_default"


def compute_mae_batch(
    tiles_meta: dict,
    out_path: str,
    model,
    device: str,
    bands: tuple[int, int, int] = (1, 1, 1),
    hann_min_weight: float = DEFAULT_HANN_MIN_WEIGHT,
    out_suffix = "_mae.tif"
):
    """
    Apply MAE ViT to all the image tiles.
    
    :required_bands: Number of bands inside target images.
    """
    if not tiles_meta:
        raise ValueError("No source tiles were provided for MAE inference.")

    model_source = _model_source(model)

    for tile_id, props in tiles_meta.items():

        tile_path = Path(props["path"])
        
        out_name = tile_path.stem + out_suffix
        out_tile_path = tile_path.parent / out_name
        
        selected_bands = ",".join(map(str, bands))
        if out_tile_path.exists():
            with rasterio.open(out_tile_path) as existing:
                if existing.tags().get("MAE_SOURCE_BANDS") == selected_bands:
                    continue

        heatmap, out_meta = compute_mae(
            tile_path,
            model,
            device,
            bands=bands,
        )

        with rasterio.open(out_tile_path, "w", **out_meta) as dst:
            dst.write(heatmap, 1)
            dst.update_tags(
                MAE_SOURCE_BANDS=selected_bands,
                MAE_BAND_INDEXING="one-based",
                MAE_MODEL_SOURCE=model_source,
            )

    merge_mae_tiles(
        tiles_meta,
        out_suffix,
        out_path,
        minimum_weight=hann_min_weight,
    )
    with rasterio.open(out_path, "r+") as dst:
        overlap = next(iter(tiles_meta.values())).get("overlap", 0)
        dst.update_tags(
            MAE_SOURCE_BANDS=",".join(map(str, bands)),
            MAE_BAND_INDEXING="one-based",
            MAE_TILE_OVERLAP=str(overlap),
            MAE_BLEND_WINDOW="raised_hann",
            MAE_HANN_MIN_WEIGHT=f"{hann_min_weight:g}",
            MAE_MODEL_SOURCE=model_source,
        )


def merge_mae_tiles(
    tiles_meta: dict,
    mae_suffix: str,
    out_path: str,
    minimum_weight: float = DEFAULT_HANN_MIN_WEIGHT,
) -> None:
    """Blend overlapping MAE predictions using raised-Hann weights.

    For every output pixel this computes ``sum(S_i * w_i) / sum(w_i)``.
    Invalid prediction pixels do not contribute to either sum.
    """
    if not tiles_meta:
        raise ValueError("No MAE tiles were provided for merging.")

    tile_records = []
    for props in tiles_meta.values():
        tile_path = Path(props["path"])
        prediction_path = tile_path.with_name(tile_path.stem + mae_suffix)
        if not prediction_path.exists():
            raise FileNotFoundError(
                f"MAE prediction tile does not exist: {prediction_path}"
            )

        x = int(props["x"])
        y = int(props["y"])
        with rasterio.open(prediction_path) as src:
            height, width = src.height, src.width
            if not tile_records:
                out_meta = src.profile.copy()
                source_transform = src.transform * Affine.translation(-x, -y)
                source_nodata = src.nodata

        tile_records.append((prediction_path, x, y, width, height))

    output_width = max(x + width for _, x, _, width, _ in tile_records)
    output_height = max(y + height for _, _, y, _, height in tile_records)
    weighted_sum = np.zeros((output_height, output_width), dtype=np.float64)
    weight_sum = np.zeros((output_height, output_width), dtype=np.float64)

    for prediction_path, x, y, width, height in tile_records:
        with rasterio.open(prediction_path) as src:
            prediction = src.read(1, masked=True)

        weights = raised_hann_window(height, width, minimum_weight)
        values = prediction.filled(0).astype(np.float64, copy=False)
        valid = ~np.ma.getmaskarray(prediction) & np.isfinite(values)
        valid_weights = weights * valid
        rows = slice(y, y + height)
        cols = slice(x, x + width)
        weighted_sum[rows, cols] += values * valid_weights
        weight_sum[rows, cols] += valid_weights

    nodata = source_nodata if source_nodata is not None else np.nan
    mosaic = np.full((output_height, output_width), nodata, dtype=np.float32)
    covered = weight_sum > 0
    mosaic[covered] = (
        weighted_sum[covered] / weight_sum[covered]
    ).astype(np.float32)

    out_meta.update(
        count=1,
        dtype="float32",
        nodata=nodata,
        height=output_height,
        width=output_width,
        transform=source_transform,
    )
    _set_tiled_gtiff_profile(out_meta, predictor=3)
    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(mosaic, 1)


def clip_mae(inp_path: Path, clip_layer_path: Path) -> None:
    """Clip a raster in place using a vector cutline."""
    temp_path = inp_path.with_stem(
        f"{inp_path.stem}_temp_{uuid4().hex}"
    )

    try:
        with rasterio.open(inp_path) as src:
            source_tags = src.tags()
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
                dst.update_tags(**source_tags)

        temp_path.replace(inp_path)

    finally:
        temp_path.unlink(missing_ok=True)
