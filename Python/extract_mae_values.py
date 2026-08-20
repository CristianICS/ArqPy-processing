"""Extract comparable MAE saliency values along interpreted cropmarks.

The input cropmarks may be points, lines, or polygons. For each cropmark and
each ``*_mae.tif`` raster, the script summarizes every raster pixel touched by
the geometry. It also reports the cropmark mean as both an empirical percentile
rank and a min-max normalized rank within the MAE raster.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import fiona
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.warp import transform_geom


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CROPMARKS = ROOT / "data" / "interpreted_cropmarks.geojson"
DEFAULT_OUTPUT = ROOT / "results" / "MAE_extracted_values_interpreted_cropmarks.csv"
STATS_FIELDS = ("mean", "median", "min", "p90", "p95", "max", "std")


@dataclass(frozen=True)
class Cropmark:
    """A source feature retained with a stable output identifier."""

    id: Any
    geometry: dict[str, Any]
    properties: dict[str, Any]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract MAE saliency summaries for each interpreted cropmark. "
            "MAE_ROOT can be one *_mae.tif or a directory searched recursively."
        )
    )
    parser.add_argument(
        "mae_root",
        type=Path,
        help="MAE GeoTIFF or directory containing MAE GeoTIFFs.",
    )
    parser.add_argument(
        "--cropmarks",
        type=Path,
        default=DEFAULT_CROPMARKS,
        help=f"Cropmark vector (default: {DEFAULT_CROPMARKS}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--id-field",
        help=(
            "Optional cropmark attribute to use as id. By default features are "
            "numbered from 1 in source order."
        ),
    )
    parser.add_argument(
        "--band",
        type=int,
        default=1,
        help="One-based MAE raster band to sample (default: 1).",
    )
    parser.add_argument(
        "--center-only",
        action="store_true",
        help=(
            "Include only pixels whose centers intersect a geometry. The default "
            "uses all pixels touched by it, which is more suitable for thin lines."
        ),
    )
    return parser.parse_args(argv)


def discover_mae_rasters(path: Path) -> list[Path]:
    """Return MAE rasters in deterministic order."""
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = [item for item in path.rglob("*") if item.is_file()]
    else:
        raise FileNotFoundError(f"MAE path does not exist: {path}")

    rasters = sorted(
        (
            item
            for item in candidates
            if item.suffix.lower() in {".tif", ".tiff"}
            and item.stem.lower().endswith("_mae")
        ),
        key=lambda item: str(item).lower(),
    )
    if not rasters:
        raise FileNotFoundError(f"No *_mae.tif or *_mae.tiff rasters found in {path}")
    return rasters


def read_cropmarks(
    path: Path, id_field: str | None
) -> tuple[list[Cropmark], Any, list[str]]:
    """Load vector features and their CRS without requiring geopandas."""
    if not path.exists():
        raise FileNotFoundError(f"Cropmark vector does not exist: {path}")

    cropmarks: list[Cropmark] = []
    with fiona.open(path) as source:
        vector_crs = source.crs_wkt or source.crs
        if not vector_crs:
            raise ValueError(f"Cropmark vector has no CRS: {path}")
        property_names = list(source.schema.get("properties", {}))
        if id_field and id_field not in property_names:
            raise ValueError(
                f"ID field {id_field!r} is not present in {path}; "
                f"available fields: {', '.join(property_names)}"
            )

        for sequence_id, feature in enumerate(source, start=1):
            if feature.geometry is None:
                continue
            properties = dict(feature.properties)
            feature_id = properties[id_field] if id_field else sequence_id
            cropmarks.append(
                Cropmark(
                    id=feature_id,
                    geometry=dict(feature.geometry),
                    properties=properties,
                )
            )

    if not cropmarks:
        raise ValueError(f"Cropmark vector contains no features with geometry: {path}")
    return cropmarks, vector_crs, property_names


def valid_values(data: np.ma.MaskedArray) -> np.ndarray:
    """Flatten a masked raster read and discard non-finite values."""
    values = np.asarray(data.compressed())
    return values[np.isfinite(values)]


def compute_layer_stats(src: rasterio.io.DatasetReader, band: int) -> dict[str, float]:
    """Compute stats when a matching mae_stats.csv row is unavailable."""
    values = valid_values(src.read(band, masked=True))
    if values.size == 0:
        raise ValueError(f"Raster contains no valid values: {src.name}")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "std": float(values.std()),
    }


def read_layer_stats(
    raster_path: Path, src: rasterio.io.DatasetReader, band: int
) -> tuple[dict[str, float], str]:
    """Read the raster's row from adjacent mae_stats.csv, or compute it."""
    stats_path = raster_path.parent / "mae_stats.csv"
    if stats_path.exists():
        with stats_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            missing = {"id", *STATS_FIELDS}.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f"Missing columns in {stats_path}: {', '.join(sorted(missing))}"
                )
            for row in reader:
                if row["id"].strip() == raster_path.stem:
                    try:
                        return (
                            {name: float(row[name]) for name in STATS_FIELDS},
                            str(stats_path),
                        )
                    except (TypeError, ValueError) as error:
                        raise ValueError(
                            f"Invalid numeric stats for {raster_path.stem} in {stats_path}"
                        ) from error

    return compute_layer_stats(src, band), "computed_from_raster"


def extract_cropmark_values(
    src: rasterio.io.DatasetReader,
    cropmarks: Iterable[Cropmark],
    vector_crs: Any,
    band: int,
    all_touched: bool,
) -> list[tuple[Cropmark, np.ndarray]]:
    """Extract valid pixels intersected by each transformed geometry."""
    extracted: list[tuple[Cropmark, np.ndarray]] = []
    for cropmark in cropmarks:
        geometry = transform_geom(vector_crs, src.crs, cropmark.geometry)
        try:
            subset, _ = mask(
                src,
                [geometry],
                crop=True,
                filled=False,
                indexes=band,
                all_touched=all_touched,
            )
            values = valid_values(subset)
        except ValueError as error:
            if "overlap" not in str(error).lower():
                raise
            values = np.empty(0, dtype=np.float32)
        extracted.append((cropmark, values))
    return extracted


def percentile_ranks(
    src: rasterio.io.DatasetReader, band: int, targets: list[float]
) -> tuple[list[float], int]:
    """Calculate exact empirical CDF ranks in one block-wise raster pass."""
    target_array = np.asarray(targets, dtype=np.float64)
    counts = np.zeros(target_array.size, dtype=np.int64)
    total = 0
    for _, window in src.block_windows(band):
        values = valid_values(src.read(band, window=window, masked=True))
        if values.size == 0:
            continue
        values.sort()
        counts += np.searchsorted(values, target_array, side="right")
        total += int(values.size)

    if total == 0:
        raise ValueError(f"Raster contains no valid values: {src.name}")
    return (counts / total).tolist(), total


def normalized_rank(value: float, minimum: float, maximum: float) -> float:
    """Return a clamped 0..1 min-max rank (NaN for a constant layer)."""
    span = maximum - minimum
    if not math.isfinite(span) or span <= 0:
        return math.nan
    return min(1.0, max(0.0, (value - minimum) / span))


def process_raster(
    raster_path: Path,
    cropmarks: list[Cropmark],
    vector_crs: Any,
    band: int,
    all_touched: bool,
) -> list[dict[str, Any]]:
    """Build one output row per cropmark for one MAE raster."""
    rows: list[dict[str, Any]] = []
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"MAE raster has no CRS: {raster_path}")
        if band < 1 or band > src.count:
            raise ValueError(
                f"Band {band} is invalid for {raster_path} ({src.count} band(s))"
            )

        layer_stats, stats_source = read_layer_stats(raster_path, src, band)
        samples = extract_cropmark_values(
            src, cropmarks, vector_crs, band, all_touched
        )
        means = [float(values.mean()) for _, values in samples if values.size]
        ranks, layer_valid_pixels = percentile_ranks(src, band, means)
        rank_iterator = iter(ranks)

        for cropmark, values in samples:
            row: dict[str, Any] = {
                "id": cropmark.id,
                "image_name": raster_path.name,
                "MAE_value": math.nan,
                "pixel_count": int(values.size),
                "cropmark_median": math.nan,
                "cropmark_min": math.nan,
                "cropmark_max": math.nan,
                "cropmark_std": math.nan,
                "percentile_rank": math.nan,
                "min_max_rank": math.nan,
                "layer_valid_pixels": layer_valid_pixels,
                "stats_source": stats_source,
            }
            if values.size:
                mean = float(values.mean())
                row.update(
                    MAE_value=mean,
                    cropmark_median=float(np.median(values)),
                    cropmark_min=float(values.min()),
                    cropmark_max=float(values.max()),
                    cropmark_std=float(values.std()),
                    percentile_rank=next(rank_iterator),
                    min_max_rank=normalized_rank(
                        mean, layer_stats["min"], layer_stats["max"]
                    ),
                )
            row.update(
                {f"layer_{name}": layer_stats[name] for name in STATS_FIELDS}
            )
            row.update(
                {
                    name: value
                    for name, value in cropmark.properties.items()
                    if name not in row
                }
            )
            rows.append(row)
    return rows


def write_rows(
    output: Path, rows: list[dict[str, Any]], property_names: list[str]
) -> None:
    """Write results atomically, with stable column ordering."""
    base_fields = [
        "id",
        "image_name",
        "MAE_value",
        "pixel_count",
        "cropmark_median",
        "cropmark_min",
        "cropmark_max",
        "cropmark_std",
        "percentile_rank",
        "min_max_rank",
        "layer_valid_pixels",
        *(f"layer_{name}" for name in STATS_FIELDS),
        "stats_source",
    ]
    fields = base_fields + [name for name in property_names if name not in base_fields]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        rasters = discover_mae_rasters(args.mae_root.resolve())
        cropmarks, vector_crs, property_names = read_cropmarks(
            args.cropmarks.resolve(), args.id_field
        )
        rows: list[dict[str, Any]] = []
        for number, raster_path in enumerate(rasters, start=1):
            print(f"[{number}/{len(rasters)}] {raster_path}", file=sys.stderr)
            rows.extend(
                process_raster(
                    raster_path,
                    cropmarks,
                    vector_crs,
                    args.band,
                    all_touched=not args.center_only,
                )
            )
        write_rows(args.output.resolve(), rows, property_names)
    except (FileNotFoundError, OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    empty = sum(row["pixel_count"] == 0 for row in rows)
    print(
        f"Wrote {len(rows)} rows for {len(cropmarks)} cropmarks and "
        f"{len(rasters)} raster(s) to {args.output.resolve()}"
    )
    if empty:
        print(
            f"Warning: {empty} cropmark/raster pair(s) had no overlapping valid pixels.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
