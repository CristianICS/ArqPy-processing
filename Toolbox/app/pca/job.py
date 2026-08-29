"""Orchestrates a full PCA run: resolve/generate band combinations, compute
each one, convert to COG, and optionally clip to a vector cutline.

Ports the per-combination loop that used to live inline in `gui/pca.py`'s
`run_job()` out of the GUI layer (which also fixes a layering violation:
the original called `sg.popup_ok(...)` directly from inside supposedly
headless business logic). Satisfies `core.processor.Processor`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from osgeo import gdal

from core.clip import clip_raster_to_vector
from core.progress import DEFAULT_PROGRESS, Progress
from core.raster_io import RasterProfile, translate_to_cog

from .utils import generate_combis, load_combis_from_csv, run_pca


@dataclass(frozen=True)
class PcaResult:
    output_paths: List[Path]


@dataclass
class PcaJob:
    img_path: Path
    out_folder: Path
    sensor_band_names: List[str]
    sensor_band_pos: List[int]
    combis_csv: Optional[Path] = None
    clip_vector: Optional[Path] = None

    def validate(self) -> None:
        if not self.img_path.exists():
            raise FileNotFoundError(f"Input image not found: {self.img_path}")
        if not self.out_folder.exists():
            raise NotADirectoryError(f"Output folder not found: {self.out_folder}")

    def run(self, progress: Progress = DEFAULT_PROGRESS) -> PcaResult:
        nodata = None
        if self.clip_vector is not None:
            ds = gdal.Open(str(self.img_path), gdal.GA_ReadOnly)
            nodata = ds.GetRasterBand(1).GetNoDataValue()
            ds = None

        if self.combis_csv is not None:
            combis_dict = load_combis_from_csv(self.combis_csv)
        else:
            combis_dict = generate_combis(
                self.sensor_band_names, self.sensor_band_pos,
                self.out_folder / "band_combinations.csv",
            )

        output_paths: List[Path] = []
        for i, item in combis_dict.items():
            out_stem = f"COMB{i}"
            out_name = f"{out_stem}.tif"
            out_name_temp = f"{out_stem}_temp.tif"

            out_clip_path = None
            if self.clip_vector is not None:
                out_name = f"{out_stem}_toclip.tif"
                out_clip_path = self.out_folder / f"{out_stem}.tif"

            out_path = self.out_folder / out_name
            out_temp_path = self.out_folder / out_name_temp

            # Skip combinations already computed in a previous run.
            if out_path.exists() or out_temp_path.exists():
                continue
            progress(f"Computing PCA {i}")

            run_pca(self.img_path, out_temp_path, item["pos"], ram=2048)

            translate_to_cog(
                out_temp_path, out_path,
                RasterProfile(compress="DEFLATE", predictor=2, bigtiff="IF_SAFER"),
                nodata=0,
            )
            out_temp_path.unlink()

            if self.clip_vector is not None:
                clip_raster_to_vector(
                    out_path, self.clip_vector, out_clip_path,
                    nodata=nodata,
                    profile=RasterProfile(compress="DEFLATE", predictor=3, bigtiff="IF_SAFER"),
                    as_cog=True,
                )
                out_path.unlink()
                output_paths.append(out_clip_path)
            else:
                output_paths.append(out_path)

        return PcaResult(output_paths=output_paths)
