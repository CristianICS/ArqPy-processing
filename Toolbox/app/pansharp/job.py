"""Orchestrates a full classic-pansharpening run: Brovey (with WV3's LMVM
NIR2 refinement) or Bayesian, then an optional clip to a vector cutline.

Ports the per-run orchestration that used to live inline in
`gui/pansharpen.py`'s `run_job()` out of the GUI layer. Satisfies
`core.processor.Processor`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from core.clip import clip_raster_to_vector
from core.progress import DEFAULT_PROGRESS, Progress
from core.raster_io import RasterProfile

from .utils import SENSORS, bayesian, brovey

METHODS = ("Bayesian", "wBrovey")


@dataclass(frozen=True)
class PansharpResult:
    output_path: Path


@dataclass
class PansharpJob:
    mul_path: Path
    pan_path: Path
    out_folder: Path
    method: str
    sensor_name: str
    bayes_lambda: float = 0.995
    clip_vector: Optional[Path] = None

    def validate(self) -> None:
        if not self.mul_path.exists():
            raise FileNotFoundError(f"MUL image not found: {self.mul_path}")
        if not self.pan_path.exists():
            raise FileNotFoundError(f"PAN image not found: {self.pan_path}")
        if not self.out_folder.exists():
            raise NotADirectoryError(f"Output folder not found: {self.out_folder}")
        if self.method not in METHODS:
            raise ValueError(f"Unknown pansharpening method: {self.method}")
        if self.sensor_name not in SENSORS:
            raise ValueError(f"Unknown sensor: {self.sensor_name}")
        if self.method == "Bayesian" and not (0 <= self.bayes_lambda <= 1):
            raise ValueError(
                "The valid range of Bayes' lambda parameter is 0-1."
            )

    def run(self, progress: Progress = DEFAULT_PROGRESS) -> PansharpResult:
        sensor = SENSORS[self.sensor_name]
        # Non-blank clip vector: pansharpen into a "_temp" file first, then
        # clip it down to the final (un-suffixed) output name below.
        suffix = "_temp" if self.clip_vector is not None else ""

        if self.method == "Bayesian":
            progress("Starting Bayesian pansharpening...")
            out_name = self.mul_path.name.replace(".tif", f"_bayes{suffix}.tif")
            out_path = self.out_folder / out_name
            bayesian(self.mul_path, self.pan_path, out_path, self.bayes_lambda)
        else:
            progress("Starting weighted Brovey pansharpening...")
            out_name = self.mul_path.name.replace(".tif", f"_brovey{suffix}.tif")
            out_path = self.out_folder / out_name
            brovey(sensor, self.mul_path, self.pan_path, out_path)

            if self.sensor_name == "WV3":
                progress("Pansharpening the NIR2 band (LMVM)...")
                lmnv_path = sensor.lmnv(self.mul_path, self.pan_path)
                sensor.merge_pansharpened_bands(out_path, lmnv_path)

        if self.clip_vector is not None:
            progress("Clipping...")
            out_clip_path = out_path.with_stem(out_path.stem.replace(suffix, ""))
            clip_raster_to_vector(
                out_path, self.clip_vector, out_clip_path,
                nodata=-9999,
                profile=RasterProfile(compress="DEFLATE", predictor=2, bigtiff="IF_SAFER"),
                as_cog=True,
            )
            out_path.unlink()
            out_path = out_clip_path

        return PansharpResult(output_path=out_path)
