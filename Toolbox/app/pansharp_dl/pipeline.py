"""End-to-end execution of the deep-learning pansharpening workflow."""

import os
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from osgeo import gdal
from scipy import io

from .grid import align_raster_grids


gdal.UseExceptions()

ALGORITHMS = (
    "Z-PNN",
    "A-PNN-TA-FR",
    "Z-PanNet",
    "PanNet-TA-FR",
    "Z-DRPNN",
    "DRPNN-TA-FR",
)
SENSORS = {"WV3": 8, "WV2": 8, "GE1": 4}
RATIO = 4


def _zpnn_root() -> Path:
    root = Path(__file__).resolve().parents[2] / "Z-PNN"
    if not (root / "main.py").is_file():
        raise FileNotFoundError(f"Bundled Z-PNN repository not found at: {root}")
    return root


def _raster_to_mat(ms_path: Path, pan_path: Path, mat_path: Path, expected_bands: int):
    ms_dataset = gdal.Open(str(ms_path), gdal.GA_ReadOnly)
    pan_dataset = gdal.Open(str(pan_path), gdal.GA_ReadOnly)
    if ms_dataset is None or pan_dataset is None:
        raise RuntimeError("Could not reopen the aligned raster inputs.")
    try:
        if ms_dataset.RasterCount != expected_bands:
            raise ValueError(
                f"The selected sensor expects {expected_bands} MS bands, "
                f"but the input contains {ms_dataset.RasterCount}."
            )
        ms = np.moveaxis(ms_dataset.ReadAsArray(), 0, -1)
        pan = pan_dataset.ReadAsArray()
        if pan.ndim != 2:
            raise ValueError("The aligned PAN input must be a two-dimensional array.")
        expected_shape = (ms.shape[0] * RATIO, ms.shape[1] * RATIO)
        if pan.shape != expected_shape:
            raise RuntimeError(
                f"Grid alignment failed: MS implies PAN {expected_shape}, got {pan.shape}."
            )
        io.savemat(str(mat_path), {"I_MS_LR": ms, "I_PAN": pan})
    finally:
        ms_dataset = None
        pan_dataset = None


def _mat_to_geotiff(mat_path: Path, reference_pan: Path, output_path: Path):
    output = io.loadmat(str(mat_path))["I_MS"]
    reference = gdal.Open(str(reference_pan), gdal.GA_ReadOnly)
    if reference is None:
        raise RuntimeError("Could not open the aligned PAN reference raster.")
    try:
        if output.shape[:2] != (reference.RasterYSize, reference.RasterXSize):
            raise RuntimeError(
                "Z-PNN output dimensions do not match the aligned PAN reference: "
                f"{output.shape[:2]} != {(reference.RasterYSize, reference.RasterXSize)}."
            )
        output = np.clip(output, 0, np.iinfo(np.uint16).max).astype(np.uint16)
        driver = gdal.GetDriverByName("GTiff")
        destination = driver.Create(
            str(output_path),
            reference.RasterXSize,
            reference.RasterYSize,
            output.shape[2],
            gdal.GDT_UInt16,
            options=["TILED=YES", "COMPRESS=DEFLATE", "PREDICTOR=2"],
        )
        if destination is None:
            raise RuntimeError(f"Could not create output raster: {output_path}")
        destination.SetGeoTransform(reference.GetGeoTransform())
        destination.SetProjection(reference.GetProjection())
        for band_index in range(output.shape[2]):
            destination.GetRasterBand(band_index + 1).WriteArray(output[:, :, band_index])
        destination.FlushCache()
        destination = None
    finally:
        reference = None


def run_pansharpening(
    ms_path: str | Path,
    pan_path: str | Path,
    algorithm: str,
    sensor: str,
    output_folder: str | Path,
    epochs: int = 1,
    use_cpu: bool = False,
    coregistration: bool = True,
) -> dict:
    """Align inputs, run a bundled Z-PNN method, and write a GeoTIFF."""

    ms_path = Path(ms_path).resolve()
    pan_path = Path(pan_path).resolve()
    output_folder = Path(output_folder).resolve()
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported pansharpening algorithm: {algorithm}")
    if sensor not in SENSORS:
        raise ValueError(f"Unsupported sensor: {sensor}")
    if epochs < 0:
        raise ValueError("Epochs cannot be negative.")
    output_folder.mkdir(parents=True, exist_ok=True)
    zpnn_root = _zpnn_root()

    safe_method = algorithm.replace("-", "_")
    output_tif = output_folder / f"{ms_path.stem}_{safe_method}.tif"
    with TemporaryDirectory(prefix="arqpy_dlpan_") as temporary:
        temporary_path = Path(temporary)
        print("Aligning MS and PAN grids...")
        alignment = align_raster_grids(ms_path, pan_path, temporary_path, ratio=RATIO)
        print(
            f"Aligned MS: {alignment.ms_width} x {alignment.ms_height}; "
            f"PAN: {alignment.pan_width} x {alignment.pan_height}."
        )

        input_mat = temporary_path / f"{ms_path.stem}.mat"
        print("Creating the Z-PNN input MAT file...")
        _raster_to_mat(alignment.ms_path, alignment.pan_path, input_mat, SENSORS[sensor])

        command = [
            sys.executable,
            str(zpnn_root / "main.py"),
            "--input",
            str(input_mat),
            "--out_dir",
            str(output_folder) + os.sep,
            "--sensor",
            sensor,
            "--method",
            algorithm,
            "--epochs",
            str(epochs),
        ]
        if use_cpu:
            command.append("--use_cpu")
        if coregistration:
            command.append("--coregistration")

        print(f"Running {algorithm}...")
        process = subprocess.Popen(
            command,
            cwd=zpnn_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="")
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"{algorithm} exited with code {return_code}.")

        # Z-PNN derives the result name with ``name.split('.')[0]`` rather
        # than pathlib's stem, so mirror that behavior for dotted filenames.
        zpnn_input_stem = input_mat.name.split(".")[0]
        output_mat = output_folder / f"{zpnn_input_stem}_{algorithm}.mat"
        if not output_mat.is_file():
            raise RuntimeError(f"Z-PNN did not create its expected output: {output_mat}")
        print("Writing the georeferenced pansharpened GeoTIFF...")
        _mat_to_geotiff(output_mat, alignment.pan_path, output_tif)

    return {"tif": output_tif, "mat": output_mat}
