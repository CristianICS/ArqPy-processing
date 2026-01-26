from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from osgeo import gdal


@dataclass(frozen=True)
class HighPassKernels:
    """Common high-pass / edge-emphasis kernels."""

    # Laplacian (4-neighborhood)
    LAPLACIAN_4N: np.ndarray = np.array(
        [[0, -1, 0],
         [-1, 4, -1],
         [0, -1, 0]],
        dtype=np.float32,
    )

    # Laplacian of Gaussian (5x5) – edge emphasis with some smoothing behavior
    LOG_5: np.ndarray = np.array(
        [[0, 0, -1, 0, 0],
         [0, -1, -2, -1, 0],
         [-1, -2, 16, -2, -1],
         [0, -1, -2, -1, 0],
         [0, 0, -1, 0, 0]],
        dtype=np.float32,
    )

    # Sobel derivatives
    SOBEL_X: np.ndarray = np.array(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        dtype=np.float32,
    )

    SOBEL_Y: np.ndarray = np.array(
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]],
        dtype=np.float32,
    )

    # High-boost / unsharp-ish 3x3
    HIGHBOOST_3: np.ndarray = np.array(
        [[-1, -1, -1],
         [-1, 9, -1],
         [-1, -1, -1]],
        dtype=np.float32,
    )


class GDALHighPassFilter:
    """
    Apply selected high-pass filters to a GDAL-readable raster band.
    Designed for large rasters: uses tiled chunk processing with overlap.

    Example:
        hp = GDALHighPassFilter(Path("in.tif"))
        hp.run_all(Path("out/corona_hp"))
    """

    def __init__(
        self,
        input_path: Path,
        band_index: int = 1,
        block: int = 1024,
        compress: str = "DEFLATE",
        bigtiff: str = "IF_SAFER",
    ) -> None:
        gdal.UseExceptions()
        self.input_path = Path(input_path)
        self.band_index = int(band_index)
        self.block = int(block)
        self.compress = compress
        self.bigtiff = bigtiff

        self._ds = gdal.Open(str(self.input_path), gdal.GA_ReadOnly)
        if self._ds is None:
            raise RuntimeError(f"Could not open input raster: {self.input_path}")

        self._band = self._ds.GetRasterBand(self.band_index)
        if self._band is None:
            raise RuntimeError(f"Could not read band {self.band_index} from {self.input_path}")

        self.nodata: Optional[float] = self._band.GetNoDataValue()

    # -------------------------
    # Public API
    # -------------------------
    def run_all(self, out_prefix: Path) -> Tuple[Path, Path, Path, Path]:
        """
        Compute:
          - laplacian
          - log5
          - sobel magnitude
          - highboost

        Returns output paths in that order.
        """
        out_prefix = Path(out_prefix)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)

        lap = self.apply_kernel(HighPassKernels.LAPLACIAN_4N, out_prefix.with_name(out_prefix.name + "_laplacian.tif"))
        log = self.apply_kernel(HighPassKernels.LOG_5, out_prefix.with_name(out_prefix.name + "_log5.tif"))
        sob = self.apply_sobel_magnitude(out_prefix.with_name(out_prefix.name + "_sobel_mag.tif"))
        hb = self.apply_kernel(HighPassKernels.HIGHBOOST_3, out_prefix.with_name(out_prefix.name + "_highboost.tif"))

        return lap, log, sob, hb

    def apply_kernel(self, kernel: np.ndarray, out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        out_ds = self._create_output_like(out_path, dtype=gdal.GDT_Float32)
        self._convolve_chunked(self._band, kernel.astype(np.float32), out_ds)
        out_ds = None
        return out_path

    def apply_sobel_magnitude(self, out_path: Path) -> Path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Temp rasters live alongside outputs
        tmp_gx = out_path.with_name(out_path.stem + "__tmp_sobelx.tif")
        tmp_gy = out_path.with_name(out_path.stem + "__tmp_sobely.tif")

        # Sobel X/Y
        dsx = self._create_output_like(tmp_gx, dtype=gdal.GDT_Float32)
        self._convolve_chunked(self._band, HighPassKernels.SOBEL_X, dsx)
        dsx = None

        dsy = self._create_output_like(tmp_gy, dtype=gdal.GDT_Float32)
        self._convolve_chunked(self._band, HighPassKernels.SOBEL_Y, dsy)
        dsy = None

        # Combine to magnitude
        gx_ds = gdal.Open(str(tmp_gx), gdal.GA_ReadOnly)
        gy_ds = gdal.Open(str(tmp_gy), gdal.GA_ReadOnly)
        gx_b = gx_ds.GetRasterBand(1)
        gy_b = gy_ds.GetRasterBand(1)

        out_ds = self._create_output_like(out_path, dtype=gdal.GDT_Float32)
        out_b = out_ds.GetRasterBand(1)

        xsize, ysize = self._ds.RasterXSize, self._ds.RasterYSize
        bs = self.block

        for y0 in range(0, ysize, bs):
            y1 = min(y0 + bs, ysize)
            for x0 in range(0, xsize, bs):
                x1 = min(x0 + bs, xsize)

                gx = gx_b.ReadAsArray(x0, y0, x1 - x0, y1 - y0).astype(np.float32)
                gy = gy_b.ReadAsArray(x0, y0, x1 - x0, y1 - y0).astype(np.float32)

                mag = np.sqrt(gx * gx + gy * gy)

                if self.nodata is not None:
                    mag = np.where((gx == self.nodata) | (gy == self.nodata), self.nodata, mag)

                out_b.WriteArray(mag, xoff=x0, yoff=y0)

        out_b.FlushCache()
        out_ds = None
        gx_ds = None
        gy_ds = None

        # Cleanup temps
        try:
            tmp_gx.unlink(missing_ok=True)
            tmp_gy.unlink(missing_ok=True)
        except TypeError:
            # Python < 3.8: missing_ok not available
            if tmp_gx.exists():
                tmp_gx.unlink()
            if tmp_gy.exists():
                tmp_gy.unlink()

        return out_path

    # -------------------------
    # Internals
    # -------------------------
    def _create_output_like(self, out_path: Path, dtype=gdal.GDT_Float32) -> gdal.Dataset:
        drv = gdal.GetDriverByName("GTiff")
        out_ds = drv.Create(
            str(out_path),
            self._ds.RasterXSize,
            self._ds.RasterYSize,
            1,
            dtype,
            options=[
                "TILED=YES",
                f"COMPRESS={self.compress}",
                "PREDICTOR=3",          # best for float
                f"BIGTIFF={self.bigtiff}",
                "NUM_THREADS=ALL_CPUS",
            ],
        )
        out_ds.SetGeoTransform(self._ds.GetGeoTransform())
        out_ds.SetProjection(self._ds.GetProjection())
        if self.nodata is not None:
            out_ds.GetRasterBand(1).SetNoDataValue(self.nodata)
        return out_ds

    def _convolve_chunked(self, band: gdal.Band, kernel: np.ndarray, out_ds: gdal.Dataset) -> None:
        """
        Chunked convolution with overlap. Edge handling: reflect padding.

        Nodata handling (robust for high-pass kernels):
        - If NoData is None: plain convolution.
        - If NoData is set:
            * For kernels with non-zero sum (typical low-pass), we do local renormalization.
            * For zero-sum kernels (typical high-pass/derivatives), renormalization is invalid;
                instead we compute convolution ignoring NoData pixels and set output to NoData
                wherever the footprint touched any NoData.
        - Supports NoData being NaN (common for float rasters).
        """
        k = kernel.astype(np.float32)
        kh, kw = k.shape
        pad_y, pad_x = kh // 2, kw // 2

        xsize, ysize = band.XSize, band.YSize
        out_band = out_ds.GetRasterBand(1)
        bs = self.block
        nd = self.nodata

        # Decide whether renormalization is meaningful (low-pass) or harmful (high-pass/derivative)
        ksum = float(np.sum(k))
        do_renorm = abs(ksum) > 1e-6  # non-zero-sum kernels only

        for y0 in range(0, ysize, bs):
            y1 = min(y0 + bs, ysize)
            for x0 in range(0, xsize, bs):
                x1 = min(x0 + bs, xsize)

                # Read with overlap
                rx0 = max(x0 - pad_x, 0)
                ry0 = max(y0 - pad_y, 0)
                rx1 = min(x1 + pad_x, xsize)
                ry1 = min(y1 + pad_y, ysize)

                arr = band.ReadAsArray(rx0, ry0, rx1 - rx0, ry1 - ry0).astype(np.float32)
                arr_p = np.pad(arr, ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")

                out_h = arr_p.shape[0] - 2 * pad_y
                out_w = arr_p.shape[1] - 2 * pad_x

                out_full = np.zeros((out_h, out_w), dtype=np.float32)

                if nd is None:
                    # Fast path: no nodata
                    for j in range(kh):
                        for i in range(kw):
                            out_full += k[j, i] * arr_p[j:j + out_h, i:i + out_w]
                else:
                    # Build nodata mask (handle NaN nodata correctly)
                    if np.isnan(nd):
                        mask_p = np.isnan(arr_p)
                    else:
                        mask_p = (arr_p == nd)

                    if do_renorm:
                        # Low-pass style: renormalize locally by sum of weights over valid pixels
                        wsum = np.zeros((out_h, out_w), dtype=np.float32)

                        for j in range(kh):
                            for i in range(kw):
                                w = k[j, i]
                                win = arr_p[j:j + out_h, i:i + out_w]
                                valid = ~mask_p[j:j + out_h, i:i + out_w]
                                out_full += w * np.where(valid, win, 0.0)
                                wsum += w * valid.astype(np.float32)

                        with np.errstate(divide="ignore", invalid="ignore"):
                            out_full = np.where(wsum != 0.0, out_full / wsum, nd)
                    else:
                        # High-pass/derivative: DO NOT renormalize (kernel sum ~ 0)
                        # Instead, ignore nodata in the sum and mark output nodata
                        # wherever any nodata was inside the footprint.
                        any_nd = np.zeros((out_h, out_w), dtype=bool)

                        for j in range(kh):
                            for i in range(kw):
                                win = arr_p[j:j + out_h, i:i + out_w]
                                nd_here = mask_p[j:j + out_h, i:i + out_w]
                                any_nd |= nd_here
                                out_full += k[j, i] * np.where(nd_here, 0.0, win)

                        out_full = np.where(any_nd, nd, out_full)

                # Crop back to non-overlap chunk
                oy0 = y0 - ry0
                ox0 = x0 - rx0
                out_chunk = out_full[oy0:oy0 + (y1 - y0), ox0:ox0 + (x1 - x0)]

                out_band.WriteArray(out_chunk, xoff=x0, yoff=y0)

        out_band.FlushCache()
