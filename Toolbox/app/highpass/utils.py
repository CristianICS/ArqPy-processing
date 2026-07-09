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
    def run_all(
        self, out_prefix: Path, clip_layer_path: str | bool = False
    ) -> Tuple[Path, Path, Path, Path, Path]:
        """
        Compute:
          - laplacian
          - log5
          - sobel magnitude
          - highboost
          - GLCM local contrast high-pass texture response

        Returns output paths in that order.
        """
        sfx = "_temp" if clip_layer_path else ""

        out_prefix = Path(out_prefix)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)

        # Create output names and paths
        lap_path = out_prefix.with_name(out_prefix.name + f"_laplacian{sfx}.tif")
        log_path = out_prefix.with_name(out_prefix.name + f"_log5{sfx}.tif")
        sob_path = out_prefix.with_name(out_prefix.name + f"_sobel_mag{sfx}.tif")
        hbt_path = out_prefix.with_name(out_prefix.name + f"_highboost{sfx}.tif")
        glcm_path = out_prefix.with_name(out_prefix.name + f"_glcm_contrast{sfx}.tif")

        lap = self.apply_kernel(HighPassKernels.LAPLACIAN_4N, lap_path)
        log = self.apply_kernel(HighPassKernels.LOG_5, log_path)
        sob = self.apply_sobel_magnitude(sob_path)
        hbt = self.apply_kernel(HighPassKernels.HIGHBOOST_3, hbt_path)
        glcm = self.apply_glcm_contrast_highpass(glcm_path)

        if clip_layer_path:
            for lpath in [lap_path, log_path, sob_path, hbt_path, glcm_path]:
                clip_out = lpath.with_name(lpath.name.replace(sfx, ""))
                co = [
                    "COMPRESS=DEFLATE",
                    "PREDICTOR=3",  # Better for floats
                    "BIGTIFF=IF_SAFER",
                ]

                gdal.Warp(
                    clip_out,
                    lpath,
                    cutlineDSName=clip_layer_path,
                    cropToCutline=True,
                    multithread=True,
                    format="COG",
                    creationOptions=co,
                    warpOptions=["NUM_THREADS=ALL_CPUS"],
                )
                lpath.unlink()

            lap = lap.with_name(lap.name.replace(sfx, ""))
            log = log.with_name(log.name.replace(sfx, ""))
            sob = sob.with_name(sob.name.replace(sfx, ""))
            hbt = hbt.with_name(hbt.name.replace(sfx, ""))
            glcm = glcm.with_name(glcm.name.replace(sfx, ""))

        return lap, log, sob, hbt, glcm

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

    def apply_glcm_contrast_highpass(
        self,
        out_path: Path,
        window_size: int = 7,
        levels: int = 32,
        offsets: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1)),
    ) -> Path:
        """
        Compute a local GLCM-contrast high-pass texture response.

        This is a co-occurrence-matrix-based high-pass measure. For each local
        window, it computes the GLCM contrast feature:

            contrast = sum(P[i, j] * (i - j)^2)

        Instead of explicitly building one GLCM per pixel, this implementation
        computes the mathematically equivalent local mean of squared gray-level
        differences for the requested offsets. This is much faster and more
        practical for large rasters.

        Args:
            out_path: Output raster path.
            window_size: Odd local window size, e.g. 5, 7, 9, 11.
            levels: Number of quantized gray levels used by the GLCM logic.
            offsets: Pixel offsets as (dy, dx), e.g. horizontal, vertical,
                and diagonal co-occurrence directions.
        """
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError("window_size must be an odd integer >= 3")
        if levels < 2:
            raise ValueError("levels must be >= 2")
        if not offsets:
            raise ValueError("offsets must contain at least one (dy, dx) pair")

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        vmin, vmax = self._valid_minmax(self._band)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            raise RuntimeError("Could not compute a valid min/max range for GLCM quantization")

        out_ds = self._create_output_like(out_path, dtype=gdal.GDT_Float32)
        self._glcm_contrast_chunked(
            self._band,
            out_ds,
            window_size=window_size,
            levels=levels,
            offsets=offsets,
            vmin=float(vmin),
            vmax=float(vmax),
        )
        out_ds = None
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

    def _valid_minmax(self, band: gdal.Band) -> Tuple[float, float]:
        """Chunked min/max that excludes NoData, including NaN NoData."""
        xsize, ysize = band.XSize, band.YSize
        bs = self.block
        nd = self.nodata
        vmin = np.inf
        vmax = -np.inf

        for y0 in range(0, ysize, bs):
            y1 = min(y0 + bs, ysize)
            for x0 in range(0, xsize, bs):
                x1 = min(x0 + bs, xsize)
                arr = band.ReadAsArray(x0, y0, x1 - x0, y1 - y0).astype(np.float32)

                valid = np.isfinite(arr)
                if nd is not None:
                    if np.isnan(nd):
                        valid &= ~np.isnan(arr)
                    else:
                        valid &= arr != nd

                if np.any(valid):
                    vals = arr[valid]
                    vmin = min(vmin, float(np.min(vals)))
                    vmax = max(vmax, float(np.max(vals)))

        return vmin, vmax

    @staticmethod
    def _box_sum_same(arr: np.ndarray, radius: int) -> np.ndarray:
        """Square-window sum with constant-zero padding, returned at input shape."""
        win = 2 * radius + 1
        padded = np.pad(arr, ((radius, radius), (radius, radius)), mode="constant", constant_values=0)
        integral = np.pad(padded.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)), mode="constant")
        return (
            integral[win:, win:]
            - integral[:-win, win:]
            - integral[win:, :-win]
            + integral[:-win, :-win]
        )

    def _glcm_contrast_chunked(
        self,
        band: gdal.Band,
        out_ds: gdal.Dataset,
        window_size: int,
        levels: int,
        offsets: Tuple[Tuple[int, int], ...],
        vmin: float,
        vmax: float,
    ) -> None:
        radius = window_size // 2
        max_dy = max(abs(dy) for dy, _ in offsets)
        max_dx = max(abs(dx) for _, dx in offsets)
        margin_y = radius + max_dy
        margin_x = radius + max_dx

        xsize, ysize = band.XSize, band.YSize
        out_band = out_ds.GetRasterBand(1)
        bs = self.block
        nd = self.nodata
        scale = (levels - 1) / (vmax - vmin)

        for y0 in range(0, ysize, bs):
            y1 = min(y0 + bs, ysize)
            for x0 in range(0, xsize, bs):
                x1 = min(x0 + bs, xsize)

                rx0 = max(x0 - margin_x, 0)
                ry0 = max(y0 - margin_y, 0)
                rx1 = min(x1 + margin_x, xsize)
                ry1 = min(y1 + margin_y, ysize)

                arr = band.ReadAsArray(rx0, ry0, rx1 - rx0, ry1 - ry0).astype(np.float32)

                valid = np.isfinite(arr)
                if nd is not None:
                    if np.isnan(nd):
                        valid &= ~np.isnan(arr)
                    else:
                        valid &= arr != nd

                q = np.zeros(arr.shape, dtype=np.int16)
                q[valid] = np.clip(
                    np.rint((arr[valid] - vmin) * scale),
                    0,
                    levels - 1,
                ).astype(np.int16)

                contrast_sum = np.zeros(arr.shape, dtype=np.float32)
                contrast_count = np.zeros(arr.shape, dtype=np.float32)
                h, w = arr.shape

                for dy, dx in offsets:
                    y_src0 = max(0, -dy)
                    y_src1 = min(h, h - dy)
                    x_src0 = max(0, -dx)
                    x_src1 = min(w, w - dx)

                    y_dst0 = y_src0 + dy
                    y_dst1 = y_src1 + dy
                    x_dst0 = x_src0 + dx
                    x_dst1 = x_src1 + dx

                    pair_diff = np.zeros(arr.shape, dtype=np.float32)
                    pair_valid = np.zeros(arr.shape, dtype=np.float32)

                    src = q[y_src0:y_src1, x_src0:x_src1]
                    dst = q[y_dst0:y_dst1, x_dst0:x_dst1]
                    ok = valid[y_src0:y_src1, x_src0:x_src1] & valid[y_dst0:y_dst1, x_dst0:x_dst1]

                    pair_diff[y_src0:y_src1, x_src0:x_src1] = np.where(
                        ok,
                        (src.astype(np.float32) - dst.astype(np.float32)) ** 2,
                        0.0,
                    )
                    pair_valid[y_src0:y_src1, x_src0:x_src1] = ok.astype(np.float32)

                    contrast_sum += self._box_sum_same(pair_diff, radius)
                    contrast_count += self._box_sum_same(pair_valid, radius)

                with np.errstate(divide="ignore", invalid="ignore"):
                    glcm_contrast = np.where(contrast_count > 0, contrast_sum / contrast_count, 0.0)

                # Mark center pixels that are NoData as NoData in the output.
                if nd is not None:
                    glcm_contrast = np.where(valid, glcm_contrast, nd).astype(np.float32)
                else:
                    glcm_contrast = glcm_contrast.astype(np.float32)

                oy0 = y0 - ry0
                ox0 = x0 - rx0
                out_chunk = glcm_contrast[oy0:oy0 + (y1 - y0), ox0:ox0 + (x1 - x0)]
                out_band.WriteArray(out_chunk, xoff=x0, yoff=y0)

        out_band.FlushCache()

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
