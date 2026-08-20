from dataclasses import dataclass, asdict, fields, replace
from pathlib import Path
import csv

import rasterio
import numpy as np

@dataclass(slots=True)
class Stats:
    """
    Store statistics for a derived image product.
    
    Attributes:
    -----------
    id : str
        Derived image product name.
    mean : float
        Mean value of the derived product's saliency map.
    median : float
        Median value of the saliency map.
    min : float
        Minimum value of the saliency map.
    p90 : float
        90th percentile of the saliency map.
    p95 : float
        95th percentile of the saliency map.
    max : float
        Maximum value of the saliency map.
    std : float
        Standard deviation of the saliency map.
    """
    id: str
    mean: float
    median: float
    min: float
    p90: float
    p95: float
    max: float
    std: float

    def round_to2(self) -> "Stats":
        """Return a copy with all statistical values rounded to two decimals."""
        return replace(
            self,
            mean=round(self.mean, 2),
            median=round(self.median, 2),
            min=round(self.min, 2),
            p90=round(self.p90, 2),
            p95=round(self.p95, 2),
            max=round(self.max, 2),
            std=round(self.std, 2),
        )
    
    def to_csv_row(self):
        """Return the stored data as a dictionary suitable for a CSV row."""
        return asdict(self.round_to2())


def is_inside_stats(stats_path: str | Path, img_name: str):
    """Check if a derived image is already inside a CSV stats file."""
    stats_path = Path(stats_path)

    if not stats_path.exists():
        return False

    with stats_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None or "id" not in reader.fieldnames:
            raise ValueError(
                "The provided CSV file with the saliency stats "
                f"does not contain a valid 'id' column: {stats_path}"
            )

        return any(row["id"] == img_name for row in reader)


def update_csv(csv_path: str | Path, stats: Stats):
    """
    Store statistics by image inside a CSV file

    Creates the file if it does not exist.
    Avoids duplicates based on img_name.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.
    stats : Stats
        Statistics to store.
        
    Returns
    -------
    bool
        True if a row was added; False if the image product
        was already present in the CSV file.
    """
    csv_path = Path(csv_path)

    fieldnames = [field.name for field in fields(Stats)]

    # Do not add duplicate entries
    if (
        csv_path.exists()
        and csv_path.stat().st_size > 0
        and is_inside_stats(csv_path, stats.id)
    ):
        return False

    row = stats.to_csv_row()

    # Write the header when creating a new or empty file
    write_header = not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if write_header:
                writer.writeheader()
            
            writer.writerow(row)
    
    return True


def compute_raster_stats(tif_path: str| Path, band = 1) -> tuple[Stats, np.ndarray]:
    """
    Compute basic statistics for a raster band using rasterio.

    Parameters
    ----------
    path : str | Path
        Path to the raster file.
    band : int, optional
        Band number to read (default is 1).
    """
    tif_path = Path(tif_path)

    # Read as a masked array so nodata values are ignored automatically
    with rasterio.open(tif_path) as src:
        data = src.read(band, masked=True)

    # Keep only valid pixels as a 1D array for fast vectorized statistics
    values = data.compressed()  # type: np.ndarray

    # Guard against empty rasters (e.g. all pixels are nodata)
    if values.size == 0:
        raise ValueError("Raster band contains no valid data.")
    
    stats = Stats(
        id = tif_path.stem,
        mean=values.mean(),
        median=np.median(values),
        min=values.min(),
        p90=np.percentile(values, 90),
        p95=np.percentile(values, 95),
        max=values.max(),
        std=values.std()
    )

    return stats, values


def percentile_rank(x, values):
    """
    Compute percentile rank for a value given raster values.

    Percentile rank does not care about the distribution. Interpretation:

    * p = 0.9 -> higher than 90% of pixels
    * p = 0.5 -> median
    * p = 0.01 -> very low value
    """
    percentile = (values <= x).sum() / values.size

    return percentile
