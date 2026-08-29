"""Shared block/tile iteration helpers for windowed raster processing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List


def axis_tile_starts(length: int, tile_size: int, overlap: int = 0) -> List[int]:
    """Return tile start offsets covering one axis of length `length`.

    Each tile is `tile_size` wide and overlaps its neighbour by `overlap`.
    The final tile is shifted left (never shrunk) so every tile stays
    full-size while the whole axis is covered, including the edge.
    """
    if tile_size < 1:
        raise ValueError("Tile size must be positive.")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("Overlap must be at least 0 and smaller than tile size.")
    if length <= tile_size:
        return [0]

    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


@dataclass(frozen=True)
class BlockWindow:
    xoff: int
    yoff: int
    xsize: int
    ysize: int


def iter_blocks(width: int, height: int, blockx: int, blocky: int) -> Iterator[BlockWindow]:
    """Iterate non-overlapping windows covering a `width` x `height` raster,
    in raster (row-major) order. Matches GDAL's native block-loop pattern:
    the last window in each row/column is shrunk to fit rather than shifted."""
    for yoff in range(0, height, blocky):
        win_ysize = min(blocky, height - yoff)
        for xoff in range(0, width, blockx):
            win_xsize = min(blockx, width - xoff)
            yield BlockWindow(xoff, yoff, win_xsize, win_ysize)
