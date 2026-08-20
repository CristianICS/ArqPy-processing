# ArqPy tool guides

These guides explain what each graphical tool does, what to enter in every field, what files it creates, and how to interpret the result. They describe the current implementation of the toolbox.

## Recommended workflow

1. [Atmospheric correction](atmospheric-correction.md) — convert raw digital numbers to surface reflectance.
2. [Pansharpening](pansharpening.md) — combine multispectral information with panchromatic spatial detail.
3. [Principal component analysis (PCA)](pca.md) — expose correlated and subtle spectral variation.
4. [Spectral indices](spectral-indices.md) — calculate vegetation, soil, and related band ratios.
5. [High-pass filters](high-pass-filters.md) — emphasize edges and local texture.
6. [MAE saliency](mae-saliency.md) — rank visually unusual regions using a pretrained image model.
7. [Deep-learning pansharpening](deep-learning-pansharpening.md) — run the bundled Z-PNN methods.

Optional external tool:

- [SAM 3 crop-mark detection](sam3-crop-marks.md) — segment candidate features from a text prompt.

The order is a recommendation, not a requirement. Keep the original data and place outputs from each stage in separate folders. A visible anomaly is not automatically archaeological: compare it with field boundaries, drainage, geology, recent land use, image seams, and evidence from other dates or methods.

## Controls shared by several tools

| Control | Behaviour |
| --- | --- |
| **Run** | Starts processing. It is disabled while a job is running. Progress and errors appear in the text panel. |
| **Exit** | Closes the window. Do not use it to interrupt a job unless necessary. |
| **Output folder** | Where offered, leaving it blank uses the automatic location described in that tool's guide. A manually selected folder usually must already exist. |
| **Vector layer to perform a clip operation** | Optional cutline. The output is cropped to the vector geometry. Use a valid `.shp`, `.gpkg`, or `.geojson` and ensure its CRS is defined. KML is currently unsupported because the packaged GDAL environment does not include the `LIBKML` driver. |

## Sensor band names

Band numbers are one-based: the first raster band is band 1.

| Position | WV3 | LEGION-06 |
| ---: | --- | --- |
| 1 | Coastal (C) | Coastal (C) |
| 2 | Blue (B) | Blue (B) |
| 3 | Green (G) | Green (G) |
| 4 | Yellow (Y) | Yellow (Y) |
| 5 | Red (R) | Red (R) |
| 6 | Red edge (RE1) | Red edge 1 (RE1) |
| 7 | Near infrared 1 (N) | Red edge 2 (RE2) |
| 8 | Near infrared 2 (N2) | Near infrared (N) |

Selecting the wrong sensor does not relabel the source raster; it makes the software interpret its band positions incorrectly.
