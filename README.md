# ArcPy Toolbox

This toolbox performs a series of remote sensing–derived products for archaeological analysis, with a particular focus on crop mark detection. It also includes an option to apply a Masked Autoencoder to identify zones with a higher probability of containing crop marks.

The toolbox is distributed as a self-contained package and **does not require Conda, Miniforge, or manual environment setup by the end user**.

The available operations are executed in the following recommended order:

1. `atmcorr` – atmospheric correction
2. `pansharpening` – spatial–spectral fusion
3. `pca` – principal component analysis
4. `spectral_indices` – spectral index computation
5. `highpass` – spatial filtering
6. `mae` – Masked Autoencoder–based analysis

Each operation provides a graphical user interface for configuring the parameters required for execution.

## Available sensors

Atmospheric correction is currently supported for the following sensors:

- GEOSAT
- WorldView-3 (WV3)
- WorldView LEGION-06

The `pca`, `pansharpening` and `spectral_indices` operations are available only for WV3 and LEGION sensors.

The `highpass` and `mae` operations are available for any image provided in GeoTIFF format.

## Data

This repository includes a reduced WorldView-3 image provided for testing and demonstration purposes. The dataset covers the Zar Tepe archaeological site (Uzbekistan).

## Installation

No installation is required beyond downloading and extracting the toolbox.

1. Download the toolbox release.
2. Extract the archive to a local directory (e.g. `C:\Toolbox`).
3. Ensure the directory structure is preserved.

The toolbox includes all required Python environments and external dependencies.

## Directory structure

After extraction, the toolbox directory must have the following structure:

```txt
Toolbox
|
|- app        (Python scripts used to perform the operations)
|- env        (packed Python environment for preprocessing)
|  |
|  |- OTB-9.1.1-Win64  (Orfeo ToolBox binaries)
|- env_mae    (packed Python environment for MAE analysis)
|- atmcorr.bat
|- ...
|- README.md
```

## External dependencies

The toolbox relies on **Orfeo ToolBox (OTB) version 9.1.1**, which is already included in the release.
No additional downloads are required.

## Running the toolbox

To launch the toolbox operations:

1. Double-click on the desired operation (e.g. `atmcorr.bat`), or
2. Run it from a Windows command prompt.

Each batch file automatically activates the appropriate environment and routes the operation to the correct backend.

## Supported systems

* Operating system: **Windows 11**
* No administrator privileges required

## Environment construction (for developers)

The release includes the complete application folder, together with the required environments.
To replicate the same structure, follow the steps below in a Miniforge or Miniconda console.

Install the environments using the provided `.yml` files:

```bash
mamba env create -f requirements_otb.yml
mamba env create -f requirements_mae.yml
```

Pack the environments and place the resulting archives inside the toolbox directory. They must be unpacked later:

```bash
conda pack -n arcpy_otb -o Toolbox/env.zip
conda pack -n mae -o Toolbox/env_mae.zip
```

Finally, include [Orfeo ToolBox](https://www.orfeo-toolbox.org/) inside the `env` folder.
Download **OTB version 9.1.1** and place it within the toolbox directory as follows:

```txt
Toolbox
|
|- env
|  |
|  |- ...
|  |- OTB-9.1.1-Win64
|  |- ...
|_ ...
```
