# ArcPy Toolbox

This toolbox generates a series of products derived from remote sensing images for archaeological analysis, with a particular focus on crop mark detection. It also includes an option to apply a Masked Autoencoder (MAE) to identify zones with a higher probability of containing crop marks.

Furthermore, the tool includes two additional functions: one to detect crop marks using [Meta's SAM 3 model](https://github.com/facebookresearch/sam3) (`sam3_detection.bat`) and to do a pansharpening operation (`pansharpen_dl.bat`) based in convolutional neural networks included in the [Z-PNN repository](https://github.com/matciotola/Z-PNN).

These two tools require additional customization. See [section Additional tools](#additional-tools).

## Introduction

The toolbox is distributed as a self-contained package and **does not require Conda, Miniforge, or manual environment setup by the end user**.

The available operations should be executed in the following recommended order:

1. `atmcorr` - atmospheric correction
2. `pansharpening` - spatial-spectral fusion
3. `pca` - principal component analysis
4. `spectral_indices` - spectral index computation
5. `highpass` - spatial filtering
6. `mae` - Masked Autoencoder-based analysis

The two additional operations are:

7. `sam3_cropmarks` - SAM 3 model to find crop marks in RGB images
8. `pansharpening_dl` - spatial-spectral fusion based on Deep Learning models

Each operation provides a graphical user interface for configuring the parameters required for execution.

## Tool instructions

The [ArqPy tool guides](docs/tools/README.md) provide a separate, archaeologist-oriented manual for every launcher. Each guide explains all GUI options and defaults, input requirements, processing behaviour, output filenames, interpretation, and important limitations.

## Available sensors

Atmospheric correction is currently supported for the following sensors:

* WorldView-3 (WV3)
* WorldView LEGION-06


The `pca`, `pansharpening`, and `spectral_indices` operations are available only for WV3 and LEGION sensors.

The `highpass` and `mae` operations are available for any image provided in GeoTIFF format.

GeoEye1 (GE1) is only supported for `pansharpening_cnn` tool.

## Data

This repository includes a reduced WorldView-3 image for testing and demonstration purposes. The dataset covers the Zar Tepe archaeological site in Uzbekistan.

## Installation

No installation is required beyond downloading and extracting the toolbox.

1. Download the toolbox release.
2. Extract the archive to a local directory, for example `C:\Toolbox`.
3. Extract the folders `env` and `env_mae`.
4. Ensure that the directory structure is preserved.

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
|- Z-PNN      (modified Z-PNN repository to match the environment)
|- atmcorr.bat
|- ...
|- README.md
```

## External dependencies

The default toolbox relies on **Orfeo ToolBox (OTB) version 9.1.1**, which is already included in the release. No additional downloads are required.

### Additional tools

There are two additional tools that has no included in the default instalation regarding their further customization. These can be used with the current toolbox, but it requires a few additional steps.

#### SAM 3 segmentation

`Toolbox/sam3_cropmarks.bat` launches a folder-level crop-mark segmentation
tool based on [Meta's SAM 3 image processor](https://github.com/facebookresearch/sam3).

The first step is to create and pack a separate environment from the project root:

```bat
mamba env create -f requirements_sam3.yml
conda-pack -n sam3_cropmarks -o Toolbox/env_sam3.zip
```

SAM 3 currently needs Python 3.12, a recent PyTorch version, and a
CUDA-compatible GPU. Extract `env_sam3.zip` as `Toolbox/env_sam3`.

Before the first run, request access to the `facebook/sam3` checkpoint on Hugging Face and authenticate from the environment:

```bat
Toolbox\env_sam3\Scripts\activate.bat
hf auth login
```

Choose browser login or paste a read-access token. Then download the checkpoint once:

```bash
hf download facebook/sam3 sam3.pt --local-dir C:\models\sam3
```

#### Z-PNN pansharpening

`Toolbox/pansharpening_dl.bat` launches the WorldView/GeoEye deep-learning pansharpening methods included in the bundled [`Z-PNN` repository](https://github.com/matciotola/Z-PNN).

Create and pack its separate environment from the project root in a Miniforge/Miniconda/Anaconda prompt:

```bat
mamba env create -f requirements_pnn.yml
conda-pack -n pansharpen_dl -o Toolbox/env_pnn.zip
```

The PyTorch packages are explicitly taken from the `pytorch` channel while the scientific and GIS stack comes from `conda-forge`. This makes the file work with strict channel priority and avoids accidentally selecting conda-forge's newer CPU-only PyTorch builds. If an older Mamba version still reports an unsatisfiable environment, use flexible priority for this command only:

```bat
mamba env create -f requirements_pnn.yml --channel-priority flexible
```

> Note: This command-line option does not modify the global Conda configuration.

To verify the important imports before packing the environment:

```bat
conda activate pansharpen_dl
python -c "import torch, scipy, skimage, PySimpleGUI; from osgeo import gdal; print('Torch', torch.__version__, 'GDAL', gdal.VersionInfo())"
```

Once the conda environment is created, extract `env_pnn.zip` as `Toolbox/env_pnn`.

Note: The Z-PNN repository is downloaded previously and is already inside the `Toolbox`. It has been modified to match the environment's dependencies.

## Running the toolbox

To launch the toolbox operations:

1. Double-click the desired operation, for example `atmcorr.bat`; or
2. Run it from a Windows command prompt.

Each batch file automatically activates the appropriate environment and routes the operation to the correct backend.

## Supported systems

* Operating system: **Windows 11**
* No administrator privileges required

## Setting up the development environment

The release includes the complete application directory and all required environments.

To reproduce the same structure, follow the steps below in a Miniforge or Miniconda terminal.

Create the environments using the provided `.yml` files:

```bash
mamba env create -f requirements_otb.yml
mamba env create -f requirements_mae.yml
```

Install `conda-pack` in your base Python environment if it is not already installed:

```bash
conda install -c conda-forge conda-pack
```

Package the environments and place the resulting archives in the `Toolbox` directory. These archives will need to be unpacked later:

```bash
conda-pack -n arcpy_otb -o Toolbox/env.zip
conda-pack -n mae -o Toolbox/env_mae.zip
```

Finally, add [Orfeo ToolBox](https://www.orfeo-toolbox.org/) to the `env` directory.

Download **OTB version 9.1.1** and place it in the toolbox directory using the following structure:

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

## Further improvements

* Investigate how to reduce the size of the toolbox environments by including only the required packages.
