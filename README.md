# ArcPy Toolbox

This toolbox generates a series of products derived from remote sensing images for archaeological analysis, with a particular focus on crop mark detection. It also includes an option to apply a Masked Autoencoder (MAE) to identify zones with a higher probability of containing crop marks.

The toolbox is distributed as a self-contained package and **does not require Conda, Miniforge, or manual environment setup by the end user**.

The available operations should be executed in the following recommended order:

1. `atmcorr` – atmospheric correction
2. `pansharpening` – spatial–spectral fusion
3. `pca` – principal component analysis
4. `spectral_indices` – spectral index computation
5. `highpass` – spatial filtering
6. `mae` – Masked Autoencoder-based analysis

Each operation provides a graphical user interface for configuring the parameters required for execution.

## Available sensors

Atmospheric correction is currently supported for the following sensors:

* GEOSAT
* WorldView-3 (WV3)
* WorldView LEGION-06

The `pca`, `pansharpening`, and `spectral_indices` operations are available only for WV3 and LEGION sensors.

The `highpass` and `mae` operations are available for any image provided in GeoTIFF format.

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
|- atmcorr.bat
|- ...
|- README.md
```

## External dependencies

The toolbox relies on **Orfeo ToolBox (OTB) version 9.1.1**, which is already included in the release. No additional downloads are required.

## Running the toolbox

To launch the toolbox operations:

1. Double-click the desired operation, for example `atmcorr.bat`; or
2. Run it from a Windows command prompt.

Each batch file automatically activates the appropriate environment and routes the operation to the correct backend.

## Supported systems

* Operating system: **Windows 11**
* No administrator privileges required

## Known issues

### Pansharpening problem

Running the Bayes pansharpening technique with original reflectance values and the default `lambda` parameter may produce the following issue:

```text
All pansharpened bands have nearly identical min, max, mean, and standard deviation.
```

This is a strong signal that the multispectral spectral information is being overwhelmed.

The PAN statistics show valid reflectance-scale values:

```text
PAN: 0–0.731
mean: 0.078
nodata: -9999
```

However, the pansharpened output may show values such as:

```text
output min:  about -1300
output max:  about 300
output mean: about 2.4
```

The most likely cause is that OTB Bayesian pansharpening is not stable with reflectance-scale inputs when using the current Bayesian parameters.

More specifically, the default `lambda = 0.995` is probably too high for data in the `0–1` range.

#### Why this can happen

The Bayesian method does not only inject spatial detail. It solves an optimization problem involving the PAN image, the multispectral image, and a regularization term.

With reflectance-scale values, the numerical magnitudes are small:

```text
0.02
0.08
0.30
```

If the method internally behaves better with image-like digital numbers, for example:

```text
200
800
3000
```

then the same `lambda` value can behave very differently.

With the default value:

```text
-method.bayes.lambda 0.995
```

the model can become numerically dominated by the PAN constraint. As a result, the output bands may collapse toward the same PAN-driven structure, and the solver may produce large positive and negative artifacts.

#### Recommendations

For archaeological crop mark detection, it is recommended to work with integer reflectance values scaled by a factor of `10000`.

Use this scaling factor inside the `atmcorr.bat` tool.

If you want to work with the original reflectance values in the `0–1` range, reduce the `lambda` value when using the Bayesian pansharpening method. A value around `0.1` is a reasonable starting point


## Environment construction for developers

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
