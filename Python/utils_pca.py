"""____________________________________________________________________________
Script Name:        pca.py
Description:        Functions to compute PCA and export the image.
Prerequisites:      GDAL version "3.1.4" or greater
____________________________________________________________________________"""
# 0. Import packages
# =============================================================================
from pathlib import Path
import os
import csv
import itertools

# Brute-force Orfeo ToolBox
OTB = r"C:\OTB-9.1.1-Win64"
py  = os.path.join(OTB, "lib", "otb", "python")
bin = os.path.join(OTB, "bin")
lib = os.path.join(OTB, "lib")

# Help the loader (even if PATH is correct)
os.add_dll_directory(bin)
os.add_dll_directory(lib)

import otbApplication as otb

class PCA:
    
    def combis(band_names: list, band_keys: list, combi_path: str) -> dict:
        """
        Store all possible combinations between image bands and export a CSV:
        - id:        Unique combi identifier
        - bands:     Acronyms/names of bands in the combi separated by '-'
        - positions: Positions (indexes) of those bands separated by '-'

        :band_names: List with band names/prefixes (e.g., ["B2","B3","B4"...]).
        :band_keys:  List with band positions aligned to band_names.
        :combi_path: File path to save a CSV with all combinations.

        Returns
        -------
        dict:
            {
                1: {'names': [...], 'pos': [...]},
                2: {'names': [...], 'pos': [...]},
                ...
            }
        """
        if len(band_names) != len(band_keys):
            raise ValueError(
                "band_names and band_keys must have the same length.")

        # Map each band name to its position
        name_to_pos = dict(zip(band_names, band_keys))

        # Build all combinations (size 3..len-1) + the full set
        # n_combis = list(range(3, len(band_names)))
        band_combis = []
        # Calculate all combinations
        for n in range(3, len(band_names)):
            band_combis.extend(itertools.combinations(band_names, n))

        # The last combination integrates all the image bands
        band_combis.append(band_names)

        # Build final dict: {id: {'names': [...], 'pos': [...]} }
        combis_dict = {}
        for idx, combo in enumerate(band_combis, start=1):
            combis_dict[idx] = {
                'names': list(combo),
                'pos': [name_to_pos[name] for name in combo]
            }

        # Write CSV with names and positions
        with open(combi_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id', 'bands', 'positions'])  # header
            for idx, item in combis_dict.items():
                bands_str = '-'.join(item['names'])
                pos_str = '-'.join(str(p) for p in item['pos'])
                writer.writerow([idx, bands_str, pos_str])

        return combis_dict

    def run(
        img_path,
        out_path,
        bands,
        normalize=False,
        outmatrix=None,
        rescale_min=None,
        rescale_max=None,
        ram=None,
        nbcomp=None,  # optional: override number of components if you don't want len(bands)
    ):
        """
        Run PCA (OTB DimensionalityReduction) on a subset of image bands.

        Parameters
        ----------
        img_path : str | PathLike
            Input multiband image.
        out_path : str | PathLike
            Output PCA image.
        bands : Sequence[int]
            1-based band indices to use for PCA (e.g., [1, 3, 5]).
        normalize : bool, default False
            Center & reduce (z-score) before PCA. (OTB's -normalize)
        outmatrix : str | None, default None
            If given, saves the transformation (eigenvectors) matrix to this file.
        rescale_min, rescale_max : float | None
            If both provided, rescale PCA output to this min/max (OTB's -rescale minmax).
        ram : int | None
            Available RAM in MB for OTB streaming.
        nbcomp : int | None
            Number of principal components to keep; defaults to len(bands).
        """

        if not bands or len(bands) == 0:
            raise ValueError("`bands` must be a non-empty list of 1-based band indices.")

        # Pre-select the desired input bands with BandMathX
        # Expression uses the BandMathX 'bands()' function to build a new 
        # vector image from selected channels (1-based indices).
        bm = otb.Registry.CreateApplication("BandMathX")
        bm.SetParameterStringList("il", [str(img_path)])
        bm_expr = f"bands(im1, {{{','.join(str(b) for b in bands)}}})"
        bm.SetParameterString("exp", bm_expr)

        # Execute the selector
        # (no file I/O; we'll connect its output downstream)
        bm.Execute()

        # 2) PCA with DimensionalityReduction
        pca = otb.Registry.CreateApplication("DimensionalityReduction")
        pca.SetParameterInputImage("in", bm.GetParameterOutputImage("out"))
        pca.SetParameterString("out", str(out_path))
        pca.SetParameterString("method", "pca")

        # Number of components: default to the number of selected bands
        if nbcomp is None:
            nbcomp = len(bands)
        pca.SetParameterInt("nbcomp", int(nbcomp))

        # Normalize (center+reduce) if requested
        if normalize:
            pca.SetParameterInt("normalize", 1)  # bool flag in OTB

        # Optional: export transformation matrix (eigenvectors)
        if outmatrix:
            pca.SetParameterString("outmatrix", str(outmatrix))

        # Optional: rescale to [min, max] range
        if (rescale_min is not None) and (rescale_max is not None):
            pca.SetParameterString("rescale", "minmax")
            pca.SetParameterFloat("rescale.minmax.outmin", float(rescale_min))
            pca.SetParameterFloat("rescale.minmax.outmax", float(rescale_max))

        # RAM for streaming
        if ram is not None:
            pca.SetParameterInt("ram", int(ram))

        # Execute the final app and write output
        pca.ExecuteAndWriteOutput()

        if outmatrix:
            print(f"Transformation matrix saved to: {outmatrix}")







