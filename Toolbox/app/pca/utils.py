from pathlib import Path
import csv
import itertools
import subprocess

from core.env import find_otb_executable


def load_combis_from_csv(combi_path: str | Path) -> dict:
    """
    Load combinations from a CSV created by generate_combis() and
    reconstruct the original dictionary format.

    Parameters
    ----------
    combi_path : str
        Path to the CSV file.

    Returns
    -------
    dict
        {
            1: {'names': [...], 'pos': [...]},
            2: {'names': [...], 'pos': [...]},
            ...
        }

    Raises
    ------
    ValueError
        If the CSV is missing, malformed, or does not contain the expected
        id/bands/positions columns.
    """
    combis_dict = {}

    with open(str(combi_path), "r", newline="") as f:
        try:
            reader = csv.DictReader(f)

            for row in reader:
                idx = int(row["id"])

                names = row["bands"].split("-")
                pos = [int(p) for p in row["positions"].split("-")]

                combis_dict[idx] = {
                    "names": names,
                    "pos": pos
                }

        except (csv.Error, KeyError, ValueError) as e:
            raise ValueError(
                f"Malformed PCA combinations CSV: {combi_path}"
            ) from e

    return combis_dict


def generate_combis(band_names: list, band_keys: list, combi_path: str) -> dict:
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


def run_pca(
    img_path,
    out_path,
    bands,
    normalize=True,
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

    otb_exe = find_otb_executable()

    # Select bands for the current PCA
    # Store the selected band in a temporary file
    temp_sel = out_path.with_suffix(".sel.tif")

    fun_params = [
        otb_exe,
        "BandMathX",
        "-il", str(img_path),
        "-exp", f"bands(im1, {{{','.join(str(b) for b in bands)}}})",
        "-out", str(temp_sel)
    ]

    subprocess.run(fun_params, check=True)

    try:
        # PCA with DimensionalityReduction
        fun_params = [
            otb_exe,
            "DimensionalityReduction",
            "-in", str(temp_sel),
            "-out", str(out_path),
            "-method", "pca",
        ]

        if nbcomp is None:
            # Number of components: default to the number of selected bands
            fun_params.extend(["-nbcomp", f"{int(len(bands))}"])
        else:
            fun_params.extend(["-nbcomp", f"{int(nbcomp)}"])

        if normalize:
            # Normalize (center+reduce)
            fun_params.extend(["normalize", str(1)])

        # Optional: export transformation matrix (eigenvectors)
        if outmatrix:
            fun_params.extend(["outmatrix", str(outmatrix)])

        # Optional: rescale to [min, max] range
        if (rescale_min is not None) and (rescale_max is not None):
            fun_params += [
                "rescale", "minmax",
                "rescale.minmax.outmin", str(float(rescale_min)),
                "rescale.minmax.outmax", str(float(rescale_max))
            ]

        # RAM for streaming
        if ram is not None:
            fun_params.extend(["ram", str(int(ram))])

        subprocess.run(fun_params, check=True)
    finally:
        temp_sel.unlink(missing_ok=True)

    if outmatrix:
        print(f"Transformation matrix saved to: {outmatrix}")
    print(f"PCA image saved to: {out_path}")
