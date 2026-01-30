from osgeo_utils.gdal_calc import main as gdal_calc
from importlib.resources import files
from pathlib import Path
from osgeo import gdal
import re
import json

gdal.DontUseExceptions()

class Indices:
    
    indices_path = files("spectral_indices.data") / "indices.json"

    def check_bands(bands_dict: dict, index_bands: list) -> bool:
        """
        Test if the index bands are inside image bands.
        """
        # for b in index_bands:
        #     if b not in img_band_keys:
        #         return False
        missing = [nm for nm in index_bands if nm not in bands_dict]
        if missing:
            return missing
        else:
            return None

    def get_index_keys():
        """Return a list with all the available keys."""
        indices_json = open(Indices.indices_path)
        indices_json = json.load(indices_json)
        return indices_json.keys()

    def compute_index(
            index_key: str,
            img_path: str|Path,
            img_band_pos: list[int],
            img_band_names: list[str],
            out_dir: str|Path,
            clip_layer_path: str|Path|bool = False,
            overwrite = False,
            creation_opts = ["COMPRESS=DEFLATE", "PREDICTOR=3"]
        ):
        """
        Write and perform gdal_calc command in cmd to compute a spectral
        index.

        :indices_path: Path to the json file with indices metadata.
        :index_key: Index short name. Must be the same as its JSON's key.
        :img_path: Image path which bands will use to compute the indices.
        :img_band_pos: Position of image bands linked to img_band_names.
        :img_band_names: Image band keys (e.g. R)
        :out_dir: Directory to store the computed index.
        :clip_layer_path: An optional path with a vector layer to clip the 
            index.
        :overwrite: Recompute an index if it exists in out_dir.
        :creation_opts: GDAL creation options.
        """
        # Extract the nodata value
        ds = gdal.Open(img_path, gdal.GA_ReadOnly)
        nodata = ds.GetRasterBand(1).GetNoDataValue()
        ds = None

        # Open index metadata's JSON
        indices_json = open(Indices.indices_path)
        indices_json = json.load(indices_json)

        # Make a name: band mapping from the two aligned lists
        if len(img_band_pos) != len(img_band_names):
            raise ValueError(
                "band_positions and band_names must be the same length")
        name_to_band = {
            nm: pos for nm, pos in zip(img_band_names, img_band_pos)}

        # Get index parameters
        index_bands = indices_json[index_key]['bands']
        formula = str(indices_json[index_key]['formula'])

        missing = Indices.check_bands(name_to_band, index_bands)
        if missing is not None:
            # raise KeyError
            print(f"Required names not found in band_names: {missing}")
            return

        # Discover the variable order by "first appearance" in the formula
        # Consider names declared as required and match them with word
        # boundaries. Short names (like 'R') don't accidentally match inside 
        # longer names (like 'RE1').
        regexp_str = r"\b(" + "|".join(map(re.escape, index_bands)) + r")\b"
        # Single regex pattern that matches any of the required band names.
        # Example if index_bands = ['N', 'R'], the pattern becomes:  \b(N|R)\b
        pattern = re.compile(regexp_str)
        # Empty list to store the variable names in the order they appear for
        #  the first time in the formula.
        seen_order = []
        # Iterate through every match of the band name in the formula string.
        # pattern.finditer(formula) returns match objects in sequence from
        # left to right.
        for m in pattern.finditer(formula):
            nm = m.group(1) # Extract the matched band name (e.g., "N" or "R")
            # Avoid duplicates if the same name appears again
            if nm not in seen_order:
                seen_order.append(nm) # Preserve the "first occurrence" order

        # If none were found (e.g., user gave required_names but formula doesn't use them)
        if not seen_order:
            print("No required band names were found in the formula text.")
            return

        if len(seen_order) > 26:
            raise ValueError("gdal_calc supports at most 26 variables (A..Z).")

        # Map names -> letters (A,B,C,…)
        letters = [chr(ord('A') + i) for i in range(len(seen_order))]
        name_to_letter = dict(zip(seen_order, letters))

        # Rewrite the formula safely
        # Whole-word replaces, longest names first is optional
        def sub_one(match: re.Match) -> str:
            return name_to_letter[match.group(0)]
        letter_formula = pattern.sub(sub_one, formula)

        # Construct argv
        argv = ["gdal_calc.py"]
        for nm in seen_order:
            L = name_to_letter[nm]
            band_idx = name_to_band[nm]  # 1-based
            argv += [f"-{L}", str(img_path), f"--{L}_band", str(band_idx)]

        if clip_layer_path:
            img_suffix = "_temp"
        else:
            img_suffix = ""

        out_path = Path(
            out_dir,
            f"{index_key}_{img_path.stem}{img_suffix}.tif"
        )

        argv += [
            # Integer constant
            "--calc", letter_formula,
            "--type", "Float32",
            "--NoDataValue", f"{nodata}",
            "--outfile", str(out_path)
        ]
        if overwrite:
            argv.append("--overwrite")
        else:
            if out_path.exists():
                return

        for co in creation_opts:
            argv += ["--co", co]

        print("formula ->", formula)
        print("name -> GDAL letter:", name_to_letter)
        print("rewritten formula:", letter_formula)
        # print("argv:", " ".join(argv))

        gdal_calc(argv)

        if clip_layer_path:
            co = [
                "COMPRESS=DEFLATE",
                "PREDICTOR=3", # Better for floats
                "BIGTIFF=IF_SAFER"
            ]
            out_clip_path = out_path.with_stem(
                out_path.stem.replace(img_suffix, "")
            )
            gdal.Warp(
                out_clip_path,              # dst
                out_path,                   # src
                cutlineDSName=clip_layer_path,
                cropToCutline=True,
                dstNodata=nodata,
                multithread=True,
                format = "COG",
                creationOptions=co,
                warpOptions=["NUM_THREADS=ALL_CPUS"]
            )
            out_path.unlink()