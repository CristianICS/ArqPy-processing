"""Classes to handle the WV3 images."""
# Type hints: help static analysis and IDEs.
from __future__ import annotations
from typing import Dict, Any, Optional, Union

from pathlib import Path
from xml.dom import minidom
from osgeo import gdal

import pandas as pd
import numpy as np
import math
import re

ROOT = Path(__file__).absolute().parent.parent

class WV3:

    def __init__(self, image_dir: Path):
        """
        Handle the WV3 images.
        """
        # Choose the starting band for haze computation
        # (for WV3 "Blue" or "Coastal" are recommended)
        self.START_BAND = "C"

        # Store image folders (MUL and PAN)
        self.mul_folder = [i for i in image_dir.glob("*_MUL")][0]
        self.pan_folder = [i for i in image_dir.glob("*_PAN")][0]
        
        # Create VRT from the above folders
        self.mul_path = self._create_vrt(self.mul_folder)
        self.pan_path = self._create_vrt(self.pan_folder)

        # Info about the wavelengths parameters
        # WV3 Spectral Response:
        # https://dg-cms-uploads-production.s3.amazonaws.com/uploads/document/file/105/DigitalGlobe_Spectral_Response_1.pdf
        self.rad_use = pd.read_csv(Path(ROOT, "data/wv3_radiometric_use.csv"))

    def _create_vrt(self, folder):
        """Transform TIL files in VRT"""
        # Translate the TIL mosaics to VRT in order to update the nodata value
        # This is mandatory to compute the stats
        tile_paths = [i for i in folder.glob("*.TIF")]
        # The name of the VRT file will be the same name as the TIL file
        til_path = [i for i in folder.glob("*.TIL")][0]
        vrt_path = Path(til_path.parent, til_path.name.replace(".TIL", ".vrt"))
        # Create the VRT
        if not vrt_path.exists():
            # Build an in-memory VRT mosaic
            # - resolution='highest' keeps the finest pixel size among inputs
            # - srcNodata/VRTNodata set the nodata value
            build_opts = gdal.BuildVRTOptions(
                resolution="highest",
                srcNodata=0,
                VRTNodata=0,
                separate=False # False = mosaic; True = stack as separate bands
            )
            vrt = gdal.BuildVRT(vrt_path, tile_paths, options=build_opts)
            # Clean up
            vrt = None
        
        return vrt_path

    def _find_cloud_cover(self, file_path, pattern="cloudCover"):
        # Load entire file into memory
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Search for the pattern in the text blob
        pos = text.find(pattern)
        if pos == -1:
            return None  # Not found

        # Get line number (count '\n' before the match)
        # +1 for human-readable indexing
        line_number = text.count("\n", 0, pos) + 1
        # Extract the full line (previous and next EOL)
        start = text.rfind("\n", 0, pos) + 1  # beginning of line
        end = text.find("\n", pos)            # end of line

        # Extract cloudCover numeric value using regex
        match_re = re.search(r"cloudCover\s*=\s*([\d\.]+)", text)
        if match_re:
            cloud_cover = float(match_re.group(1))
            return cloud_cover
        else:
            return None

    def _extract_meta(self, img: Path):
        """
        Each image tile is stored inside a folder where its metadata is stored
        """
        # The image subtiles aren't named as the metadata files. Extract the
        # original image name, i.e., the TIL filename
        original_img_til = [i for i in img.parent.glob("*.TIL")]
        assert len(original_img_til) == 1, f"Error saving TIL in {img.parent}"
        original_img_til = str(original_img_til[0])
        imd_path = original_img_til.replace(".TIL", '.IMD')
        # XML metadata file is required in 6S correction.
        xml_path = original_img_til.replace(".TIL", '.XML')
        # TXT metadata file to retrieve cloud cover
        txt_path = [i for i in img.parent.parent.glob("*README.TXT")][0]
        
        self.imd = self._parse_imd(Path(imd_path))
        self.xml = minidom.parse(xml_path)

        self.cloud_cover = self._find_cloud_cover(txt_path)

    def _extract_dn_min(self, img: Path, band_keys):
        """
        Open WV3 image and compute the min ND value in start band.
        """
        start_bidx = [
            i for i, b in enumerate(band_keys) 
            if b == self.START_BAND
        ][0]

        with gdal.Open(img) as ds:
            
            for bidx in range(1, ds.RasterCount + 1):  # GDAL bands are 1-based

                if bidx != start_bidx + 1:
                    continue

                # Extract GDAL band object to avoid write the whole raster
                band = ds.GetRasterBand(bidx)
                # Avoid counting no data values inside the histogram
                band.SetNoDataValue(band.GetNoDataValue())
                
                # GDAL-based tool "statsitics" knows that 0 is nodata
                # stats = ds.statistics(bidx, approx=False)  # rasterio>=1.3
                vmin, vmax, *_ = band.GetStatistics(False, True)

                # Extract the 2nd percentile, better approach that min value
                # Get a list of counts
                counts = band.GetHistogram(
                    vmin,
                    vmax,
                    buckets=4096,
                    include_out_of_range=int(False),
                    approx_ok=int(False))

                # Cumulative counts and target rank
                cum_counts = np.cumsum(counts)
                total = cum_counts[-1]

                # target position in sorted data (2nd percentile)
                # bin-level approximation
                if self.START_BAND == "P":
                    # Extract the minimum DN to avoid nodata in dark pixels
                    target = 0.1 / 100.0 * total
                else:
                    target = 2 / 100.0 * total

                # find first bin where cumulative count >= target
                haze_idx = np.searchsorted(cum_counts, target)

                # bin width and lower edge
                bin_width = (vmax - vmin) / len(counts)
                # bin edges are min + i * bin_width
                return vmin + haze_idx * bin_width

    def _parse_imd(self, imd_path: Path) -> Dict[str, Any]:
        """
        Parse an IMD (key=value) metadata file that may contain 1-level groups:

            BEGIN_GROUP = GROUP_NAME
            KEY = VALUE
            ...
            END_GROUP = GROUP_NAME

        Returns a dictionary with top-level key/value pairs plus nested dicts
        for each group. Also adds `band_keys` with the suffixes of any 
        top-level keys that start with "BAND_".
        """
        metadata: Dict[str, Any] = {}

        current_group: Optional[Dict[str, str]] = None
        group_name: Optional[str] = None

        with imd_path.open(encoding="utf-8", errors="ignore") as src:
            for lineno, raw in enumerate(src, 1):
                # Trim whitespace and ignore blank/comment-ish lines
                line = raw.strip()
                if not line or line.startswith(("#", "//", "<!--")):
                    continue

                # IMD lines sometimes end with a semicolon; strip it safely.
                if line.endswith(";"):
                    line = line[:-1].rstrip()

                # Skip lines that aren't simple KEY=VALUE statements.
                if "=" not in line:
                    # Also skip XML-ish headers if present (e.g., <?xml ...?>)
                    if line.startswith("<") and line.endswith(">"):
                        continue
                    continue

                # Split on the first '=' only (values may contain '=').
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()

                # Handle group boundaries.
                if key == "BEGIN_GROUP":
                    current_group = {}
                    group_name = value
                    continue

                if key == "END_GROUP":
                    # Commit any open group
                    # (be tolerant if END appears without BEGIN).
                    if group_name is not None:
                        metadata[group_name] = current_group or {}
                    current_group = None
                    group_name = None
                    continue

                # Regular key=value: route into current group or top level.
                if current_group is not None:
                    current_group[key] = value
                else:
                    metadata[key] = value

        # If the file ended without a matching END_GROUP,
        # commit the open group.
        if current_group is not None and group_name is not None:
            metadata[group_name] = current_group

        # Extract band suffixes from top-level keys like "BAND_X"
        metadata["band_keys"] = [
            k.split("_", 1)[1]
            for k in metadata.keys()
            if k.startswith("BAND_") and "_" in k
        ]

        return metadata

    def _get_rad_use_prop(self, band_key, prop):
        rad_meta = self.rad_use.query(f"band == '{band_key}'")
        return rad_meta.iloc[0][prop]
    
    def _compute_rayleigh_coeffs(self):
        """
        Compute the scaterring coefficients following Chavez 1988
        
        Important: Wavelengths must be in micrometers (e.g. 0.550)
        """
        # Get the effective center wavelengths in micrometers
        wav_c = {}
        cols = ["lowerBandEdge", "upperBandEdge"]
        for b in self.imd["band_keys"]:
            try:
                band_rad_params = self.rad_use.query(f"band == '{b}'")
                w_min, w_max = band_rad_params[cols].iloc[0].tolist()
            except:
                raise ValueError(f"Current band {b} is not in index.")
            wav_c[b] = np.mean([w_min, w_max])
        
        if hasattr(self, "cloud_cover"):
            if self.cloud_cover < 5:
                r_power = 4
            elif self.cloud_cover >= 5 and self.cloud_cover < 10:
                r_power = 2
            elif self.cloud_cover >= 10 and self.cloud_cover < 30:
                r_power = 1
            elif self.cloud_cover >= 30 and self.cloud_cover < 50:
                r_power = 0.7
            elif self.cloud_cover >= 50:
                r_power = 0.5
            
        else:
            r_power = 4
        
        s = {b: (wav_c[self.START_BAND] / wav_c[b])**r_power for b in wav_c}
        return s

    def extract_params_per_band(self, img: Path):
        
        self._extract_meta(img)
        lamb_scatter = self._compute_rayleigh_coeffs()
        # Get the dark object DN value (start band)
        min_dn_value = self._extract_dn_min(img, self.imd["band_keys"])

        band_params = []
        
        for i, band in enumerate(self.imd["band_keys"]):
            # Return image metadata by band
            band_meta = self.imd['BAND_'+band]

            # Construct the dict with the band metadata
            band_dict = {
                "name": band,
                "gain": self._get_rad_use_prop(band, 'GAIN_2015v2'),
                "offset": self._get_rad_use_prop(band, 'OFFSET_2015v2'),
                "e_thuillier": self._get_rad_use_prop(band, 'E_Thuillier'),
                "abs_factor": band_meta['absCalFactor'],
                "effective_band_width": band_meta["effectiveBandwidth"],
                "lowerBandEdge": self._get_rad_use_prop(band, "lowerBandEdge"),
                "upperBandEdge": self._get_rad_use_prop(band, "upperBandEdge"),
                "lamb_scatter": lamb_scatter[band],
                "dark_object_dn": min_dn_value
            }
            
            band_params.append(band_dict)
        
        return band_params
    
    def extract_ctx_params(self, img: Path):
        
        self._extract_meta(img)

        # Retrieve the scene BBOX
        # Its coordinates are in EPSG:4326
        coords = []
        # for i in ['NWLONG','NWLAT','SELONG','SELAT']:
        for i in ['ULLON','ULLAT','LRLON','LRLAT']:
            coord = self.xml.getElementsByTagName(i)[0].firstChild.data
            coords.append(float(coord))


        # Complement of the solar elevation angle
        sun_zen_angle = round(90 - float(self.imd['IMAGE_1']['meanSunEl']), 2)
        return {
            'capture_date': self.imd['MAP_PROJECTED_PRODUCT']['earliestAcqTime'],
            'sun_zen_ang': sun_zen_angle,
            'meanSatAz': float(self.imd['IMAGE_1']['meanSatAz']),
            'meanSunAz': float(self.imd['IMAGE_1']['meanSunAz']),
            'bbox': coords
        }
    
    def get_images(self):
        return {'mul': self.mul_path, 'pan': self.pan_path}

