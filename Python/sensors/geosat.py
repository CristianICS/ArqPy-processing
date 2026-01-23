"""Classes to handle the World View LEGION constellation images."""
from __future__ import annotations
from importlib.resources import files

from pathlib import Path
from xml.dom import minidom

from .utils import load_radiometric_csv, get_properties
from atmo_correction.utils import BandParams, CtxParams

import numpy as np

class Geosat:

    def __init__(self, img_path: Path, start_band=4):
        """
        Handle GEOSAT L1C ORTHO images.

        1: NIR
        2: RED
        3: GREEN
        4: BLUE
        """
        if not Path(img_path).exists():
            raise FileNotFoundError(f"The GEOSAT image {img_path} does not exist.")

        self.path = img_path

        # Store metadata files
        dim_path = self.path.with_suffix(".dim")
        self.dim = minidom.parse(str(dim_path))

        # Choose the starting band for haze computation (default Blue)
        self.START_BAND = start_band

        self.cloud_cover = self._get_quality_prop('CLOUD_COVER_PERCENTAGE')

        rad_use_path = files("sensors.data") / "geosat_radiometric_use.csv"
        self.rad_use = load_radiometric_csv(rad_use_path, "band_index")

        # Init the dictionary with all the metadata by band
        self._init_band_metadata()
        self._get_band_params('Band_Statistics')
        self._get_band_params('Spectral_Band_Info')
        self._extract_dn_min()

    def _init_band_metadata(self):
        """Get image band indexes and initialize a dictionary with them."""
        self.band_metadata = {}
        qp = self.dim.getElementsByTagName('Spectral_Band_Info')
        for child in qp:
            b_key = child.getElementsByTagName('BAND_INDEX')[0].firstChild.data
            self.band_metadata[b_key] = {}

    def _get_band_params(self, xml_key):
        """
        Include band params to the dict with band statistics
        
        :xml_key: The key of the DIM file section with target parameters.
        """
        for elem in self.dim.getElementsByTagName(xml_key):
            stats = {
                x.tagName: x.childNodes[0].data
                for x in elem.childNodes
                if x.nodeType == minidom.Node.ELEMENT_NODE
            }
            bindex = stats.pop("BAND_INDEX")
            self.band_metadata[int(bindex)].update(stats)

    def _get_quality_prop(self, key='SATELLITE_AZIMUTH'):
        """Retrieve a value from a key inside Quality Parameter's section."""
        qp = self.dim.getElementsByTagName('Quality_Parameter')
        for child in qp:
            qp_desc = child.getElementsByTagName('QUALITY_PARAMETER_DESC')
            # Get the key of the first child
            i_key = qp_desc[0].firstChild.data
            if i_key == key:
                values = child.getElementsByTagName('QUALITY_PARAMETER_VALUE')
                if values[0].firstChild is not None:
                    value = values[0].firstChild.data

                return float(value)
        # When the value has not been found, return None
        return None

    def _get_main_prop(self, keys: list):
        """
        Get values from DIM file by key.

        It's only for unique keys (appears 1 in the XML tree).

        Note: Integers values are returned as strings.
        """
        params = {}
        for key in keys:
            val = self.dim.getElementsByTagName(key)[0].firstChild.data
            params[key] = val

        return params
    
    def _get_bbox(self):
        """
        Retrieve BBOX coords.

        The first 4 FRAME_LON and FRAME_LAT are the BBOX
        NW, NE, SE, SW coordinates in WGS84 (GEE accepted EPSG)

        Coordinates must be inserted into GEE with this format:
        ['NWLONG','NWLAT','SELONG','SELAT']
        """
        coords_long = self.dim.getElementsByTagName('FRAME_LON')
        coords_lat = self.dim.getElementsByTagName('FRAME_LAT')
        
        # Retrieve NWLONG and NWLAT
        nwlong = float(coords_long[0].firstChild.data)
        nwlat = float(coords_lat[0].firstChild.data)
        # Retrieve SELONG and SELAT
        selong = float(coords_long[2].firstChild.data)
        selat = float(coords_lat[2].firstChild.data)

        return([nwlong, nwlat, selong, selat])

    def _extract_dn_min(self):
        """Get image bands min values (GDAL is required)"""
        for band in self.band_metadata.keys():
            dn_min = float(self.band_metadata[band]['STX_MIN'])
            # Handle errors
            if dn_min > 0:
                self.band_metadata[band].update({'dn_min': dn_min})
            else:
                self.band_metadata[band].update({'dn_min': 1})

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
                w_min, w_max = get_properties(self.rad_use, b, *cols)
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

        else:
            r_power = 4
        
        s = {b: (wav_c[self.START_BAND] / wav_c[b])**r_power for b in wav_c}
        return s

    def extract_params_per_band(self):

        lamb_scatter = self._compute_rayleigh_coeffs()

        band_params = []
        
        for i, band in enumerate(self.band_metadata.keys()):
            # Return image metadata by band
            band_meta = self.band_metadata[band]

            # Construct the dict with the band metadata
            params = BandParams(
                name=band,
                gain=band_meta['PHYSICAL_GAIN'],
                offset=band_meta['PHYSICAL_BIAS'],
                e_thuillier=band_meta['ESUN'],
                lowerBandEdge=self.rad_use[band]["lowerBandEdge"],
                upperBandEdge=self.rad_use[band]["upperBandEdge"],
                lamb_scatter=lamb_scatter,
                dark_object_dn=band_meta["dn_min"]
            )
            
            band_params.append(params)
        
        return band_params
    
    def extract_ctx_params(self):
        
        # Retrieve the scene BBOX
        # Its coordinates are in EPSG:4326
        coords = self._get_bbox()

        props_dict = self._get_main_prop(
            ["SUN_ELEVATION", "SUN_AZIMUTH", "EARTH_SUN_DISTANCE"])
        sat_az = self._get_quality_prop('SATELLITE_AZIMUTH')

        # Complement of the solar elevation angle
        sun_zen_angle = round(90 - float(props_dict["SUN_ELEVATION"]), 2)

        return CtxParams(
            d=float(props_dict["EARTH_SUN_DISTANCE"]),
            sun_zen_ang=sun_zen_angle,
            mean_sat_az=float(sat_az),
            mean_sun_az=float(props_dict["SUN_AZUMUTH"]),
            bbox=coords
        )

    @staticmethod
    def radiometric_calibration(dn: np.ndarray, band: BandParams):
        """Bound method to perform the ARC operation before atm correction."""
        dn = dn.astype("float32", copy=False)
        return band.gain * dn + band.offset
