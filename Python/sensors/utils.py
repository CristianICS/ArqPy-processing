import csv

def load_radiometric_csv(csv_path, index_column):
    """
    Load the band calibration CSV and return a dictionary indexed by the
    'band' field.

    Example return structure:
    {
        "P": {"GAIN_2015v2": 0.923, "OFFSET_2015v2": -1.7, ...},
        "C": {...},
        ...
    }
    """
    data = {}

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            clean = {}

            for key, value in row.items():
                key = key.strip().strip('"')
                val = value.strip().strip('"')

                # Try to convert numeric values to float
                try:
                    val = float(val)
                except ValueError:
                    pass

                clean[key] = val

            band = clean.pop(index_column)  # remove and use as key
            data[band] = clean

    return data

def get_properties(band_dict, band, *props):
    """
    Retrieve any number of properties by name for a given band.
    Example:
      get_properties(dict, "N", "GAIN_2015v2", "OFFSET_2015v2", "E_Thuillier")
    """
    if band not in band_dict:
        raise KeyError(f"Band '{band}' not found.")

    data = band_dict[band]

    return tuple(data[p] for p in props)
