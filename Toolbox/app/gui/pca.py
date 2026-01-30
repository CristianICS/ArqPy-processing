from pathlib import Path
from osgeo import gdal
from pca import PCA

import subprocess
import PySimpleGUI as sg

gdal.DontUseExceptions()

def run_job(values):
    """Perform Principal Components Analysis between all image bands."""
    try:
        # Check input MUL and PAN images
        img_path = Path(values["-IMG-"])
        if not img_path.exists():
            return FileNotFoundError(str(img_path))
        
        if not values["-OUT-"]:
            out_folder = img_path.parent.parent / "PCA"
        else:
            out_folder = Path(values["-OUT-"])

        if not out_folder.exists():
            if not values["-OUT-"]:
                out_folder.mkdir(exist_ok=True)
            else:
                return FileNotFoundError(str(out_folder))

        clip_path = Path(values.get("-CLIP_VECTOR-", ""))
        if not clip_path.exists():
                sg.popup_ok("The vector folder does not exist.\
                             Please switch it to a valid one.")
        if clip_path == Path(""):
            clip_path = False
        else:
            # Extract the nodata value
            ds = gdal.Open(img_path, gdal.GA_ReadOnly)
            nodata = ds.GetRasterBand(1).GetNoDataValue()
            ds = None

        if values["-SENSOR-"] == "WV3":
            band_pos = [1, 2, 3, 4, 5, 6, 7, 8]
            band_keys = ["C", "B", "G", "Y", "R", "RE1", "N", "N2"]
        elif values["-SENSOR-"] == "LEGION":
            band_pos = [1, 2, 3, 4, 5, 6, 7, 8]
            band_keys = ["C", "B", "G", "Y", "R", "RE1", "RE2", "N"]

        combis_path = Path(values.get("-COMBIS_PATH-", ""))
        if not combis_path.exists():
            sg.popup_ok(
                "The provided file with PCA combinations does not exist. Please switch it to a valid one."
            )
        if combis_path != Path(""):
            combis_dict = PCA.load_combis_from_csv(combis_path)
            if combis_dict is None:
                return ValueError(" ".join([
                    "Current path with PCA combinations does not contain",
                    "valid data. Autogenerate a combis file to see",
                    "a valid one."
                ]))
        else:
            combis_dict = PCA.combis(
                band_keys,
                band_pos,
                out_folder / 'band_combinations.csv'
            )
        
        for i, item in combis_dict.items():
            
            out_stem = f"COMB{i}"
            out_name = f"{out_stem}.tif"
            out_name_temp = f"{out_stem}_temp.tif"
            
            if clip_path:
                out_name = f"{out_stem}_toclip.tif"
                out_clip_name  = f"{out_stem}.tif"
                out_clip_path = out_folder / out_clip_name

            out_path = out_folder / out_name
            out_temp_path = out_folder / out_name_temp
            
            # Check if the combi has already been created (as temp or as cog)
            if out_path.exists() or out_temp_path.exists():
                continue
            print(f"Computing PCA {i}")

            PCA.run(img_path, out_temp_path, item["pos"], ram=2048)

            # Translate to COG
            subprocess.run([
                "gdal_translate",
                str(out_temp_path),
                str(out_path),
                # Match the OTB fillnodata value
                "-a_nodata", "0",
                "-of", "COG",
                "-co", "COMPRESS=DEFLATE",
                "-co", "PREDICTOR=2",
                "-co", "BIGTIFF=IF_SAFER"
            ], check=True)

            out_temp_path.unlink()

            if clip_path:
                
                co = [
                    "COMPRESS=DEFLATE",
                    "PREDICTOR=3", # Better for floats
                    "BIGTIFF=IF_SAFER"
                ]

                gdal.Warp(
                    out_clip_path,              # dst
                    out_path,                   # src
                    cutlineDSName=clip_path,
                    cropToCutline=True,
                    dstNodata=nodata,
                    multithread=True,
                    format = "COG",
                    creationOptions=co,
                    warpOptions=["NUM_THREADS=ALL_CPUS"]
                )
                out_path.unlink()

    except Exception as e:
        return e

def main():
    sg.theme('VSCodeDark')

    title = [sg.Text("Compute PCA", font=("Arial", 16, "bold"))]

    layout = [
        title,

        # Input and output placeholders
        [sg.Text("Input image:")],
        [sg.Input(key="-IMG-"), sg.FileBrowse()],
        [sg.Text("Select folder to store the results:")],
        [sg.Input(key="-OUT-"), sg.FolderBrowse()],
        [sg.Text("Leave it blank if you want to autocreate it", font=("Arial", 10, "italic"))],
        [sg.Text("Select the CSV file containing the PCA combinations:")],
        [sg.Text("Leave it blank to autocreate it", font=("Arial", 10, "italic"))],
        [sg.Input(key="-COMBIS_PATH-"), sg.FileBrowse(
            file_types=(
                ("Combinations files", "*.csv"),
                ("All files", "*.*"),
            )
        )],

        # Clip operation
        # Vector layer path
        [sg.Text("Vector layer to perform a clip operation:")],
        [sg.Text("Leave it blank to skip", font=("Arial", 10, "italic"))],
        [
            sg.Input(key="-CLIP_VECTOR-",),
            sg.FileBrowse(
                file_types=(
                    ("Vector files", "*.shp;*.gpkg;*.geojson;*.kml"),
                    ("All files", "*.*"),
                )
            )
        ],

        [sg.Text("Sensor"), sg.Combo(
            ["WV3", "LEGION"],
            key="-SENSOR-",
            default_value="WV3",
            readonly=True,
            size=(20, 1)
        )],
        [sg.Button("Run"), sg.Button("Exit")],
        [sg.Output(size=(80, 20))]
    ]

    window = sg.Window("Pansharpening", layout)

    while True:
        event, values = window.read()


        if event in (sg.WINDOW_CLOSED, "Exit"):
            break
        if event == "Run":
            
            if not values["-IMG-"]:
                sg.popup_ok(f"Please select a valid image.")
                continue

            # Disable Run button and show a status message
            window["Run"].update(disabled=True)
            print("Starting pansharpening operation...")

            # Launch long operation in a worker thread
            window.perform_long_operation(
                lambda: run_job(values),
                "-JOB-DONE-",
            )

        elif event == "-JOB-DONE-":

            # Re-enable Run button
            window["Run"].update(disabled=False)

            result = values["-JOB-DONE-"]

            if isinstance(result, FileNotFoundError):
                sg.popup_ok("FileNotFoundError.", str(result))
                print(f"Error: {result}")
            elif isinstance(result, Exception):
                sg.popup_ok("An error occurred:", str(result))
                print(f"Error: {result}")
            elif isinstance(result, RuntimeError):
                sg.popup_ok("Error:", str(result))
                print(f"Error: {result}")
            else:
                sg.popup_ok("Processing completed successfully.")
                print("Processing completed successfully.")

    window.close()
