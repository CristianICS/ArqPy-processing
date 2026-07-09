from pathlib import Path
from osgeo import gdal
from pansharp import SENSORS
from pansharp import brovey, bayesian

import PySimpleGUI as sg

gdal.UseExceptions()

def run_job(values):
    """Perform a pansharpening operation."""
    try:
        # Check input MUL and PAN images
        mul_img_path = Path(values["-MUL-"])
        if not mul_img_path.exists():
            return FileNotFoundError(mul_img_path)
        
        pan_img_path = Path(values["-PAN-"])
        if not pan_img_path.exists():
            return FileNotFoundError(pan_img_path)
        
        clip_path = Path(values.get("-CLIP_VECTOR-", ""))
        if not clip_path.exists():
                sg.popup_ok("The vector folder does not exist.\
                             Please switch it to a valid one.")
        if clip_path == Path(""):
            img_suffix = ""
        else:
            # Append a suffix to the pansharpen image to overwrite it
            img_suffix = "_temp"

        out_folder = Path(values["-OUT-"])
        if not out_folder.exists():
            out_folder.mkdir(exist_ok=True)
        
        method = values["-ALGORITHM-"]
        if method == "Bayesian":
            bayes_lambda = float(values["-LAMBDA-"])

            print("Starting Bayesian pansharpening...")
            out_img_name = mul_img_path.name.replace(
                ".tif",
                f"_bayes{img_suffix}.tif"
            )
            out_img_path = out_folder / out_img_name
            bayesian(mul_img_path, pan_img_path, out_img_path, bayes_lambda)

        else:
            # Perform weighted Brovey pansharpening
            out_img_name = mul_img_path.name.replace(
                ".tif", 
                f"_brovey{img_suffix}.tif"
            )
            out_img_path = out_folder / out_img_name
            brovey(
                SENSORS[values["-SENSOR-"]],
                mul_img_path, pan_img_path, out_img_path)

            if values["-SENSOR-"] == "WV3":
                lmnv_path = SENSORS[values["-SENSOR-"]].lmnv(
                    mul_img_path, pan_img_path)
                SENSORS[values["-SENSOR-"]].merge_pansharpened_bands(
                    out_img_path, lmnv_path
                )

        if clip_path != Path(""):
            print("Clipping...")
            co = [
                "COMPRESS=DEFLATE",
                "PREDICTOR=2",
                "BIGTIFF=IF_SAFER"
            ]
            out_clip_path = out_img_path.with_stem(
                out_img_path.stem.replace(img_suffix, "")
            )
            gdal.Warp(
                out_clip_path,              # dst
                out_img_path,                   # src
                cutlineDSName=clip_path,
                cropToCutline=True,
                dstNodata=-9999,
                multithread=True,
                format = "COG",
                creationOptions=co,
                warpOptions=["NUM_THREADS=ALL_CPUS"]
            )
            out_img_path.unlink()

    except Exception as e:
        return e

def main():
    sg.theme('VSCodeDark')

    title = [sg.Text("Pansharpening Tool", font=("Arial", 16, "bold"))]

    layout = [
        title,

        # Input and output placeholders
        [sg.Text("Input MUL Image:")],
        [sg.Input(key="-MUL-"), sg.FileBrowse()],
        [sg.Text("Input PAN Image:")],
        [sg.Input(key="-PAN-"), sg.FileBrowse()],
        [sg.Text("Output Folder:")],
        [sg.Text("Leave it blank if you want to autocreate it", font=("Arial", 10, "italic"))],
        [sg.Input(key="-OUT-"),sg.FolderBrowse()],
        
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
            list(SENSORS.keys()),
            key="-SENSOR-",
            default_value=list(SENSORS.keys())[0],
            readonly=True,
            size=(20, 1)
        )],

        [sg.Text("Algorithm"), sg.Combo(
            list(["Bayesian", "wBrovey"]),
            key="-ALGORITHM-",
            default_value="Bayesian",
            readonly=True,
            size=(20, 1)
        )],

        [sg.Text("If Bayesian algorithm is selected, define a valid 'lambda' parameter.")],
        [sg.Text(
            "For integer reflectance values (constant applied) use a higher value than for original reflectance (0-1) values",
            font=("Arial", 9, "italic")
        )],
        [sg.Input(default_text='0.995', key='-LAMBDA-')],

        [sg.Button("Run"), sg.Button("Exit")],
        [sg.Output(size=(80, 20))]
    ]

    window = sg.Window("Pansharpening", layout)

    while True:
        event, values = window.read()


        if event in (sg.WINDOW_CLOSED, "Exit"):
            break
        if event == "Run":
            
            if not values["-MUL-"]:
                sg.popup_ok(f"Please select a valid MUL image directory.")
                continue

            if not values["-PAN-"]:
                sg.popup_ok(f"Please select a valid PAN image directory.")
                continue
            
            if not values["-OUT-"]:
                values["-OUT-"] = str(Path(values["-MUL-"]).parent)

            try:
                lambda_val = float(values['-LAMBDA-'])
                if lambda_val > 1 or lambda_val < 0:
                    sg.popup("The valid range of Bayes' lambda parameter is 0-1")
                    continue
            except:
                sg.popup("Only float numbers between 0 and 1 are allowed")
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
