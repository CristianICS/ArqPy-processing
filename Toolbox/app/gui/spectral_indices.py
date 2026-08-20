from pathlib import Path

import PySimpleGUI as sg

from spectral_indices import Indices
from .icon_base64 import ICON


def run_job(values):
    """Computing spectral indices operation."""
    try:
        # Check input MUL and PAN images
        img_path = Path(values["-IMG-"])
        if not img_path.exists():
            return FileNotFoundError(str(img_path))
        
        if not values["-OUT-"]:
            out_folder = img_path.parent.parent / "SPECTRAL_INDICES"
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

        if values["-SENSOR-"] == "WV3":
            band_pos = [1, 2, 3, 4, 5, 6, 7, 8]
            band_keys = ["C", "B", "G", "Y", "R", "RE1", "N", "N2"]
        elif values["-SENSOR-"] == "LEGION":
            band_pos = [1, 2, 3, 4, 5, 6, 7, 8]
            band_keys = ["C", "B", "G", "Y", "R", "RE1", "RE2", "N"]

        for index_key in Indices.get_index_keys():
            print("\n", index_key)
            print('-' * 20)
            co = [
                "COMPRESS=DEFLATE",
                "PREDICTOR=2",
                "BIGTIFF=IF_SAFER",
                "NUM_THREADS=ALL_CPUS"
            ]
            
            Indices.compute_index(
                index_key,
                img_path,
                band_pos, band_keys, out_folder,
                clip_layer_path = clip_path,
                creation_opts=co
            )

    except Exception as e:
        return e

def main():
    sg.theme('VSCodeDark')

    title = [sg.Text("Spectral Indices", font=("Arial", 16, "bold"))]

    layout = [
        title,

        # Input and output placeholders
        [sg.Text("Input image:")],
        [sg.Input(key="-IMG-"), sg.FileBrowse()],
        [sg.Text("Select folder to store the results:")],
        [sg.Text("Leave it blank if you want to autocreate it", font=("Arial", 10, "italic"))],
        [sg.Input(key="-OUT-"), sg.FolderBrowse()],

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

    window = sg.Window("Spectral indices", layout, icon=ICON)

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
            print("Starting computations...")

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
