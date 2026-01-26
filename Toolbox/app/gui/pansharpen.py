from pathlib import Path

from pansharp import SENSORS
from pansharp import brovey, bayesian

import PySimpleGUI as sg

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
        
        out_folder = Path(values["-OUT-"])
        if not out_folder.exists():
            out_folder.mkdir(exist_ok=True)
        
        method = values["-ALGORITHM-"]
        if method == "Bayesian":
            print("Starting Bayesian pansharpening...")
            bayes_img_name = mul_img_path.name.replace(".tif", "_bayes.tif")
            bayes_img_path = out_folder / bayes_img_name
            bayesian(mul_img_path, pan_img_path, bayes_img_path)

        else:
            # Perform weighted Brovey pansharpening
            brov_img_name = mul_img_path.name.replace(".tif", "_brovey.tif")
            brov_img_path = out_folder / brov_img_name
            brovey(
                SENSORS[values["-SENSOR-"]],
                mul_img_path, pan_img_path, brov_img_path)

            if values["-SENSOR-"] == "WV3":
                lmnv_path = SENSORS[values["-SENSOR-"]].lmnv(
                    mul_img_path, pan_img_path)
                SENSORS[values["-SENSOR-"]].merge_pansharpened_bands(
                    brov_img_path, lmnv_path
                )
    except Exception as e:
        return e

def main():
    sg.theme('VSCodeDark')

    title = [sg.Text("Pansharpening Tool for WV3", font=("Arial", 16, "bold"))]

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
