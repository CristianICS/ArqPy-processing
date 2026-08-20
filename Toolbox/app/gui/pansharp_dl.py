"""PySimpleGUI front end for the bundled Z-PNN pansharpening methods."""

from pathlib import Path

import PySimpleGUI as sg

from pansharp_dl import ALGORITHMS, SENSORS, run_pansharpening
from .icon_base64 import ICON

def _validate(values):
    """Check every value inside GUI is valid."""
    ms_path = Path(values["-MS-"])
    pan_path = Path(values["-PAN-"])

    if not ms_path.is_file():
        raise ValueError("Please select a valid multispectral raster.")

    if not pan_path.is_file():
        raise ValueError("Please select a valid panchromatic raster.")

    if values["-ALGORITHM-"] not in ALGORITHMS:
        raise ValueError("Please select a supported pansharpening algorithm.")

    if values["-SENSOR-"] not in SENSORS:
        raise ValueError("Please select a supported sensor.")

    epochs = int(values["-EPOCHS-"])
    if epochs < 0:
        raise ValueError("Epochs cannot be negative.")

    output_text = values["-OUT-"].strip()
    if output_text and not Path(output_text).is_dir():
        raise ValueError("The selected output folder does not exist.")


def run_job(values):
    try:
        ms_path = Path(values["-MS-"])
        output_text = values["-OUT-"].strip()
        output_folder = Path(output_text) if output_text else ms_path.parent / "pansharp_dl"
        return run_pansharpening(
            ms_path=ms_path,
            pan_path=Path(values["-PAN-"]),
            algorithm=values["-ALGORITHM-"],
            sensor=values["-SENSOR-"],
            output_folder=output_folder,
            epochs=int(values["-EPOCHS-"]),
            use_cpu=values["-CPU-"],
            coregistration=values["-COREGISTER-"],
        )
    except Exception as exc:
        return exc


def main():
    sg.theme("VSCodeDark")

    raster_types = (("GeoTIFF", "*.tif;*.tiff"), ("All files", "*.*"))

    layout = [
        [sg.Text("Deep-learning Pansharpening", font=("Arial", 16, "bold"))],

        [sg.Text("Multispectral (MS/ML) raster:")],
        [sg.Input(key="-MS-"), sg.FileBrowse(file_types=raster_types)],

        [sg.Text("Panchromatic (PAN) raster:")],
        [sg.Input(key="-PAN-"), sg.FileBrowse(file_types=raster_types)],

        [
            sg.Text("Pansharpening algorithm:"),
            sg.Combo(
                ALGORITHMS,
                default_value="Z-PNN",
                readonly=True,
                key="-ALGORITHM-",
                size=(18, 1)
            ),
            sg.Text("Sensor:"),
            sg.Combo(
                tuple(SENSORS),
                default_value="WV3",
                readonly=True,
                key="-SENSOR-",
                size=(8, 1)
            ),
        ],

        [
            sg.Text("Fine-tuning epochs:"),
            sg.Input("1", key="-EPOCHS-", size=(7, 1)),
            sg.Checkbox("Use CPU", key="-CPU-", default=False),
            sg.Checkbox(
                "Enable co-registration loss",
                key="-COREGISTER-",
                default=True
            ),
        ],

        [sg.Text(
            "Output folder (blank creates 'pansharp_dl' beside the MS raster):"
        )],
        [sg.Input(key="-OUT-"), sg.FolderBrowse()],

        [sg.Button("Run"), sg.Button("Exit")],
        [sg.Output(size=(95, 22))],
    ]

    window = sg.Window("Deep-learning Pansharpening", layout, icon=ICON)

    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSED, "Exit"):
            break

        if event == "Run":
            try:
                _validate(values)
            except (TypeError, ValueError) as exc:
                sg.popup_ok("Invalid settings", str(exc))
                continue

            window["Run"].update(disabled=True)
            print("Starting deep-learning pansharpening...")

            window.perform_long_operation(lambda: run_job(values), "-JOB-DONE-")

        elif event == "-JOB-DONE-":
            window["Run"].update(disabled=False)

            result = values["-JOB-DONE-"]

            if isinstance(result, Exception):
                print(f"Error: {result}")
                sg.popup_error("Pansharpening failed", str(result))
            else:
                message = f"Pansharpening completed: {result['tif']}"
                print(message)
                sg.popup_ok(message)

    window.close()
