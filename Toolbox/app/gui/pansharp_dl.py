"""PySimpleGUI front end for the bundled Z-PNN pansharpening methods."""

from pathlib import Path

import PySimpleGUI as sg

from pansharp_dl import ALGORITHMS, SENSORS, run_pansharpening

from .framework import ToolWindow, output_folder_row, resolve_output_folder_value


def run_job(values: dict):
    try:
        raw_ms = values.get("-MS-", "")
        if not raw_ms:
            return ValueError("Please select a valid multispectral raster.")
        ms_path = Path(raw_ms)
        if not ms_path.is_file():
            return FileNotFoundError(f"Multispectral raster not found: {ms_path}")

        raw_pan = values.get("-PAN-", "")
        if not raw_pan:
            return ValueError("Please select a valid panchromatic raster.")
        pan_path = Path(raw_pan)
        if not pan_path.is_file():
            return FileNotFoundError(f"Panchromatic raster not found: {pan_path}")

        if values["-ALGORITHM-"] not in ALGORITHMS:
            return ValueError("Please select a supported pansharpening algorithm.")
        if values["-SENSOR-"] not in SENSORS:
            return ValueError("Please select a supported sensor.")

        try:
            epochs = int(values["-EPOCHS-"])
        except ValueError:
            return ValueError("Epochs must be an integer number.")
        if epochs < 0:
            return ValueError("Epochs cannot be negative.")

        out_folder = resolve_output_folder_value(
            values, "-OUT-",
            default_factory=lambda: ms_path.parent / "pansharp_dl",
        )

        result = run_pansharpening(
            ms_path=ms_path,
            pan_path=pan_path,
            algorithm=values["-ALGORITHM-"],
            sensor=values["-SENSOR-"],
            output_folder=out_folder,
            epochs=epochs,
            use_cpu=values["-CPU-"],
            coregistration=values["-COREGISTER-"],
        )
        print(f"Pansharpening completed: {result['tif']}")
        return result
    except Exception as exc:
        return exc


def main():
    tw = ToolWindow(
        title="Deep-learning Pansharpening",
        window_title="Deep-learning Pansharpening",
    )

    raster_types = (("GeoTIFF", "*.tif;*.tiff"), ("All files", "*.*"))

    tw.add_rows(
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
    )
    tw.add_rows(*output_folder_row(
        "-OUT-", "Output folder (blank creates 'pansharp_dl' beside the MS raster):"
    ))
    tw.run(job=run_job, start_message="Starting deep-learning pansharpening...")
