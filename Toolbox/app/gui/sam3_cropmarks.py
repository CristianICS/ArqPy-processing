"""PySimpleGUI front end for crop-mark segmentation with SAM 3."""

from pathlib import Path

import PySimpleGUI as sg

from sam3_cropmarks import parse_rgb_bands, process_input
from .icon_base64 import ICON

def run_job(values):
    try:
        input_path = Path(values["-INPUT-"])
        output_text = values["-OUT-"].strip()
        input_folder = input_path if input_path.is_dir() else input_path.parent
        output_folder = Path(output_text) if output_text else input_folder / "sam3_crop_marks"
        checkpoint_text = values["-CHECKPOINT-"].strip()
        checkpoint = Path(checkpoint_text) if checkpoint_text else None
        return process_input(
            input_path=input_path,
            output_folder=output_folder,
            prompt=values["-PROMPT-"].strip(),
            confidence=float(values["-CONFIDENCE-"]),
            rgb_bands=parse_rgb_bands(values["-RGB-"]),
            tile_size=int(values["-TILE-SIZE-"]),
            overlap=int(values["-OVERLAP-"]),
            checkpoint=checkpoint,
            overwrite=values["-OVERWRITE-"],
        )
    except Exception as exc:
        return exc


def _validate(values):
    if not values["-INPUT-"].strip():
        raise ValueError("Please select an input GeoTIFF or folder.")
    input_path = Path(values["-INPUT-"])
    if not input_path.exists():
        raise ValueError("Please select a valid input GeoTIFF or folder.")
    if input_path.is_file() and input_path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("The selected input file must be a .tif or .tiff image.")
    if not input_path.is_file() and not input_path.is_dir():
        raise ValueError("The selected input must be a GeoTIFF or folder.")
    if not values["-PROMPT-"].strip():
        raise ValueError("The text prompt cannot be empty.")
    confidence = float(values["-CONFIDENCE-"])
    if not 0 < confidence <= 1:
        raise ValueError("Confidence must be greater than 0 and at most 1.")
    parse_rgb_bands(values["-RGB-"])
    tile_size = int(values["-TILE-SIZE-"])
    overlap = int(values["-OVERLAP-"])
    if tile_size < 256:
        raise ValueError("Tile size must be at least 256 pixels.")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("Overlap must be non-negative and smaller than tile size.")
    checkpoint_text = values["-CHECKPOINT-"].strip()
    if checkpoint_text and not Path(checkpoint_text).is_file():
        raise ValueError("The selected checkpoint file does not exist.")


def main():
    sg.theme("VSCodeDark")
    layout = [
        [sg.Text("SAM 3 Crop-mark Segmentation", font=("Arial", 16, "bold"))],
        [sg.Text("Extract high-confidence crop-mark masks from one GeoTIFF or a folder.")],
        [sg.Text("Input GeoTIFF or folder:")],
        [
            sg.Input(key="-INPUT-"),
            sg.FileBrowse(
                "Select TIFF",
                target="-INPUT-",
                file_types=(("GeoTIFF images", "*.tif;*.tiff"),),
            ),
            sg.FolderBrowse("Select Folder", target="-INPUT-"),
        ],
        [sg.Text("Output folder (blank creates 'sam3_crop_marks' beside the input):")],
        [sg.Input(key="-OUT-"), sg.FolderBrowse()],
        [sg.Text("Text prompt:"), sg.Input("crop mark", key="-PROMPT-", size=(35, 1))],
        [
            sg.Text("Minimum confidence:"),
            sg.Input("0.70", key="-CONFIDENCE-", size=(8, 1)),
            sg.Text("RGB bands (1-based):"),
            sg.Input("1,2,3", key="-RGB-", size=(9, 1)),
        ],
        [
            sg.Text("Tile size:"), sg.Input("1008", key="-TILE-SIZE-", size=(8, 1)),
            sg.Text("Overlap:"), sg.Input("128", key="-OVERLAP-", size=(8, 1)),
        ],
        [sg.Text("Optional local SAM 3 checkpoint (blank uses Hugging Face):")],
        [
            sg.Input(key="-CHECKPOINT-"),
            sg.FileBrowse(file_types=(("PyTorch checkpoint", "*.pt;*.pth"), ("All files", "*.*"))),
        ],
        [sg.Checkbox("Overwrite existing outputs", key="-OVERWRITE-", default=False)],
        [sg.Button("Run"), sg.Button("Exit")],
        [sg.Output(size=(90, 20))],
    ]
    window = sg.Window("SAM 3 Crop-mark Segmentation", layout, icon=ICON)

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
            print("Starting crop-mark segmentation...")
            window.perform_long_operation(lambda: run_job(values), "-JOB-DONE-")
        elif event == "-JOB-DONE-":
            window["Run"].update(disabled=False)
            result = values["-JOB-DONE-"]
            if isinstance(result, Exception):
                print(f"Error: {result}")
                sg.popup_error("Segmentation failed", str(result))
            else:
                detected = sum(int(item["instances"]) for item in result)
                processed = sum(not int(item["skipped"]) for item in result)
                message = (
                    f"Completed {processed} image(s); detected {detected} "
                    "tile-level candidate mask(s)."
                )
                print(message)
                sg.popup_ok(message)
    window.close()
