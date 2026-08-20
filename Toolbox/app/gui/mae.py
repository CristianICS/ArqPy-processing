from pathlib import Path
import shutil
import PySimpleGUI as sg

from mae import (
    tilling,
    compute_mae_batch,
    define_model,
    clip_mae,
    parse_mae_bands,
    validate_source_bands,
    compute_raster_stats,
    update_csv
)

from .icon_base64 import ICON


def run_job(values):
    """Computing MAE over all images inside a folder."""
    try:
        bands = parse_mae_bands(values["-BANDS-"])

        # Retrieve the model
        model, device = define_model()

        imgs_folder = Path(values["-FOLDER-"])

        # Define the output folder
        out_folder = imgs_folder / "mae_saliency"
        if not out_folder.exists():
            out_folder.mkdir()

        # Define the csv path to store MAE statistics
        stats_path = Path(out_folder, "mae_stats.csv")
        
        clip_path = Path(values.get("-CLIP_VECTOR-", ""))
        if not clip_path.exists():
                sg.popup_ok("The vector folder does not exist.\
                             Please switch it to a valid one.")
        if clip_path == Path(""):
            clip_path = False

        for img_path in imgs_folder.glob("*.tif"):
            # Output name for the MAE's saliency map
            out_name = img_path.stem + "_mae.tif"
            out_path = out_folder / out_name

            if out_path.exists() or img_path.stem.endswith("_mae"):
                continue
            print(f"Computing img {img_path.stem}")

            # Prepare the current image to perform MAE
            validate_source_bands(img_path, bands)
            tiles_meta, tiles_folder = tilling(img_path)

            compute_mae_batch(
                tiles_meta,
                out_path,
                model,
                device,
                bands=bands,
            )

            if clip_path:
                print("Perform clip operation...")
                clip_mae(out_path, clip_path)

            # Remove the temporal folder with the tiles
            shutil.rmtree(tiles_folder)

            stats, _ = compute_raster_stats(out_path)
            update_csv(stats_path, stats)

    except Exception as e:
        return e


def _validate(values):
    """Validate settings that do not require opening every source raster."""
    if not Path(values["-FOLDER-"]).is_dir():
        raise ValueError("Please select a valid input folder.")
    parse_mae_bands(values["-BANDS-"])

    clip_text = values.get("-CLIP_VECTOR-", "").strip()
    if clip_text and not Path(clip_text).is_file():
        raise ValueError("The selected vector layer does not exist.")


def main():
    sg.theme('VSCodeDark')

    title = [sg.Text("MAE detection", font=("Arial", 16, "bold"))]

    layout = [
        title,
        # Description
        [sg.Text(
            "Use the Masked Autoencoder technique to detect regions and "\
            "images with higher probabilities of containing crop marks."
        )],

        [sg.Text(
            "Extract the MAE saliency values for all the GeoTIFF images "\
            "inside the input folder."
        )],

        # Input and output placeholders
        [sg.Text("Input folder:")],
        [sg.Input(key="-FOLDER-"), sg.FolderBrowse()],
        [
            sg.Text("Bands to process (1-based, comma-separated):"),
            sg.Input("1", key="-BANDS-", size=(12, 1)),
        ],
        [sg.Text(
            "Enter one to three bands. One band is repeated across all "
            "three model channels.",
            font=("Arial", 10, "italic"),
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
        [sg.Button("Run"), sg.Button("Exit")],
        [sg.Output(size=(80, 20))]
    ]

    window = sg.Window("MAE detection", layout, icon=ICON)

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
