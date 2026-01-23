from pathlib import Path
import shutil
import PySimpleGUI as sg
from mae import tilling, compute_mae_batch, define_model


def run_job(values):
    """Computing MAE over all images inside a folder."""
    try:
        # Retrieve the model
        model, device = define_model()

        imgs_folder = Path(values["-FOLDER-"])
        # Define the csv path to store MAE statistics
        stats_path = Path(imgs_folder, "mae_stats.csv")
        
        for img_path in imgs_folder.glob("*.tif"):
            # Output name for the MAE's heatmap
            out_name = img_path.stem + "_mae.tif"
            out_path = img_path.parent / out_name

            if out_path.exists() or img_path.stem.endswith("_mae"):
                continue
            print(f"Computing img {img_path.stem}")

            # Prepare the current image to perform MAE
            tiles_meta, tiles_folder = tilling(img_path)

            compute_mae_batch(tiles_meta, out_path, stats_path, model, device)

            # Remove the temporal folder with the tiles
            shutil.rmtree(tiles_folder)

    except Exception as e:
        return e


def main():
    sg.theme('VSCodeDark')

    title = [sg.Text("MAE detection", font=("Arial", 16, "bold"))]

    layout = [
        title,
        # Description
        [sg.Text(
            "Use the Masked Autoencoder technique to detect regions and"\
            "images with higher probabilities of containing crop marks."
        )],

        # Input and output placeholders
        [sg.Text("Input folder:")],
        [sg.Input(key="-FOLDER-"), sg.FolderBrowse()],
        [sg.Button("Run"), sg.Button("Exit")],
        [sg.Output(size=(80, 20))]
    ]

    window = sg.Window("MAE detection", layout)

    while True:
        event, values = window.read()


        if event in (sg.WINDOW_CLOSED, "Exit"):
            break
        if event == "Run":
            
            if not values["-FOLDER-"]:
                sg.popup_ok(f"Please select a valid folder.")
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
