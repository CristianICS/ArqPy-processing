from pathlib import Path

from highpass.utils import GDALHighPassFilter

import PySimpleGUI as sg

def run_job(values):
    """Computing high pass filters over one image band."""
    try:
        # Check input MUL and PAN images
        img_path = Path(values["-IMG-"])
        if not img_path.exists():
            return FileNotFoundError(str(img_path))
        
        if not values["-OUT-"]:
            out_folder = img_path.parent.parent / "HIGHPASS"
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

        # Get band index
        band = int(values["-BAND-"])

        hp = GDALHighPassFilter(img_path, band_index=band)
        lap, log, sob, hb = hp.run_all(out_folder / img_path.stem, clip_path)

    except Exception as e:
        return e


def main():
    sg.theme('VSCodeDark')

    title = [sg.Text("Highpass filters", font=("Arial", 16, "bold"))]

    layout = [
        title,
        # Description
        [sg.Text(
            "Apply common high-pass filters (Laplacian, LoG, Sobel"\
            "magnitude, High-boost) using GDAL+NumPy."
        )],

        # Input and output placeholders
        [sg.Text("Input image:")],
        [sg.Input(key="-IMG-"), sg.FileBrowse()],
        [sg.Text("Image band to process:")],
        [sg.Text("Band index (1-based). Default: 1", font=("Arial", 10, "italic"))],
        [sg.Input(key="-BAND-", default_text="1")],
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

        [sg.Button("Run"), sg.Button("Exit")],
        [sg.Output(size=(80, 20))]
    ]

    window = sg.Window("Highpass filters", layout)

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