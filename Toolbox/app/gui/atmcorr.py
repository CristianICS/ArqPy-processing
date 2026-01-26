from pathlib import Path
from osgeo import gdal

from atmo_correction import OutputSpec, SensorParams
from atmo_correction import FORMULAS
from atmo_correction import correct_raster_streaming

from sensors import WV3, LG06

import PySimpleGUI as sg

gdal.UseExceptions()

SENSORS = {"WV3": WV3, "LEGION_06": LG06}

def run_job(img_folder, out_folder, sg_values):
    try:

        image_c = SENSORS[sg_values["-SENSOR-"]]
        image = image_c(img_folder)

        clip_path = Path(sg_values.get("-CLIP_VECTOR-", ""))
        if not clip_path.exists():
                sg.popup_ok("The vector folder does not exist.\
                             Please switch it to a valid one.")

        atm_type = sg_values["-ATM-"]
        
        print(f"Running atmospheric correction on: {img_folder.stem}")

        for img_type, img_path in image.get_images().items():

            init_message = f"Correcting image {img_path.stem}"
            print(init_message)
            print("-"*len(init_message)*2)

            if clip_path != Path(""):
                print("Clipping...")
                out_clip_path = Path(out_folder, img_path.stem + "_temp.tif")
                gdal.Warp(
                    out_clip_path,              # dst
                    img_path,                   # src
                    cutlineDSName=clip_path,
                    cropToCutline=True,
                    dstNodata=0,
                    multithread=True,
                    creationOptions=["TILED=YES","COMPRESS=DEFLATE","PREDICTOR=2"],
                    warpOptions=["NUM_THREADS=ALL_CPUS"]
                )

            if atm_type == "DOS3":
                
                if img_type == "mul":
                    # Band to compute the min DN
                    image.START_BAND = "C"

                elif img_type == "pan":
                    image.START_BAND = "P"

                band_params = image.extract_params_per_band(img_path)
                ctx_params = image.extract_ctx_params(img_path)

                params = SensorParams(
                    sg_values["-SENSOR-"],
                    band_params,
                    ctx_params,
                    image_c.radiometric_calibration
                )

                # Create output names
                outname = img_path.stem + f"_{atm_type}.tif"
                out_path = Path(out_folder, outname)

                if clip_path != Path(""):
                    img_path_init = img_path
                    img_path = out_clip_path
                print("Starting DOS3 correction...")
                band_formulas = [atm_type] * len(band_params)
                correct_raster_streaming(
                    infile=img_path,
                    outfile=out_path,
                    band_formulas=band_formulas,
                    sensor=params,
                    out=OutputSpec(
                        dtype="float32",
                        nodata=0,
                        scale=None,
                        compress="DEFLATE",
                        predictor=2,
                        blocksize=512,
                        tiled=True,
                        cog=True),
                    build_overviews=True,
                    scale_to_int=True
                )

                if clip_path != Path(""):
                    print("Removing temp files...")
                    # Remove temp image
                    out_clip_path.unlink()
                    img_path = img_path_init
    
    except Exception as e:
        return e

def main():
    sg.theme('VSCodeDark')

    title = [sg.Text("Atmospheric Correction Tool", font=("Arial", 16, "bold"))]

    layout = [
        title,
        # Input and output placeholders
        [sg.Text("Input Image Folder:")],
        [sg.Input(key="-IMG-"), sg.FolderBrowse()],
        [sg.Text("Output Folder:")],
        [sg.Text(
            "Leave it blank if you want to autocreate it",
            font=("Arial", 10, "italic")
        )],
        [sg.Input(key="-OUT-"),sg.FolderBrowse()],
        
        # Clip operation
        # Vector layer path
        [sg.Text("Vector layer to perform a clip operation:")],
        [sg.Text("Leave it blank to skip", font=("Arial", 10, "italic"))],
        [
            sg.Input(key="-CLIP_VECTOR-",),
            sg.FileBrowse(
                file_types=(
                    ("Vector files", "*.shp;*.gpkg;*.geojson"),
                    ("All files", "*.*"),
                )
            )
        ],

        [sg.Text("Sensor"), sg.Combo(
            list(SENSORS.keys()),
            key="-SENSOR-",
            default_value="WV3",
            readonly=True,
            size=(20, 1)
        )],

        [sg.Text("Atmospheric Correction formula:"), sg.Combo(
            list(FORMULAS.keys()),
            key="-ATM-",
            default_value="DOS3",
            readonly=True,
            size=(20, 1)
        )],

        [sg.Button("Run"), sg.Button("Exit")],
        [sg.Output(size=(80, 20))]
    ]

    window = sg.Window("Atmospheric Correction", layout)

    while True:
        event, values = window.read()

        if event in (sg.WINDOW_CLOSED, "Exit"):
            break

        if event == "Run":
            
            if not values["-IMG-"]:
                sg.popup_ok(f"Please select a valid image directory.")
                continue
            else:
                img_global_path = Path(values["-IMG-"])

            if not values["-OUT-"]:
                # Check if the default output directory exists
                img_id = img_global_path.name
                boa_folder = img_global_path / (img_id + "_BOA")
                window["-OUT-"].update(str(boa_folder))
            
                if boa_folder.exists():
                    sg.popup_ok(
                        "The default output folder exists."\
                        "Please switch it to another one or re run again to continue."
                    )
                    continue
                else:
                    boa_folder.mkdir(exist_ok=True)

            else:
                boa_folder = Path(values["-OUT-"])
                if not boa_folder.exists():
                    sg.popup_ok("The output folder does not exist.\
                                 Please switch it to a valid one.")

            # Disable Run button and show a status message
            window["Run"].update(disabled=True)
            print("Starting atmospheric correction...")

            # Launch long operation in a worker thread
            window.perform_long_operation(
                lambda: run_job(img_global_path, boa_folder, values),
                "-JOB-DONE-",
            )

        elif event == "-JOB-DONE-":

            # Re-enable Run button
            window["Run"].update(disabled=False)

            result = values["-JOB-DONE-"]

            if isinstance(result, FileNotFoundError):
                sg.popup_ok("The image folder is not valid.", str(result))
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
