from pathlib import Path

import PySimpleGUI as sg

from pca.job import PcaJob
from sensors import SENSORS

from .framework import (
    ToolWindow,
    clip_vector_row,
    output_folder_row,
    resolve_clip_vector_value,
    resolve_output_folder_value,
    sensor_combo_row,
)


def run_job(values: dict):
    """Perform Principal Components Analysis between all image bands."""
    try:
        raw_img = values.get("-IMG-", "")
        if not raw_img:
            return ValueError("Please select a valid image.")
        img_path = Path(raw_img)
        if not img_path.exists():
            return FileNotFoundError(str(img_path))

        out_folder = resolve_output_folder_value(
            values, "-OUT-",
            default_factory=lambda: img_path.parent.parent / "PCA",
        )
        clip_vector = resolve_clip_vector_value(values)

        raw_combis = values.get("-COMBIS_PATH-", "")
        if raw_combis:
            combis_csv = Path(raw_combis)
            if not combis_csv.exists():
                return FileNotFoundError(
                    f"PCA combinations file not found: {combis_csv}"
                )
        else:
            combis_csv = None

        sensor_cls = SENSORS[values["-SENSOR-"]]

        job = PcaJob(
            img_path=img_path,
            out_folder=out_folder,
            sensor_band_names=sensor_cls.BAND_KEYS,
            sensor_band_pos=sensor_cls.BAND_POS,
            combis_csv=combis_csv,
            clip_vector=clip_vector,
        )
        job.validate()
        job.run(progress=print)
    except Exception as e:
        return e


def main():
    tw = ToolWindow(
        title="Compute PCA",
        window_title="Principal Component Analysis",
    )
    tw.add_rows(
        [sg.Text("Input image:")],
        [sg.Input(key="-IMG-"), sg.FileBrowse()],
    )
    tw.add_rows(*output_folder_row("-OUT-", "Select folder to store the results:"))
    tw.add_rows(
        [sg.Text("Select the CSV file containing the PCA combinations:")],
        [sg.Text("Leave it blank to autocreate it", font=("Arial", 10, "italic"))],
        [sg.Input(key="-COMBIS_PATH-"), sg.FileBrowse(
            file_types=(
                ("Combinations files", "*.csv"),
                ("All files", "*.*"),
            )
        )],
    )
    tw.add_rows(*clip_vector_row())
    tw.add_rows(*sensor_combo_row())
    tw.run(job=run_job, start_message="Starting PCA computation...")
