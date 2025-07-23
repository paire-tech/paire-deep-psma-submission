import logging
from functools import partial
from pathlib import Path
from typing import Any, Dict, Generator, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk
from dotenv import load_dotenv
from rich.progress import track
from typer import Option, Typer

from .config import settings
from .inference import execute_lesions_segmentation
from .utils import find_file_path, load_json

IMAGE_EXTS = [".nii.gz", ".mha", ".tif", ".tiff"]


app = Typer(
    help="CLI for PaIRe DEEP PSMA Grand Challenge Submission",
    pretty_exceptions_enable=False,
)

log = logging.getLogger(__name__)


@app.command()
def main(
    input_format: str = Option(
        settings.INPUT_FORMAT,
        "--input-format",
        "-f",
        help="Format of the input files. Supported formats are 'gc' and 'csv'.",
    ),
    input_dir: str = Option(
        settings.INPUT_DIR,
        "--input-dir",
        "-i",
        help="Directory containing the input file(s) for the model.",
        exists=True,
        dir_okay=True,
        readable=True,
    ),
    input_csv: str = Option(
        settings.INPUT_CSV,
        "--input-csv",
        help="Path to the CSV file containing input data. Required if input_format is 'csv'.",
    ),
    output_dir: str = Option(
        settings.OUTPUT_DIR,
        "--output-dir",
        "-o",
        help="Directory to save the output file(s).",
        exists=True,
        dir_okay=True,
        readable=True,
    ),
    device: str = Option(
        settings.DEVICE,
        "--device",
        "-d",
        help="Device to use for inference. Options are 'cuda', 'cpu' or 'mps'.",
    ),
) -> None:
    # This line is used to automatically load nnUNet environment variables in case they
    # are set in a .env file.
    load_dotenv(override=True)

    if input_format not in ["gc", "csv"]:
        raise ValueError(f"Unsupported input format: {input_format}. Supported formats are 'gc' and 'csv'.")

    # Load the model only once
    # model = load_model(weights_dir, device=device)

    iter_data = iter_grand_challenge_data if input_format == "gc" else iter_csv_data
    for data in iter_data(input_dir, output_dir):
        # Run inference for PSMA inputs
        log.info("[PSMA] Running lesions segmentation inference")
        psma_pred_image = execute_lesions_segmentation_ensemble(
            pt_image=data["psma_pt_image"],
            ct_image=data["psma_ct_image"],
            suv_threshold=data["psma_pt_suv_threshold"],
            tracer_name="PSMA",
        )

        pred_path = Path(data["psma_pred_path"])
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Saving prediction to '%s'", pred_path)
        sitk.WriteImage(sitk.Cast(psma_pred_image, sitk.sitkUInt8), pred_path)

        # Run inference for FDG inputs
        log.info("[FDG ] Running lesions segmentation inference")
        fdg_pred_image = execute_lesions_segmentation_ensemble(
            pt_image=data["fdg_pt_image"],
            ct_image=data["fdg_ct_image"],
            suv_threshold=data["fdg_pt_suv_threshold"],
            tracer_name="FDG",
        )

        pred_path = Path(data["fdg_pred_path"])
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Saving prediction to '%s'", pred_path)
        sitk.WriteImage(sitk.Cast(fdg_pred_image, sitk.sitkUInt8), pred_path)


def iter_grand_challenge_data(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> Generator[Dict[str, Any], None, None]:
    # Grand Challenge data have only one set of inputs, and the algorithm / docker is used for each set / exam
    input_dir = Path(input_dir)
    images_dir = input_dir / "images"

    psma_ct_image_path = find_file_path(images_dir / "psma-ct", ext=IMAGE_EXTS)
    psma_ct_image_organ_segmentation_path = find_file_path(images_dir / "psma-ct-organ-segmentation", ext=IMAGE_EXTS)
    psma_pt_image_path = find_file_path(images_dir / "psma-pet-ga-68", ext=IMAGE_EXTS)
    psma_pt_suv_threshold_path = input_dir / "psma-pet-suv-threshold.json"
    fdg_ct_image_path = find_file_path(images_dir / "fdg-ct", ext=IMAGE_EXTS)
    fdg_ct_image_organ_segmentation_path = find_file_path(images_dir / "fdg-ct-organ-segmentation", ext=IMAGE_EXTS)
    fdg_pt_image_path = find_file_path(images_dir / "fdg-pet", ext=IMAGE_EXTS)
    fdg_pt_suv_threshold_path = input_dir / "fdg-pet-suv-threshold.json"
    psma_to_fdg_registration_path = input_dir / "psma-to-fdg-registration.json"

    psma_to_fdg_registration = load_json(psma_to_fdg_registration_path)["3d_affine_transform"]
    psma_to_fdg_registration = np.array(psma_to_fdg_registration, dtype=np.float32)

    yield {
        "psma_ct_image": sitk.ReadImage(psma_ct_image_path),
        "psma_organ_segmentation_image": sitk.ReadImage(psma_ct_image_organ_segmentation_path),
        "psma_pt_image": sitk.ReadImage(psma_pt_image_path),
        "psma_pt_suv_threshold": load_json(psma_pt_suv_threshold_path),
        "fdg_ct_image": sitk.ReadImage(fdg_ct_image_path),
        "fdg_organ_segmentation_image": sitk.ReadImage(fdg_ct_image_organ_segmentation_path),
        "fdg_pt_image": sitk.ReadImage(fdg_pt_image_path),
        "fdg_pt_suv_threshold": load_json(fdg_pt_suv_threshold_path),
        "psma_to_fdg_registration": psma_to_fdg_registration,
        # forward the path to the predictions output
        "psma_pred_path": Path(output_dir, "images", "psma-pet-ttb", "output.mha"),
        "fdg_pred_path": Path(output_dir, "images", "fdg-pet-ttb", "output.mha"),
    }


def iter_csv_data(
    input_csv: Union[str, Path],
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> Generator[Dict[str, Any], None, None]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    df = pd.read_csv(input_csv)
    log.info("Loaded %s entries from '%s'", len(df), input_csv)

    df["psma_ct_path"] = df["psma_ct_path"].apply(lambda path: input_dir / path)
    df["psma_organ_segmentation_path"] = df["psma_organ_segmentation_path"].apply(lambda path: input_dir / path)
    df["psma_pt_path"] = df["psma_pt_path"].apply(lambda path: input_dir / path)
    df["psma_pred_path"] = df["psma_pred_path"].apply(lambda path: output_dir / path)

    df["fdg_ct_path"] = df["fdg_ct_path"].apply(lambda path: input_dir / path)
    df["fdg_organ_segmentation_path"] = df["fdg_organ_segmentation_path"].apply(lambda path: input_dir / path)
    df["fdg_pt_path"] = df["fdg_pt_path"].apply(lambda path: input_dir / path)
    df["fdg_pred_path"] = df["fdg_pred_path"].apply(lambda path: output_dir / path)

    for _, row in track(df.iterrows(), total=len(df), description="Processing"):
        yield {
            "psma_ct_image": sitk.ReadImage(row["psma_ct_path"]),
            "psma_organ_segmentation_image": sitk.ReadImage(row["psma_organ_segmentation_path"]),
            "psma_pt_image": sitk.ReadImage(row["psma_pt_path"]),
            "psma_pt_suv_threshold": row["psma_pt_suv_threshold"],
            "fdg_ct_image": sitk.ReadImage(row["fdg_ct_path"]),
            "fdg_organ_segmentation_image": sitk.ReadImage(row["fdg_organ_segmentation_path"]),
            "fdg_pt_image": sitk.ReadImage(row["fdg_pt_path"]),
            "fdg_pt_suv_threshold": row["fdg_pt_suv_threshold"],
            "psma_to_fdg_registration": row.get("psma_to_fdg_registration"),
            # forward the path to the predictions output
            "psma_pred_path": row["psma_pred_path"],
            "fdg_pred_path": row["fdg_pred_path"],
        }
