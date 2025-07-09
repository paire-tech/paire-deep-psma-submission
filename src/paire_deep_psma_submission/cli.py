from pathlib import Path
from typing import Any, Dict, Generator, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk
from rich.progress import track
from typer import Option, Typer

from .inference import execute_lesions_segmentation
from .model import load_model
from .utils import find_file_path, load_json

IMAGE_EXTS = [".nii.gz", ".mha", ".tif", ".tiff"]
INPUT_FORMAT = "gc"
INPUT_DIR = "/input"
OUTPUT_DIR = "/output"
WEIGHTS_DIR = "/opt/ml/model"
DEVICE = "cpu"


app = Typer(
    help="CLI for PaIRe DEEP PSMA Grand Challenge Submission",
    pretty_exceptions_enable=False,
)


@app.command()
def main(
    input_format: str = Option(
        INPUT_FORMAT,
        "--input-format",
        "-f",
        help="Format of the input files. Supported formats are 'gc' and 'csv'.",
    ),
    input_dir: str = Option(
        INPUT_DIR,
        "--input-dir",
        "-i",
        help="Directory containing the input file(s) for the model.",
        exists=True,
        dir_okay=True,
        readable=True,
    ),
    output_dir: str = Option(
        OUTPUT_DIR,
        "--output-dir",
        "-o",
        help="Directory to save the output file(s).",
        exists=True,
        dir_okay=True,
        readable=True,
    ),
    device: str = Option(
        DEVICE,
        "--device",
        "-d",
        help="Device to run the model on. Use 'cpu', 'cuda', or 'auto' for automatic detection.",
    ),
    use_mixed_precision: bool = Option(
        False,
        "--use-mixed-precision",
        "-m",
        help="Use mixed precision for inference. This can speed up inference on compatible hardware.",
        is_flag=True,
    ),
    weights_dir: str = Option(
        WEIGHTS_DIR,
        "--weights-dir",
        "-w",
        help="Directory containing the model weights.",
    ),
) -> None:
    if input_format not in ["gc", "csv"]:
        raise ValueError(f"Unsupported input format: {input_format}. Supported formats are 'gc' and 'csv'.")

    # Load the model only once
    model = load_model(weights_dir, device=device)

    iter_data = iter_grand_challenge_data if input_format == "gc" else iter_csv_data
    for data in iter_data(input_dir, output_dir):
        # Run inference for PSMA inputs
        pred_image = execute_lesions_segmentation(
            pt_image=data["psma_pt_image"],
            ct_image=data["psma_ct_image"],
            organs_segmentation_image=data["psma_ct_image_organ_segmentation"],
            suv_threshold=data["psma_pt_suv_threshold"],
            model=model,
            device=device,
        )

        pred_path = Path(data["psma_pt_ttb_path"])
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(pred_image, pred_path)

        # Run inference for FDG inputs
        # pred_image = execute_lesions_segmentation(
        #     pt_image=data["fdg_pt_image"],
        #     ct_image=data["fdg_ct_image"],
        #     organs_segmentation_image=data["fdg_ct_image_organ_segmentation"],
        #     suv_threshold=data["fdg_pt_suv_threshold"],
        #     model=model,
        #     device=device,
        # )

        # pred_path = Path(data["fdg_pt_ttb_path"])
        # pred_path.parent.mkdir(parents=True, exist_ok=True)
        # sitk.WriteImage(pred_image, pred_path)


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
        "psma_ct_image_organ_segmentation": sitk.ReadImage(psma_ct_image_organ_segmentation_path),
        "psma_pt_image": sitk.ReadImage(psma_pt_image_path),
        "psma_pt_suv_threshold": load_json(psma_pt_suv_threshold_path),
        "fdg_ct_image": sitk.ReadImage(fdg_ct_image_path),
        "fdg_ct_image_organ_segmentation": sitk.ReadImage(fdg_ct_image_organ_segmentation_path),
        "fdg_pt_image": sitk.ReadImage(fdg_pt_image_path),
        "fdg_pt_suv_threshold": load_json(fdg_pt_suv_threshold_path),
        "psma_to_fdg_registration": psma_to_fdg_registration,
        # forward the path to the predictions (ttb)
        "psma_pt_ttb_path": Path(output_dir, "images", "psma-pet-ttb", "output.mha"),
        "fdg_pt_ttb_path": Path(output_dir, "images", "fdg-pet-ttb", "output.mha"),
    }


def iter_csv_data(input_dir: Union[str, Path], output_dir: Union[str, Path]) -> Generator[Dict[str, Any], None, None]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    csv_path = find_file_path(input_dir, ext=".csv")
    df = pd.read_csv(csv_path)

    df["psma_ct_image"] = df["psma_ct_image"].apply(lambda path: input_dir / path)
    df["psma_ct_image_organ_segmentation"] = df["psma_ct_image_organ_segmentation"].apply(lambda path: input_dir / path)
    df["psma_pt_image"] = df["psma_pt_image"].apply(lambda path: input_dir / path)
    df["psma_pt_ttb_image"] = df["psma_pt_ttb_image"].apply(lambda path: output_dir / path)

    df["fdg_ct_image"] = df["fdg_ct_image"].apply(lambda path: input_dir / path)
    df["fdg_ct_image_organ_segmentation"] = df["fdg_ct_image_organ_segmentation"].apply(lambda path: input_dir / path)
    df["fdg_pt_image"] = df["fdg_pt_image"].apply(lambda path: input_dir / path)
    df["fdg_pt_ttb_image"] = df["fdg_pt_ttb_image"].apply(lambda path: output_dir / path)

    for _, row in track(df.iterrows(), total=len(df), description="Processing"):
        yield {
            "psma_ct_image": sitk.ReadImage(row["psma_ct_image"]),
            "psma_ct_image_organ_segmentation": sitk.ReadImage(row["psma_ct_image_organ_segmentation"]),
            "psma_pt_image": sitk.ReadImage(row["psma_pt_image"]),
            "psma_pt_suv_threshold": row["psma_pt_suv_threshold"],
            "fdg_ct_image": sitk.ReadImage(row["fdg_ct_image"]),
            "fdg_ct_image_organ_segmentation": sitk.ReadImage(row["fdg_ct_image_organ_segmentation"]),
            "fdg_pt_image": sitk.ReadImage(row["fdg_pt_image"]),
            "fdg_pt_suv_threshold": row["fdg_pt_suv_threshold"],
            "psma_to_fdg_registration": row.get("psma_to_fdg_registration"),  # Not used (yet)
            # forward the path to the predictions (ttb)
            "psma_pt_ttb_path": row["psma_pt_ttb_image"],
            "fdg_pt_ttb_path": row["fdg_pt_ttb_image"],
        }
