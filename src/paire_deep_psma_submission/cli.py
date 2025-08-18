import glob
import logging
from functools import partial
from pathlib import Path
from typing import Generator, NotRequired, TypedDict, Union

import numpy as np
import pandas as pd
import SimpleITK as sitk
from rich.progress import track
from typer import Option, Typer

from .config import settings
from .inference import (
    execute_multiple_folds_lesions_segmentation,
    refine_fdg_prediction_from_psma_prediction,
    refine_my_ttb_label,
)
from .utils import find_file_path, load_json

IMAGE_EXTS = [".nii.gz", ".mha", ".tif", ".tiff"]


app = Typer(
    help="CLI for PaIRe DEEP PSMA Grand Challenge Submission",
    pretty_exceptions_enable=False,
)

log = logging.getLogger(__name__)


class DataDict(TypedDict):
    psma_ct_path: str
    psma_organ_segmentation_path: str
    psma_pt_path: str
    psma_pt_suv_threshold: float
    psma_pred_path: str
    fdg_ct_path: str
    fdg_organ_segmentation_path: str
    fdg_pt_path: str
    fdg_pt_suv_threshold: float
    fdg_pred_path: str
    psma_to_fdg_registration: NotRequired[np.ndarray]


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
) -> None:
    use_tta = not no_tta  # TTA is used unless --no-tta is passed
    postprocess_fdg_based_on_psma_classes = not do_not_postprocess_fdg_based_on_psma_classes
    if input_format not in ["gc", "csv"]:
        raise ValueError(f"Unsupported input format: {input_format}. Supported formats are 'gc' and 'csv'.")

    # Load the model only once
    # model = load_model(weights_dir, device=device)
    REFINE_TTB_LABEL = True
    REFINE_FDG_BASED_ON_PSMA = True

    iter_data = iter_grand_challenge_data if input_format == "gc" else partial(iter_csv_data, input_csv=input_csv)
    for data in iter_data(input_dir=input_dir, output_dir=output_dir):
        # Run inference for PSMA inputs
        log.info("[PSMA] Running lesions segmentation inference")
        list_path_to_psma_models = glob.glob("/app/nnUNet_results/*PSMA*/**/*.pth", recursive=True)

        psma_pred_image, psma_physiological_image = execute_multiple_folds_lesions_segmentation(
            pt_image=data["psma_pt_image"],
            ct_image=data["psma_ct_image"],
            total_segmentator_image=data["psma_organ_segmentation_image"],
            suv_threshold=data["psma_pt_suv_threshold"],
            list_path_to_pth_for_tracer=list_path_to_psma_models,
            tracer_name="PSMA",
        )

        # expand nnU-Net predicted disease region
        if REFINE_TTB_LABEL:
            psma_pred_image = refine_my_ttb_label(
                ttb_image=psma_pred_image,
                pet_image=data["psma_pt_image"],
                totseg_multilabel=data["psma_organ_segmentation_image"],
                expansion_radius_mm=5.0,
                pet_threshold_value=data["psma_pt_suv_threshold"],
                totseg_non_expand_values=[1, 2, 3, 5, 21],
                normal_image=psma_physiological_image,
            )

        pred_path = Path(data["psma_pred_path"])
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Saving prediction to '%s'", pred_path)
        sitk.WriteImage(psma_pred_image, pred_path)

        # Run inference for FDG inputs
        log.info("[FDG ] Running lesions segmentation inference")
        list_path_to_fdg_models = glob.glob("/app/nnUNet_results/*FDG*/**/*.pth", recursive=True)

        fdg_pred_image, fdg_physiological_image = execute_multiple_folds_lesions_segmentation(
            pt_image=data["fdg_pt_image"],
            ct_image=data["fdg_ct_image"],
            total_segmentator_image=data["fdg_organ_segmentation_image"],
            suv_threshold=data["fdg_pt_suv_threshold"],
            list_path_to_pth_for_tracer=list_path_to_fdg_models,
            tracer_name="FDG",
        )
        if postprocess_fdg_based_on_psma_classes:
            fdg_pred_image, psma_pred_image = final_postprocessing(
                fdg_pt_image=fdg_pt_image,
                fdg_pred_image=fdg_pred_image,
                fdg_organ_segmentation_image=fdg_organ_segmentation_image,
                psma_pt_image=psma_pt_image,
                psma_pred_image=psma_pred_image,
                psma_organ_segmentation_image=psma_organ_segmentation_image,
            )

        if REFINE_TTB_LABEL:
            fdg_pred_image = refine_my_ttb_label(
                ttb_image=fdg_pred_image,
                pet_image=data["fdg_pt_image"],
                totseg_multilabel=data["fdg_organ_segmentation_image"],
                expansion_radius_mm=5.0,
                pet_threshold_value=data["fdg_pt_suv_threshold"],
                totseg_non_expand_values=[1, 2, 3, 5, 21, 90],
                normal_image=fdg_physiological_image,
            )

        # NOTE: It appears this does not improve the performances
        if REFINE_FDG_BASED_ON_PSMA:
            fdg_pred_image = refine_fdg_prediction_from_psma_prediction(
                fdg_pt_image=data["fdg_pt_image"],
                fdg_pred_image=fdg_pred_image,
                fdg_totseg_image=data["fdg_organ_segmentation_image"],
                psma_pred_image=psma_pred_image,
                psma_totseg_image=data["psma_organ_segmentation_image"],
            )

        pred_path = Path(data["fdg_pred_path"])
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("Saving prediction to '%s'", pred_path)
        sitk.WriteImage(fdg_pred_image, pred_path)


def iter_grand_challenge_data(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> Generator[DataDict, None, None]:
    # Grand Challenge data have only one set of inputs, and the algorithm / docker is used for each set / exam
    input_dir = Path(input_dir)
    images_dir = input_dir / "images"
    psma_ct_path = find_file_path(images_dir / "psma-ct", ext=IMAGE_EXTS)
    psma_ct_image_organ_segmentation_path = find_file_path(images_dir / "psma-ct-organ-segmentation", ext=IMAGE_EXTS)
    psma_pt_path = find_file_path(images_dir / "psma-pet-ga-68", ext=IMAGE_EXTS)
    psma_pt_suv_threshold_path = input_dir / "psma-pet-suv-threshold.json"
    psma_pred_path = Path(output_dir, "images", "psma-pet-ttb", "output.mha")

    fdg_ct_path = find_file_path(images_dir / "fdg-ct", ext=IMAGE_EXTS)
    fdg_ct_image_organ_segmentation_path = find_file_path(images_dir / "fdg-ct-organ-segmentation", ext=IMAGE_EXTS)
    fdg_pt_path = find_file_path(images_dir / "fdg-pet", ext=IMAGE_EXTS)
    fdg_pt_suv_threshold_path = input_dir / "fdg-pet-suv-threshold.json"
    fdg_pred_path = Path(output_dir, "images", "fdg-pet-ttb", "output.mha")

    psma_to_fdg_registration_path = input_dir / "psma-to-fdg-registration.json"

    psma_to_fdg_registration = load_json(psma_to_fdg_registration_path)["3d_affine_transform"]
    psma_to_fdg_registration = np.array(psma_to_fdg_registration, dtype=np.float32)

    yield {
        "psma_ct_path": psma_ct_path.as_posix(),
        "psma_organ_segmentation_path": psma_ct_image_organ_segmentation_path.as_posix(),
        "psma_pt_path": psma_pt_path.as_posix(),
        "psma_pt_suv_threshold": load_json(psma_pt_suv_threshold_path),
        "psma_pred_path": psma_pred_path.as_posix(),
        "fdg_ct_path": fdg_ct_path.as_posix(),
        "fdg_organ_segmentation_path": fdg_ct_image_organ_segmentation_path.as_posix(),
        "fdg_pt_path": fdg_pt_path.as_posix(),
        "fdg_pt_suv_threshold": load_json(fdg_pt_suv_threshold_path),
        "fdg_pred_path": fdg_pred_path.as_posix(),
        "psma_to_fdg_registration": psma_to_fdg_registration,
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
            "psma_ct_path": row["psma_ct_path"],
            "psma_organ_segmentation_path": row["psma_organ_segmentation_path"],
            "psma_pt_path": row["psma_pt_path"],
            "psma_pt_suv_threshold": row["psma_pt_suv_threshold"],
            "psma_pred_path": row["psma_pred_path"],
            "fdg_ct_path": row["fdg_ct_path"],
            "fdg_organ_segmentation_path": row["fdg_organ_segmentation_path"],
            "fdg_pt_path": row["fdg_pt_path"],
            "fdg_pt_suv_threshold": row["fdg_pt_suv_threshold"],
            "fdg_pred_path": row["fdg_pred_path"],
            "psma_to_fdg_registration": row.get("psma_to_fdg_registration", None),
        }


if __name__ == "__main__":
    app()
