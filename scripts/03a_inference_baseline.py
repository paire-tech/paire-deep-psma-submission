import json
import os
import subprocess
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Union

import pandas as pd
import SimpleITK as sitk
from dotenv import load_dotenv

load_dotenv(override=True)

# nnUNet configuration paths
NNUNET_RAW_DIR = os.environ["nnUNet_raw"]
NNUNET_PREPROCESSED_DIR = os.environ["nnUNet_preprocessed"]
NNUNET_RESULTS_DIR = os.environ["nnUNet_results"]


def main() -> None:
    args = parse_args()
    print("Using nnUNet configuration:")
    print(f"  nnUNet_raw: {NNUNET_RAW_DIR}")
    print(f"  nnUNet_preprocessed: {NNUNET_PREPROCESSED_DIR}")
    print(f"  nnUNet_results: {NNUNET_RESULTS_DIR}")

    df = pd.read_csv(args.input_csv)

    print(f"Will run inference on {len(df)} cases from {args.input_csv}.")
    if not args.yes:
        if input("Do you want to continue? (y/N): ").lower() != "y":
            print("Exiting without preprocessing.")
            return

    print("\nRunning inference with nnUNet!\n")
    for _, row in df.iterrows():
        pt_path = Path(args.input_dir, row[f"{args.tracer_name.lower()}_pt_path"])
        ct_path = Path(args.input_dir, row[f"{args.tracer_name.lower()}_ct_path"])
        suv_threshold_path = Path(args.input_dir, row[f"{args.tracer_name.lower()}_pt_suv_threshold"])

        missing_files = [Path(p) for p in [pt_path, ct_path, suv_threshold_path] if not p.exists()]
        if missing_files:
            print(f"ERROR! The following files are missing: {', '.join(str(p) for p in missing_files)}")
            continue

        pt_image = sitk.ReadImage(pt_path)
        ct_image = sitk.ReadImage(ct_path)
        suv_threshold = load_json(suv_threshold_path)["suv_threshold"]

        print(f"Predicting segmentation for case: {row['id']}")
        pred_image = predict_segmentation(
            pt_image=pt_image,
            ct_image=ct_image,
            dataset_id=args.dataset_id,
            suv_threshold=suv_threshold,
            fold=args.fold,
        )

        output_path = Path(args.output_dir, str(row["id"]), args.tracer_name.upper(), "PREDS.nii.gz")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(pred_image, output_path)
        print(f"-> Saved prediction to: {output_path}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Run inference and save predictions to the desired output directory.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the directory containing the input data (e.g. DEEP_PSMA_CHALLENGE_DATA).",
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to the CSV file containing the input data to use for inference (e.g. provide the validation fold).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the directory where the output predictions will be saved.",
    )
    parser.add_argument(
        "--tracer-name",
        type=str,
        required=True,
        choices=["PSMA", "FDG"],
        help="Name of the tracer used in the dataset (e.g. PSMA or FDG).",
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        required=True,
        help="ID of the dataset to use for inference (e.g. 801 for PSMA, 802 for FDG).",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold number to use for inference (default: 0).",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically answer 'yes' to prompts (use with caution).",
    )
    return parser.parse_args()


def predict_segmentation(
    pt_image: sitk.Image,
    ct_image: sitk.Image,
    dataset_id: int,
    suv_threshold: float,
    fold: int = 0,
) -> sitk.Image:
    pt_image = pt_image / suv_threshold
    ct_image = sitk.Resample(ct_image, pt_image, sitk.TranslationTransform(3), sitk.sitkLinear, -1000)

    with TemporaryDirectory(dir=Path.cwd(), prefix="tmp_") as tmp_dir:
        input_dir = Path(tmp_dir, "input")
        output_dir = Path(tmp_dir, "output")
        Path(input_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(pt_image, input_dir / "deep-psma_0000.nii.gz")
        sitk.WriteImage(ct_image, input_dir / "deep-psma_0001.nii.gz")

        subprocess.run(
            [
                "nnUNetv2_predict",
                "-i",
                input_dir.as_posix(),
                "-o",
                output_dir.as_posix(),
                "-d",
                str(dataset_id),
                "-c",
                "3d_fullres",
                "-f",
                str(fold),
                "-npp",
                "0",
                "-nps",
                "0",
            ],
            check=True,
        )

        return sitk.ReadImage(output_dir / "deep-psma.nii.gz")


def load_json(file_path: Union[str, Path]) -> Any:
    with open(file_path, "r") as f:
        return json.load(f)
