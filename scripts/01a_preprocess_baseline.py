import json
import os
import subprocess
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import SimpleITK as sitk
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(override=True)

# nnUNet configuration paths
NNUNET_RAW_DIR = os.environ["nnUNet_raw"]
NNUNET_PREPROCESSED_DIR = os.environ["nnUNet_preprocessed"]
NNUNET_RESULTS_DIR = os.environ["nnUNet_results"]


def main() -> None:
    args = parse_args()
    # Show nnUNet configuration paths
    print("Using nnUNet configuration:")
    print(f"  nnUNet_raw: {NNUNET_RAW_DIR}")
    print(f"  nnUNet_preprocessed: {NNUNET_PREPROCESSED_DIR}")
    print(f"  nnUNet_results: {NNUNET_RESULTS_DIR}")
    print()

    dataset_files = scan_dataset_files(args.data_dir, args.tracer_name)
    dataset_name = f"Dataset{args.dataset_id}_{args.tracer_name}_PET"
    print(f"Found {len(dataset_files):,} cases for dataset {dataset_name!r}.")

    print(f"Will preprocess {len(dataset_files)} in {NNUNET_RAW_DIR} directory.")
    if not args.yes:
        if input("Do you want to continue? (y/N): ").lower() != "y":
            print("Exiting without preprocessing.")
            return

    print()
    pbar = tqdm(dataset_files, total=len(dataset_files), desc=f"Preprocessing {args.tracer_name}")
    for data in pbar:
        case_name = data["name"]
        pbar.set_postfix({"case": case_name})

        # Prepare nnUNet raw data
        ct_image = sitk.ReadImage(data["ct_path"])
        pt_image = sitk.ReadImage(data["pt_path"])
        ttb_image = sitk.ReadImage(data["ttb_path"])
        suv_threshold = load_json(data["threshold_path"])["suv_threshold"]

        # Generate the ground truth label based on the TTB and PET images
        gt_image = generate_ground_truth(ttb_image, pt_image, suv_threshold)
        gt_image = sitk.Cast(gt_image, sitk.sitkUInt8)

        # Resample the CT and TotalSegmentator images to match the PET image
        ct_image = sitk.Resample(ct_image, pt_image, sitk.TranslationTransform(3), sitk.sitkLinear, -1000)

        # Normalize the PET image by the SUV threshold
        pt_image = pt_image / suv_threshold

        # Save the preprocessed images
        images_tr_dir = Path(NNUNET_RAW_DIR, dataset_name, "imagesTr")
        labels_tr_dir = Path(NNUNET_RAW_DIR, dataset_name, "labelsTr")
        images_tr_dir.mkdir(parents=True, exist_ok=True)
        labels_tr_dir.mkdir(parents=True, exist_ok=True)

        # Save the training images
        # 0: PET image, 1: CT image, 2-10: organ masks
        sitk.WriteImage(pt_image, images_tr_dir / f"{case_name}_0000.nii.gz")
        sitk.WriteImage(ct_image, images_tr_dir / f"{case_name}_0001.nii.gz")

        # Save the labels
        sitk.WriteImage(gt_image, labels_tr_dir / f"{case_name}.nii.gz")

    # Setup nnUNet required dataset.json file
    print("Saving dataset information")
    dataset_info = {
        "channel_names": {"0": "noNorm", "1": "CT"},
        "labels": {"background": 0, "ttb": 1, "norm": 2},
        "numTraining": len(dataset_files),
        "file_ending": ".nii.gz",
    }
    dataset_info_path = Path(NNUNET_RAW_DIR, dataset_name, "dataset.json")
    save_json(dataset_info, dataset_info_path)

    # Run nnUNet preprocessing command to preprocess the dataset
    subprocess.run(
        [
            "nnUNetv2_plan_and_preprocess",
            "-d",
            str(args.dataset_id),
            "-c",
            "3d_fullres",
            "--verify_dataset_integrity",
            "-pl",
            "nnUNetPlannerResEncL",
        ],
        check=True,
    )
    print(f"Preprocessing completed for dataset {dataset_name}.")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Preprocess DeepPSMA dataset for nnUNet training.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing the DeepPSMA dataset.",
    )
    parser.add_argument(
        "--tracer-name",
        type=str,
        required=True,
        choices=["PSMA", "FDG"],
        help="Tracer name to preprocess.",
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        required=True,
        choices=[801, 802],
        help="nnUNet dataset ID corresponding to the tracer.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically answer 'yes' to prompts.",
    )
    return parser.parse_args()


def generate_ground_truth(ttb_image: sitk.Image, pt_image: sitk.Image, suv_threshold: float) -> sitk.Image:
    """Generate ground truth labels from instance segmentation TTB image, PET image and the SUV threshold.
    Generated labels:
    - 0 for background
    - 1 for above threshold and annotated as disease
    - 2 for normal tissue above threshold
    """
    pt_normalized = pt_image / suv_threshold
    pt_normalized_array = sitk.GetArrayFromImage(pt_normalized)
    ttb_array = sitk.GetArrayFromImage(ttb_image)
    ttb_normal_array = np.zeros_like(ttb_array)

    # 0 for background, 1 for above threshold and annotated as disease, 2 for normal tissue above threshold
    ttb_normal_array[ttb_array > 0] = 1
    ttb_normal_array[np.logical_and(pt_normalized_array >= 1.0, ttb_array == 0)] = 2
    ttb_normal_label = sitk.GetImageFromArray(ttb_normal_array)
    ttb_normal_label.CopyInformation(ttb_image)

    return ttb_normal_label


def scan_dataset_files(data_dir: str, tracer_name: str) -> List[Dict[str, Any]]:
    """Scan the data directory to retrieve paths for CT, PET etc. associated to the specified tracer."""
    dataset_files = []
    for case_path in sorted(Path(data_dir).iterdir()):
        if not case_path.is_dir():
            continue

        data = {
            "name": case_path.name,
            "tracer_name": tracer_name,
            "ct_path": case_path / tracer_name / "CT.nii.gz",
            "pt_path": case_path / tracer_name / "PET.nii.gz",
            "totseg24_path": case_path / tracer_name / "totseg_24.nii.gz",
            "ttb_path": case_path / tracer_name / "TTB.nii.gz",
            "threshold_path": case_path / tracer_name / "threshold.json",
        }
        dataset_files.append(data)

    return dataset_files


def load_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
