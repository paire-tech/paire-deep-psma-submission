import json
import os
import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv(override=True)

# nnUNet configuration paths
NNUNET_RAW_DIR = os.environ["nnUNet_raw"]
NNUNET_PREPROCESSED_DIR = os.environ["nnUNet_preprocessed"]
NNUNET_RESULTS_DIR = os.environ["nnUNet_results"]
# Specify the splits_final.json file for reproducibility
SCRIPTS_DIR = Path(__file__).parent.as_posix()
SPLITS_FINAL_PATH = Path(SCRIPTS_DIR, "splits_final.json").as_posix()


def main() -> None:
    args = parse_args()
    print("Using nnUNet configuration:")
    print(f"  nnUNet_raw: {NNUNET_RAW_DIR}")
    print(f"  nnUNet_preprocessed: {NNUNET_PREPROCESSED_DIR}")
    print(f"  nnUNet_results: {NNUNET_RESULTS_DIR}")

    if not args.yes:
        if input("\nDo you want to continue? (y/N): ").lower() != "y":
            print("Exiting without preprocessing.")
            return

    if not Path(SPLITS_FINAL_PATH).exists():
        print(f"ERROR! {SPLITS_FINAL_PATH} does not exist. Please run nnUNet training first.")
        return

    dataset_name = f"Dataset{args.dataset_id}_{args.tracer_name}_PET"
    dataset_dir = Path(NNUNET_PREPROCESSED_DIR, dataset_name)

    print("\nExporting cross-validation splits_final.json for nnUNet!")
    splits_final_path = dataset_dir / "splits_final.json"
    splits_final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(SPLITS_FINAL_PATH, splits_final_path)

    print("\nExporting cross-validation splits for inference in CSV format!")
    splits_data = load_json(splits_final_path)
    for fold, fold_data in enumerate(splits_data):
        items = []
        for split, case_names in fold_data.items():
            for case_name in case_names:
                item = {
                    "psma_ct_path": f"{case_name}/PSMA/CT.nii.gz",
                    "psma_pt_path": f"{case_name}/PSMA/PET.nii.gz",
                    "psma_pt_ttb_path": f"{case_name}/PSMA/TTB.nii.gz",
                    "psma_pt_suv_threshold": f"{case_name}/PSMA/threshold.json",
                    "psma_organ_segmentation_path": f"{case_name}/PSMA/totseg_24.nii.gz",
                    "fdg_ct_path": f"{case_name}/FDG/CT.nii.gz",
                    "fdg_pt_path": f"{case_name}/FDG/PET.nii.gz",
                    "fdg_pt_ttb_path": f"{case_name}/FDG/TTB.nii.gz",
                    "fdg_pt_suv_threshold": f"{case_name}/FDG/threshold.json",
                    "fdg_organ_segmentation_path": f"{case_name}/FDG/totseg_24.nii.gz",
                    "split": split,
                    "fold": fold,
                }
                items.append(item)

        df_fold = pd.DataFrame(items)
        fold_path = dataset_dir / f"{args.tracer_name.lower()}_val_fold{fold}.csv"
        print(f"Writing fold {fold} to: {fold_path}")
        df_fold.to_csv(fold_path, index=False)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Export validation splits for nnUNet training in CSV format.")
    parser.add_argument(
        "--tracer-name",
        type=str,
        required=True,
        choices=["PSMA", "FDG"],
        help="Name of the tracer used for training (e.g., PSMA, FDG).",
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        required=True,
        help="ID of the dataset to export splits for (e.g., 801 for PSMA, 802 for FDG).",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically answer 'yes' to prompts (use with caution).",
    )
    return parser.parse_args()


def load_json(file_path: Path) -> dict:
    with open(file_path, "r") as file:
        return json.load(file)
