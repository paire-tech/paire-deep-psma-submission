import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

import pandas as pd


def main() -> None:
    args = parse_args()
    if not Path(args.splits_path).exists():
        print(f"ERROR! {args.splits_path} does not exist.")
        return

    print("\nExporting cross-validation splits for inference in CSV format!")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    splits_data = load_json(args.splits_path)
    for fold, fold_data in enumerate(splits_data):
        items = []
        for split, case_names in fold_data.items():
            for case_name in case_names:
                item = {
                    "id": case_name,
                    "split": split,
                    "fold": fold,
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
                }
                items.append(item)

        df_fold = pd.DataFrame(items)

        fold_path = Path(args.output_dir, f"val_fold{fold}.csv")
        print(f"-> Writing val fold {fold} to: {fold_path}")
        df_fold[df_fold["split"] == "val"].to_csv(fold_path, index=False)

        fold_path = Path(args.output_dir, f"train_fold{fold}.csv")
        print(f"-> Writing train fold {fold} to: {fold_path}")
        df_fold[df_fold["split"] == "train"].to_csv(fold_path, index=False)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Export validation splits for nnUNet training in CSV format.")
    parser.add_argument(
        "--splits-path",
        type=str,
        required=True,
        help="Path to the splits_final.json file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the exported splits CSV files.",
    )
    return parser.parse_args()


def load_json(file_path: Path) -> dict:
    with open(file_path, "r") as file:
        return json.load(file)


if __name__ == "__main__":
    main()
