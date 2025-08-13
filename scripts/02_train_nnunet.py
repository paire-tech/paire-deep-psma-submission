import os
import subprocess
from argparse import ArgumentParser, Namespace

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
    print()

    if not args.yes:
        if input("Do you want to continue? (y/N): ").lower() != "y":
            print("Exiting without preprocessing.")
            return

    print(f"Starting nnUNet training for dataset ID {args.dataset_id} with fold {args.fold}!\n")
    subprocess.run(
        ["nnUNetv2_train", str(args.dataset_id), "3d_fullres", str(args.fold), "-device", args.device],
        check=True,
    )


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Train nnUNet models for PET tracers.")
    parser.add_argument(
        "--dataset-id",
        type=int,
        required=True,
        help="ID of the dataset to train on (e.g., 1001 for PSMA, 1002 for FDG).",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="Fold number for cross-validation (default: 0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to use for training (default: 'cuda').",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically answer 'yes' to prompts (use with caution).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
