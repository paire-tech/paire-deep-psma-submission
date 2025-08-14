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
    # Show nnUNet configuration paths
    print("Using nnUNet configuration:")
    print(f"  nnUNet_raw: {NNUNET_RAW_DIR}")
    print(f"  nnUNet_preprocessed: {NNUNET_PREPROCESSED_DIR}")
    print(f"  nnUNet_results: {NNUNET_RESULTS_DIR}")
    print()

    if not args.yes:
        if input("Do you want to continue? (y/N): ").lower() != "y":
            print("Exiting without preprocessing.")
            return

    # https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md#how-to-use-the-new-presets
    subprocess.run(
        ["nnUNetv2_plan_experiment", "-d", str(args.dataset_id), "-pl", args.plan],
        check=True,
    )


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Preprocess DeepPSMA dataset for nnUNet training.")
    parser.add_argument(
        "--dataset-id",
        type=int,
        required=True,
        help="nnUNet dataset ID corresponding to the tracer (e.g. 801 for PSMA, 802 for FDG).",
    )
    parser.add_argument(
        "--plan",
        type=str,
        default="nnUNetPlans",
        help="The plan to use for training (e.g., 'nnUNetPlannerResEncL').",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically answer 'yes' to prompts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
