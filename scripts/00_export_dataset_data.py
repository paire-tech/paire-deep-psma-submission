import shutil
from argparse import ArgumentParser, Namespace
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

INPUT_DIR = "/ssd/datasets/DeepPSMA/Task000_Baseline/data/raw/DEEP_PSMA"
OUTPUT_DIR = "/data/DEEP_PSMA_CHALLENGE_DATA/CHALLENGE_DATA"
STUDY_ID_TO_TRACER_NAME = {
    "00000000": "FDG",
    "00000001": "PSMA",
}


def main() -> None:
    args = parse_args()

    if not Path(args.input_dir).exists():
        raise FileNotFoundError(f"Input directory {args.input_dir} does not exist.")
    if not Path(args.output_dir).exists():
        raise FileNotFoundError(f"Output directory {args.output_dir} does not exist.")

    prompt = (
        f"This will export the DeepPSMA dataset from {args.input_dir} to {args.output_dir}. "
        "Do you want to continue? (y/n): "
    )
    if not args.yes and input(prompt).lower() != "y":
        print("Operation cancelled.")
        return

    print("\nStarting exporting!\n")

    for patient_dir in sorted(Path(args.input_dir).iterdir()):
        if not patient_dir.is_dir():
            continue

        case_name = patient_dir.name
        print(f"Processing {case_name}")
        for study_date, tracer_name in STUDY_ID_TO_TRACER_NAME.items():
            src_gt_path = patient_dir / study_date / "GT" / "DeepPSMA" / "volume.nii.gz"
            dst_ttb_path = Path(args.output_dir) / case_name / tracer_name / "TTB_fixed.nii.gz"
            if args.force or not dst_ttb_path.exists():
                print(f"  Copying {src_gt_path} -> {dst_ttb_path}")
                dst_ttb_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_gt_path, dst_ttb_path)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Preprocess DeepPSMA dataset for nnUNet training.")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the input directory containing the (fixed) DeepPSMA dataset, in PaIRe format.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to the output directory where the preprocessed dataset will be saved, in original format.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Automatically answer 'yes' to prompts.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite existing files in the output directory.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
