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
# TotalSegmentator organs mapping classes
ORGANS_MAPPING = {
    0: 0,  #    unspecified                   -> unspecified
    1: 1,  #    spleen                        -> spleen
    2: 2,  #    kidney_right                  -> kidney
    3: 2,  #    kidney_left                   -> kidney
    4: 0,  #    gallbladder                   -> unspecified
    5: 3,  #    liver                         -> liver
    6: 0,  #    stomach                       -> unspecified
    7: 0,  #    pancreas                      -> unspecified
    8: 0,  #    adrenal_gland_right           -> unspecified
    9: 0,  #    adrenal_gland_left            -> unspecified
    10: 4,  #   lung_upper_lobe_left          -> lung
    11: 4,  #   lung_lower_lobe_left          -> lung
    12: 4,  #   lung_upper_lobe_right         -> lung
    13: 4,  #   lung_middle_lobe_right        -> lung
    14: 4,  #   lung_lower_lobe_right         -> lung
    15: 0,  #   esophagus                     -> unspecified
    16: 0,  #   trachea                       -> unspecified
    17: 8,  #   thyroid_gland                 -> unspecified
    18: 0,  #   small_bowel                   -> unspecified
    19: 0,  #   duodenum                      -> unspecified
    20: 0,  #   colon                         -> unspecified
    21: 6,  #   urinary_bladder               -> unspecified
    22: 0,  #   prostate                      -> unspecified
    23: 0,  #   kidney_cyst_left              -> unspecified
    24: 0,  #   kidney_cyst_right             -> unspecified
    25: 7,  #   sacrum                        -> osteo_medullar
    26: 7,  #   vertebrae_S1                  -> osteo_medullar
    27: 7,  #   vertebrae_L5                  -> osteo_medullar
    28: 7,  #   vertebrae_L4                  -> osteo_medullar
    29: 7,  #   vertebrae_L3                  -> osteo_medullar
    30: 7,  #   vertebrae_L2                  -> osteo_medullar
    31: 7,  #   vertebrae_L1                  -> osteo_medullar
    32: 7,  #   vertebrae_T12                 -> osteo_medullar
    33: 7,  #   vertebrae_T11                 -> osteo_medullar
    34: 7,  #   vertebrae_T10                 -> osteo_medullar
    35: 7,  #   vertebrae_T9                  -> osteo_medullar
    36: 7,  #   vertebrae_T8                  -> osteo_medullar
    37: 7,  #   vertebrae_T7                  -> osteo_medullar
    38: 7,  #   vertebrae_T6                  -> osteo_medullar
    39: 7,  #   vertebrae_T5                  -> osteo_medullar
    40: 7,  #   vertebrae_T4                  -> osteo_medullar
    41: 7,  #   vertebrae_T3                  -> osteo_medullar
    42: 7,  #   vertebrae_T2                  -> osteo_medullar
    43: 7,  #   vertebrae_T1                  -> osteo_medullar
    44: 7,  #   vertebrae_C7                  -> osteo_medullar
    45: 7,  #   vertebrae_C6                  -> osteo_medullar
    46: 7,  #   vertebrae_C5                  -> osteo_medullar
    47: 7,  #   vertebrae_C4                  -> osteo_medullar
    48: 7,  #   vertebrae_C3                  -> osteo_medullar
    49: 7,  #   vertebrae_C2                  -> osteo_medullar
    50: 7,  #   vertebrae_C1                  -> osteo_medullar
    51: 0,  #   heart                         -> unspecified
    52: 0,  #   aorta                         -> unspecified
    53: 0,  #   pulmonary_vein                -> unspecified
    54: 0,  #   brachiocephalic_trunk         -> unspecified
    55: 0,  #   subclavian_artery_right       -> unspecified
    56: 0,  #   subclavian_artery_left        -> unspecified
    57: 0,  #   common_carotid_artery_right   -> unspecified
    58: 0,  #   common_carotid_artery_left    -> unspecified
    59: 0,  #   brachiocephalic_vein_left     -> unspecified
    60: 0,  #   brachiocephalic_vein_right    -> unspecified
    61: 0,  #   atrial_appendage_left         -> unspecified
    62: 0,  #   superior_vena_cava            -> unspecified
    63: 0,  #   inferior_vena_cava            -> unspecified
    64: 0,  #   portal_vein_and_splenic_vein  -> unspecified
    65: 0,  #   iliac_artery_left             -> unspecified
    66: 0,  #   iliac_artery_right            -> unspecified
    67: 0,  #   iliac_vena_left               -> unspecified
    68: 0,  #   iliac_vena_right              -> unspecified
    69: 7,  #   humerus_left                  -> osteo_medullar
    70: 7,  #   humerus_right                 -> osteo_medullar
    71: 7,  #   scapula_left                  -> osteo_medullar
    72: 7,  #   scapula_right                 -> osteo_medullar
    73: 7,  #   clavicula_left                -> osteo_medullar
    74: 7,  #   clavicula_right               -> osteo_medullar
    75: 7,  #   femur_left                    -> osteo_medullar
    76: 7,  #   femur_right                   -> osteo_medullar
    77: 7,  #   hip_left                      -> osteo_medullar
    78: 7,  #   hip_right                     -> osteo_medullar
    79: 0,  #   spinal_cord                   -> unspecified
    80: 0,  #   gluteus_maximus_left          -> unspecified
    81: 0,  #   gluteus_maximus_right         -> unspecified
    82: 0,  #   gluteus_medius_left           -> unspecified
    83: 0,  #   gluteus_medius_right          -> unspecified
    84: 0,  #   gluteus_minimus_left          -> unspecified
    85: 0,  #   gluteus_minimus_right         -> unspecified
    86: 0,  #   autochthon_left               -> unspecified
    87: 0,  #   autochthon_right              -> unspecified
    88: 0,  #   iliopsoas_left                -> unspecified
    89: 0,  #   iliopsoas_right               -> unspecified
    90: 5,  #   brain                         -> brain
    91: 7,  #   skull                         -> osteo_medullar
    92: 7,  #   rib_left_1                    -> osteo_medullar
    93: 7,  #   rib_left_2                    -> osteo_medullar
    94: 7,  #   rib_left_3                    -> osteo_medullar
    95: 7,  #   rib_left_4                    -> osteo_medullar
    96: 7,  #   rib_left_5                    -> osteo_medullar
    97: 7,  #   rib_left_6                    -> osteo_medullar
    98: 7,  #   rib_left_7                    -> osteo_medullar
    99: 7,  #   rib_left_8                    -> osteo_medullar
    100: 7,  #  rib_left_9                    -> osteo_medullar
    101: 7,  #  rib_left_10                   -> osteo_medullar
    102: 7,  #  rib_left_11                   -> osteo_medullar
    103: 7,  #  rib_left_12                   -> osteo_medullar
    104: 7,  #  rib_right_1                   -> osteo_medullar
    105: 7,  #  rib_right_2                   -> osteo_medullar
    106: 7,  #  rib_right_3                   -> osteo_medullar
    107: 7,  #  rib_right_4                   -> osteo_medullar
    108: 7,  #  rib_right_5                   -> osteo_medullar
    109: 7,  #  rib_right_6                   -> osteo_medullar
    110: 7,  #  rib_right_7                   -> osteo_medullar
    111: 7,  #  rib_right_8                   -> osteo_medullar
    112: 7,  #  rib_right_9                   -> osteo_medullar
    113: 7,  #  rib_right_10                  -> osteo_medullar
    114: 7,  #  rib_right_11                  -> osteo_medullar
    115: 7,  #  rib_right_12                  -> osteo_medullar
    116: 7,  #  sternum                       -> osteo_medullar
    117: 0,  #  costal_cartilages             -> unspecified
}


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
        totseg24_path = Path(args.input_dir, row[f"{args.tracer_name.lower()}_organ_segmentation_path"])
        suv_threshold_path = Path(args.input_dir, row[f"{args.tracer_name.lower()}_pt_suv_threshold"])

        missing_files = [Path(p) for p in [pt_path, ct_path, totseg24_path, suv_threshold_path] if not p.exists()]
        if missing_files:
            print(f"ERROR! The following files are missing: {', '.join(str(p) for p in missing_files)}")
            continue

        pt_image = sitk.ReadImage(pt_path)
        ct_image = sitk.ReadImage(ct_path)
        totseg24_image = sitk.ReadImage(totseg24_path)
        suv_threshold = load_json(suv_threshold_path)["suv_threshold"]

        print(f"Predicting segmentation for case: {row['id']}")
        pred_image = predict_segmentation(
            pt_image=pt_image,
            ct_image=ct_image,
            totseg24_image=totseg24_image,
            dataset_id=args.dataset_id,
            suv_threshold=suv_threshold,
            fold=args.fold,
        )

        output_path = Path(args.output_dir, str(row["id"]), args.tracer_name.upper(), "PRED.nii.gz")
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
    totseg24_image: sitk.Image,
    dataset_id: int,
    suv_threshold: float,
    fold: int = 0,
) -> sitk.Image:
    pt_image = pt_image / suv_threshold
    ct_image = sitk.Resample(ct_image, pt_image, sitk.TranslationTransform(3), sitk.sitkLinear, -1000)
    totseg24_image = sitk.Resample(totseg24_image, pt_image, sitk.TranslationTransform(3), sitk.sitkNearestNeighbor, 0)
    organs_image = sitk.ChangeLabel(totseg24_image, ORGANS_MAPPING)

    with TemporaryDirectory(dir=Path.cwd(), prefix="tmp_") as tmp_dir:
        input_dir = Path(tmp_dir, "input")
        output_dir = Path(tmp_dir, "output")
        Path(input_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(pt_image, input_dir / "deep-psma_0000.nii.gz")
        sitk.WriteImage(ct_image, input_dir / "deep-psma_0001.nii.gz")
        for i in [1, 2, 3, 4, 5, 6, 7, 8]:
            organ_mask_image = sitk.Cast(organs_image == i, sitk.sitkUInt8)
            sitk.WriteImage(organ_mask_image, input_dir / f"deep-psma_{i + 1:04d}.nii.gz")

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
