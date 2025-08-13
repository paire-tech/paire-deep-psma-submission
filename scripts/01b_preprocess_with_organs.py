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

# Data directory containing the challenge data
DATA_DIR = "/data/DEEP_PSMA_CHALLENGE_DATA/CHALLENGE_DATA"
# nnUNet configuration paths
NNUNET_RAW_DIR = os.environ["nnUNet_raw"]
NNUNET_PREPROCESSED_DIR = os.environ["nnUNet_preprocessed"]
NNUNET_RESULTS_DIR = os.environ["nnUNet_results"]
# Expected number of cases in the dataset
EXPECTED_NUM_CASES = 100
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
    # Show nnUNet configuration paths
    print("Using nnUNet configuration:")
    print(f"  nnUNet_raw: {NNUNET_RAW_DIR}")
    print(f"  nnUNet_preprocessed: {NNUNET_PREPROCESSED_DIR}")
    print(f"  nnUNet_results: {NNUNET_RESULTS_DIR}")
    print()

    dataset_files = scan_dataset_files(args.data_dir, args.tracer_name)
    dataset_name = f"Dataset{args.dataset_id}_{args.tracer_name}_PET"

    print(f"Found {len(dataset_files):,} cases for dataset {dataset_name!r}.")
    if len(dataset_files) != EXPECTED_NUM_CASES:
        print(f"WARNING! Expected {EXPECTED_NUM_CASES:,} cases, but found {len(dataset_files):,}.")

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
        totseg24_image = sitk.ReadImage(data["totseg24_path"])
        suv_threshold = load_json(data["threshold_path"])["suv_threshold"]

        # Generate the ground truth label based on the TTB and PET images
        gt_image = generate_ground_truth(ttb_image, pt_image, suv_threshold)
        gt_image = sitk.Cast(gt_image, sitk.sitkUInt8)

        # Resample the CT and TotalSegmentator images to match the PET image
        ct_image = sitk.Resample(ct_image, pt_image, sitk.TranslationTransform(3), sitk.sitkLinear, -1000)
        totseg24_image = sitk.Resample(totseg24_image, pt_image, sitk.TranslationTransform(3), sitk.sitkNearestNeighbor)
        organs_image = sitk.ChangeLabel(totseg24_image, ORGANS_MAPPING)

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
        for i in [1, 2, 3, 4, 5, 6, 7, 8]:
            organ_mask_image = sitk.Cast(organs_image == i, sitk.sitkUInt8)
            sitk.WriteImage(organ_mask_image, images_tr_dir / f"{case_name}_{i + 1:04d}.nii.gz")

        # Save the labels
        sitk.WriteImage(gt_image, labels_tr_dir / f"{case_name}.nii.gz")

    # Setup nnUNet required dataset.json file
    print("Saving dataset information")
    dataset_info = {
        "channel_names": {
            "0": "noNorm",
            "1": "CT",
            "2": "noNorm",
            "3": "noNorm",
            "4": "noNorm",
            "5": "noNorm",
            "6": "noNorm",
            "7": "noNorm",
            "8": "noNorm",
            "9": "noNorm",
        },
        "labels": {"background": 0, "ttb": 1, "norm": 2},
        "numTraining": len(dataset_files),
        "file_ending": ".nii.gz",
    }
    dataset_info_path = Path(NNUNET_RAW_DIR, dataset_name, "dataset.json")
    save_json(dataset_info, dataset_info_path)

    # Run nnUNet preprocessing command to preprocess the dataset
    subprocess.run(
        ["nnUNetv2_plan_and_preprocess", "-d", str(args.dataset_id), "-c", "3d_fullres", "--verify_dataset_integrity"],
        check=True,
    )
    print(f"Preprocessing completed for dataset {dataset_name}.")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Preprocess DeepPSMA dataset for nnUNet training.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DATA_DIR,
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
