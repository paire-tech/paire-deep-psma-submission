import logging
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Sequence, TypedDict, Union

import numpy as np
import SimpleITK as sitk

log = logging.getLogger(__name__)


class PreprocessingConfig(TypedDict):
    pt: Literal[True]
    ct: Literal[True]
    organs: Literal["sdf_mask", "binary_mask", False]


class Config(TypedDict):
    id: str
    tracer_name: str
    dataset_id: int
    plan: str
    trainer: str
    config: str
    fold: int
    checkpoint: str
    preprocessing: PreprocessingConfig


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

PSMA_CONFIG: Config = {
    "id": "nnUNet-FDG-921",
    "tracer_name": "PSMA",
    "dataset_id": 921,
    "plan": "nnUNetResEncUNetMPlans",
    "trainer": "nnUNetTrainer_250epochs",
    "config": "3d_fullres",
    "fold": 2,
    "checkpoint": "checkpoint_final.pth",
    "preprocessing": {
        "pt": True,
        "ct": True,
        "organs": "sdf_mask",
    },
}

FDG_CONFIG: Config = {
    "id": "nnUNet-FDG-922",
    "tracer_name": "FDG",
    "dataset_id": 922,
    "plan": "nnUNetResEncUNetMPlans",
    "trainer": "nnUNetTrainer_250epochs",
    "config": "3d_fullres",
    "fold": 2,
    "checkpoint": "checkpoint_final.pth",
    "preprocessing": {
        "pt": True,
        "ct": True,
        "organs": "sdf_mask",
    },
}


def execute_lesions_segmentation_ensemble(
    pt_image: sitk.Image,
    ct_image: sitk.Image,
    organs_image: sitk.Image,
    suv_threshold: float,
    *,
    configs: Sequence[Config],
    device: str = "cuda",
) -> sitk.Image:
    log.info("Starting lesions segmentation ensemble!")

    log.info("Preprocessing inputs...")
    ct_image = sitk.Resample(ct_image, pt_image, sitk.TranslationTransform(3), sitk.sitkLinear, -1000)
    organs_image = sitk.Resample(organs_image, pt_image, sitk.TranslationTransform(3), sitk.sitkNearestNeighbor)
    organs_image = sitk.ChangeLabel(organs_image, ORGANS_MAPPING)
    pt_image = pt_image / suv_threshold

    with TemporaryDirectory(dir=Path.cwd(), prefix="tmp_") as tmp_dir:
        input_dir = Path(tmp_dir, "input")
        output_dir = Path(tmp_dir, "output")
        Path(input_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(pt_image, input_dir / "deep-psma_0000.nii.gz")
        sitk.WriteImage(ct_image, input_dir / "deep-psma_0001.nii.gz")

    raise NotImplementedError


def execute_lesions_segmentation(
    pt_image: sitk.Image,
    ct_image: sitk.Image,
    totseg_image: sitk.Image,
    suv_threshold: float,
    *,
    config: Config,
    device: str = "cuda",
    return_probabilities: bool = False,
) -> sitk.Image:
    log.info("Starting lesions segmentation!")

    log.info("Preprocessing inputs...")
    ct_image = sitk.Resample(ct_image, pt_image, sitk.TranslationTransform(3), sitk.sitkLinear, -1000)
    organs_image = sitk.Resample(totseg_image, pt_image, sitk.TranslationTransform(3), sitk.sitkNearestNeighbor)
    organs_image = sitk.ChangeLabel(organs_image, ORGANS_MAPPING)
    pt_image = pt_image / suv_threshold

    with TemporaryDirectory(dir=Path.cwd(), prefix="tmp_") as tmp_dir:
        input_dir = Path(tmp_dir, "input")
        output_dir = Path(tmp_dir, "output")
        Path(input_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(pt_image, input_dir / "deep-psma_0000.nii.gz")
        sitk.WriteImage(ct_image, input_dir / "deep-psma_0001.nii.gz")

        if config["preprocessing"]["organs"]:
            log.info("Preprocessing organs...")
            for channel_idx, organ_id in enumerate([1, 2, 3, 4, 5, 6, 7, 8], start=2):
                organ_mask_image = sitk.Cast(organs_image == organ_id, sitk.sitkUInt8)

                if config["preprocessing"]["organs"] == "binary_mask":
                    log.info("Using binary mask for organ %d", organ_id)

                if config["preprocessing"]["organs"] == "sdf_mask":
                    log.info("Using SDF mask for organ %d", organ_id)
                    # Use SignedMaurerDistanceMap to create a distance map for the organ mask
                    organ_mask_image = sitk.SignedMaurerDistanceMap(
                        organ_mask_image,
                        insideIsPositive=True,
                        squaredDistance=False,
                        useImageSpacing=True,
                    )
                    organ_mask_image = 1 / (1 + sitk.Exp(-organ_mask_image / 2.275830678197542))
                    organ_mask_image = sitk.Cast(organ_mask_image, sitk.sitkFloat32)

                sitk.WriteImage(organ_mask_image, input_dir / f"deep-psma_{channel_idx:04d}.nii.gz")

        nnunet_predict(
            input_dir=input_dir,
            output_dir=output_dir,
            dataset_id=config["dataset_id"],
            config=config["config"],
            trainer=config["trainer"],
            checkpoint=config["checkpoint"],
            plan=config["plan"],
            fold=config["fold"],
            device=device,
            save_probabilities=return_probabilities,
        )

        if return_probabilities:
            scores = np.load(output_dir / "deep-psma.npz")
            return scores["probabilities"]

        pred_image = sitk.ReadImage(output_dir / "deep-psma.nii.gz")

    pt_array = sitk.GetArrayFromImage(pt_image)
    tar = (pt_array >= 1.0).astype("int8")

    pred_ttb_ar = (sitk.GetArrayFromImage(pred_image) == 1).astype("int8")
    pred_norm_ar = (sitk.GetArrayFromImage(pred_image) == 2).astype("int8")

    # convert predicted TTB label to sitk format with spacing information to run grow/expansion function
    pred_ttb_label = sitk.GetImageFromArray(pred_ttb_ar)
    pred_ttb_label.CopyInformation(pred_image)

    # expand nnU-Net predicted disease region
    pred_ttb_label_expanded = expand_contract_label(pred_ttb_label, distance=7.0)
    pred_ttb_ar_expanded = sitk.GetArrayFromImage(pred_ttb_label_expanded)
    pred_ttb_ar_expanded = np.logical_and(pred_ttb_ar_expanded > 0, tar > 0)
    output_ar = np.logical_and(pred_ttb_ar_expanded > 0, pred_norm_ar == 0).astype("int8")

    output_label = sitk.GetImageFromArray(output_ar)
    output_label.CopyInformation(pred_image)
    output_label = sitk.Resample(output_label, pt_image, sitk.TranslationTransform(3), sitk.sitkNearestNeighbor, 0)

    return output_label


def nnunet_predict(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    dataset_id: int,
    config: str,
    trainer: str,
    checkpoint: str,
    plan: str,
    fold: int,
    device: str = "cuda",
    save_probabilities: bool = False,
) -> None:
    log.info("Running nnU-Net inference!")
    args = [
        "nnUNetv2_predict",
        "-i",
        Path(input_dir).as_posix(),
        "-o",
        Path(output_dir).as_posix(),
        "-d",
        str(dataset_id),
        "-c",
        config,
        "-tr",
        trainer,
        "-p",
        plan,
        "-f",
        str(fold),
        "-chk",
        checkpoint,
        "-device",
        device,
        "-npp",
        "0",
        "-nps",
        "0",
    ]
    if save_probabilities:
        args.append("--save_probabilities")

    log.info("Executing command: %s", " ".join(args))
    subprocess.run(args, check=True)


def nnunet_ensemble(
    input_dirs: Sequence[Union[str, Path]],
    output_dir: Union[str, Path],
) -> None:
    log.info("Running nnU-Net ensemble inference!")
    args = [
        "nnUNetv2_ensemble",
        "-i",
        *[Path(d).as_posix() for d in input_dirs],
        "-o",
        Path(output_dir).as_posix(),
    ]

    log.info("Executing command: %s", " ".join(args))
    subprocess.run(args, check=True)


def expand_contract_label(label_image: sitk.Image, distance: float) -> sitk.Image:
    label_array = sitk.GetArrayFromImage(label_image)
    label_single = sitk.GetImageFromArray((label_array > 0).astype("int16"))
    label_single.CopyInformation(label_image)
    distance_filter = sitk.SignedMaurerDistanceMapImageFilter()
    distance_filter.SetUseImageSpacing(True)
    distance_filter.SquaredDistanceOff()
    dmap = distance_filter.Execute(label_single)
    dmap_ar = sitk.GetArrayFromImage(dmap)
    new_label_ar = (dmap_ar <= distance).astype("int16")
    new_label = sitk.GetImageFromArray(new_label_ar)
    new_label.CopyInformation(label_image)
    return new_label


def refine_predicted_ttb_image(
    ttb_image: sitk.Image,
    pt_image: sitk.Image,
    totseg_image: sitk.Image,
    suv_threshold: float,
    expansion_radius_mm: float = 5.0,
    totseg_non_expand_values: list = [1, 2, 3, 5, 21],
    normal_image: sitk.Image | None = None,
) -> sitk.Image:
    ttb_original_array = sitk.GetArrayFromImage(ttb_image)
    ttb_expanded_image = expand_contract_label(ttb_image, expansion_radius_mm)
    ttb_expanded_array = sitk.GetArrayFromImage(ttb_expanded_image)
    pet_threshold_array = sitk.GetArrayFromImage(pt_image) >= suv_threshold
    ttb_rethresholded_array = np.logical_and(ttb_expanded_array, pet_threshold_array)

    totseg_image = sitk.Resample(totseg_image, ttb_image, sitk.TranslationTransform(3), sitk.sitkNearestNeighbor, 0)
    totseg_array = sitk.GetArrayFromImage(totseg_image)
    for totseg_value in totseg_non_expand_values:
        ttb_rethresholded_array[totseg_array == totseg_value] = ttb_original_array[totseg_array == totseg_value]

    if normal_image is not None:
        normal_array = sitk.GetArrayFromImage(normal_image)
        ttb_rethresholded_array[normal_array > 0] = 0

    ttb_rethresholded_image = sitk.GetImageFromArray(ttb_rethresholded_array.astype("int16"))
    ttb_rethresholded_image.CopyInformation(ttb_image)
    return ttb_rethresholded_image


def refine_fdg_prediction_from_psma_prediction(
    fdg_pt_image: sitk.Image,
    fdg_pred_image: sitk.Image,
    fdg_totseg_image: sitk.Image,
    psma_pred_image: sitk.Image,
    psma_totseg_image: sitk.Image,
) -> sitk.Image:
    """Removes FDG lesions (from connected components) if no corresponding lesion is present
    in PSMA for the same anatomical class (based on TotalSegmentator labels).
    """
    tic = time.monotonic()
    log.info("Starting FDG refinement from PSMA prediction!")

    # Ensure the organs are in the same space as the predictions
    fdg_totseg_image = sitk.Resample(
        fdg_totseg_image,
        fdg_pred_image,
        sitk.TranslationTransform(3),
        sitk.sitkNearestNeighbor,
        0,
    )
    psma_totseg_image = sitk.Resample(
        psma_totseg_image,
        psma_pred_image,
        sitk.TranslationTransform(3),
        sitk.sitkNearestNeighbor,
        0,
    )

    fdg_pred_image = sitk.ConnectedComponent(fdg_pred_image)
    fdg_pred_array = sitk.GetArrayFromImage(fdg_pred_image)
    fdg_num_lesions = int(fdg_pred_array.max())
    fdg_totseg_array = sitk.GetArrayFromImage(fdg_totseg_image)

    psma_pred_array = sitk.GetArrayFromImage(psma_pred_image) > 0
    psma_totseg_array = sitk.GetArrayFromImage(psma_totseg_image)
    psma_totseg_labels = np.unique(psma_totseg_array[psma_pred_array])

    fdg_pt_array = sitk.GetArrayFromImage(fdg_pt_image)

    # Used to log some statistics about the FDG post-processing
    stats = []
    # Allocate memory for post-processed FDG prediction
    fdg_out_array = np.zeros_like(fdg_pred_array, dtype=np.uint8)

    for fdg_lesion_id in range(1, fdg_num_lesions + 1):
        fdg_lesion_mask = fdg_pred_array == fdg_lesion_id
        suvmax = fdg_pt_array[fdg_lesion_mask].max()
        if not fdg_lesion_mask.any():
            continue

        # Get class labels from TotalSegmentator for this lesion
        fdg_totseg_labels = np.unique(fdg_totseg_array[fdg_lesion_mask])

        kept = any(label in psma_totseg_labels for label in fdg_totseg_labels if label != 0)
        volume = np.sum(fdg_lesion_mask) * np.prod(fdg_pred_image.GetSpacing()) / 1000
        # the idea is to remove lesions that are only in one total segmentators classes
        # that do not match any totalsegmentator of psma
        if kept | (volume > 10.0) | (suvmax > 10):
            fdg_out_array[fdg_lesion_mask] = 1

        stats.append(
            {
                "lesion_id": fdg_lesion_id,
                "lesion_kept": kept,
                "lesion_volume": volume,
            }
        )

    volume_removed = sum(stat["lesion_volume"] for stat in stats if not stat["lesion_kept"])
    num_removed_lesions = sum(1 for stat in stats if not stat["lesion_kept"])
    log.info(
        "FDG post-processing: %d lesions out of %d were removed (volume of %.2f mm3)",
        num_removed_lesions,
        fdg_num_lesions,
        volume_removed,
    )

    fdg_out_image = sitk.GetImageFromArray(fdg_out_array.astype(np.uint8))
    fdg_out_image.CopyInformation(fdg_pred_image)

    log.info("Final postprocessing completed in %.2f seconds", time.monotonic() - tic)
    return fdg_out_image
