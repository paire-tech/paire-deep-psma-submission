import logging
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import SimpleITK as sitk

log = logging.getLogger(__name__)

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

DEFAULT_EXPANSION_RADIUS_MM = 7.0
DEFAULT_IGNORED_ORGAN_IDS = [1, 2, 3, 5, 21]

def mask_to_distance_map(mask_sitk: sitk.Image) -> sitk.Image:
    mask = sitk.Cast(mask_sitk > 0, sitk.sitkUInt8)
    dist = sitk.SignedMaurerDistanceMap(
        mask,
        insideIsPositive=True,
        squaredDistance=False,
        useImageSpacing=True,
    )
    return dist


def crop_sitk_to_mask(
    sitk_image: sitk.Image, sitk_mask: sitk.Image, except_in_dims: list[int] | None = None
) -> sitk.Image:
    """crop a sitk image to its foreground
    will be cropped to volumes where sitk_mask > 0"""
    sitk_mask_connected = sitk.ConnectedComponent(sitk_mask)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk_mask_connected)
    if len(stats.GetLabels()) == 0:
        print("WARNING in crop_sitk_to_mask, mask has no value > 0, do nothing")
        return sitk_image
    # take the largest connected component
    largest_label = None
    largest_size = 0
    for label in stats.GetLabels():
        size = stats.GetNumberOfPixels(label)
        if size > largest_size:
            largest_label = label
            largest_size = size
    bbox = stats.GetBoundingBox(largest_label)
    xmin, ymin, zmin, xsize, ysize, zsize = bbox
    slicer = (
        slice(xmin, xmin + xsize),
        slice(ymin, ymin + ysize),
        slice(zmin, zmin + zsize),
    )
    if except_in_dims is not None:
        slicer = list(slicer)  # type: ignore
        for dim in except_in_dims:
            slicer[dim] = slice(None, None)  # type: ignore
        slicer = tuple(slicer)  # type: ignore
    cropped_sitk_obj = sitk_image[slicer]
    return cropped_sitk_obj


def majority_vote_onehot(preds: np.ndarray, n_classes: int) -> np.ndarray:
    onehot = np.eye(n_classes, dtype=np.int32)[preds]  # (N, *S, K)
    counts = onehot.sum(axis=0)  # (*S, K)
    return counts.argmax(axis=-1)


def execute_multiple_folds_lesions_segmentation(
    pt_image: sitk.Image,
    ct_image: sitk.Image,
    total_segmentator_image: sitk.Image,
    suv_threshold: float,
    list_path_to_pth_for_tracer: list = [],  # ...or "checkpoint_final.pth"
    tracer_name: str = "PSMA",
) -> tuple[sitk.Image, sitk.Image]:
    pt_image = crop_sitk_to_mask(pt_image, pt_image > 0.05)  # in suv -> crop it

    ct_image = sitk.Resample(ct_image, pt_image, sitk.TranslationTransform(3), sitk.sitkLinear, -1000)
    organs_image_resampled = sitk.Resample(
        total_segmentator_image, pt_image, sitk.TranslationTransform(3), sitk.sitkNearestNeighbor
    )
    organs_image_resampled = sitk.ChangeLabel(organs_image_resampled, ORGANS_MAPPING)
    pt_image = pt_image / suv_threshold

    # list_probabilities = []
    list_preds = []
    for checkpoint in list_path_to_pth_for_tracer:
        *_, plan, arch = str(checkpoint).split("/")[-3].split("__")  # nnUNetTrainer__nnUNetResEncUNetLPlans__3d_fullres
        fold = str(checkpoint).split("/")[-2][-1]  # fold0
        dataset_id = str(checkpoint).split("/")[-4].split("_")[0].replace("Dataset", "")
        log.info("Running nnU-Net inference for %s", dataset_id)
        with TemporaryDirectory(dir=Path.cwd(), prefix="tmp_") as tmp_dir:
            input_dir = Path(tmp_dir, "input")
            output_dir = Path(tmp_dir, "output")
            Path(input_dir).mkdir(parents=True, exist_ok=True)
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            sitk.WriteImage(pt_image, input_dir / "deep-psma_0000.nii.gz")
            sitk.WriteImage(ct_image, input_dir / "deep-psma_0001.nii.gz")

            if dataset_id.startswith("9"):
                for i in [1, 2, 3, 4, 5, 6, 7, 8]:
                    organ_mask_image = sitk.Cast(organs_image_resampled == i, sitk.sitkUInt8)
                    if dataset_id.startswith("92"):
                        organ_mask_image = mask_to_distance_map(organ_mask_image)
                        # noramlize organ_mask_image to 0-1
                        organ_mask_data = sitk.GetArrayFromImage(organ_mask_image)
                        organ_mask_data /= 2.275830678197542
                        # sigmoid
                        organ_mask_data = 1 / (1 + np.exp(-organ_mask_data))
                        organ_mask_image = sitk.GetImageFromArray(organ_mask_data)
                        organ_mask_image.CopyInformation(organs_image_resampled)
                    sitk.WriteImage(organ_mask_image, input_dir / f"deep-psma_{i + 1:04d}.nii.gz")

            log.info(f"Running nnU-Net inference for with {checkpoint}")
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
                    arch,
                    "-p",
                    plan,
                    "-f",
                    str(fold),
                    "-chk",
                    checkpoint,
                    "-npp",
                    "0",
                    "-nps",
                    "0",
                    # "--save_probabilities",
                ],
                check=True,
            )

            pred_image = sitk.ReadImage(output_dir / "deep-psma.nii.gz")
            pred_data = sitk.GetArrayFromImage(pred_image)
            # probabilities = np.load(output_dir / "deep-psma.npz")["probabilities"]
            # list_probabilities.append(probabilities)  # 3, C, H, W
            list_preds.append(pred_data)
    # mean_probabilities = np.stack(list_probabilities, axis=0)
    # mean_probabilities = np.mean(mean_probabilities, axis=0)
    # if tracer_name == "FDG":
    #    pred_ttb_ar = (mean_probabilities[1, ...] > 0.33).astype("int8")
    # pred_norm_ar = (mean_probabilities[2, ...] > 0.66).astype("int8")
    # else:
    #    pred_ttb_ar = (mean_probabilities[1, ...] > 0.5).astype("int8")
    #    pred_norm_ar = (mean_probabilities[2, ...] > 0.5).astype("int8")
    preds_array = np.stack(list_preds, axis=0)
    if len(list_preds) > 1:
        if tracer_name == "PSMA":
            log.info("Using majority voting for PSMA with %d models", len(list_preds))
            preds_array = majority_vote_onehot(preds_array, 3)
        else:
            log.info("Using maximal voting for FDG with %d models", len(list_preds))
        preds_array = np.argmax(preds_array, axis=0)  # maximalist voting
    else:
        preds_array = preds_array[0]
    pred_ttb_ar = (preds_array == 1).astype("int8")
    pred_norm_image = (preds_array == 2).astype("int8")

    # pt_array = sitk.GetArrayFromImage(pt_image)
    # tar = (pt_array >= 1.0).astype("int8")

    # convert predicted TTB label to sitk format with spacing information to run grow/expansion function
    pred_ttb_label = sitk.GetImageFromArray(pred_ttb_ar)
    pred_ttb_label.CopyInformation(pred_image)

    pred_norm_image = sitk.GetImageFromArray(pred_norm_image)
    pred_norm_image.CopyInformation(pred_image)

    return pred_ttb_label, pred_norm_image


def expand_contract_label(label: sitk.Image, distance: float = 5.0) -> sitk.Image:
    lar = sitk.GetArrayFromImage(label)
    label_single = sitk.GetImageFromArray((lar > 0).astype("int16"))
    label_single.CopyInformation(label)
    distance_filter = sitk.SignedMaurerDistanceMapImageFilter()
    distance_filter.SetUseImageSpacing(True)
    distance_filter.SquaredDistanceOff()
    dmap = distance_filter.Execute(label_single)
    dmap_ar = sitk.GetArrayFromImage(dmap)
    new_label_ar = (dmap_ar <= distance).astype("int16")
    new_label = sitk.GetImageFromArray(new_label_ar)
    new_label.CopyInformation(label)
    return new_label


def refine_my_ttb_label(
    ttb_image: sitk.Image,
    pet_image: sitk.Image,
    totseg_multilabel: sitk.Image,
    expansion_radius_mm: float = 5.0,
    pet_threshold_value: float = 3.0,
    totseg_non_expand_values: list = [1, 2, 3, 5, 21],
    normal_image: sitk.Image | None = None,
) -> sitk.Image:
    """refine ttb label by expanding and rethresholding the original prediction
    set expansion radius mm to control distance that inferred TTB boundary is initially grown
    set pet_threshold_value to match designated value for ground truth contouring workflow
    (eg PSMA PET SUV=3).
    Includes option to avoid growing the label in certain
    tissue types in the total segmentator label (ex PSMA avoid expanding into liver, could
    include [2,3,5,21] to also avoid kidneys and urinary bladder)
    For other organ values see "total" class map from:
    https://github.com/wasserth/TotalSegmentator/blob/master/totalsegmentator/map_to_binary.py
    Lastly, possible to include the "normal" tissue inferred label from the baseline example
    algorithm and will similarly avoid expanding into this region"""
    ttb_original_array = sitk.GetArrayFromImage(
        ttb_image
    )  # original TTB array for inpainting back voxels in certain tissues
    ttb_expanded_image = expand_contract_label(
        ttb_image, expansion_radius_mm
    )  # expand inferred label with function above
    ttb_expanded_array = sitk.GetArrayFromImage(ttb_expanded_image)  # convert to numpy array
    pet_threshold_array = (
        sitk.GetArrayFromImage(pet_image) >= pet_threshold_value
    )  # get numpy array of PET image voxels above threshold value
    ttb_rethresholded_array = np.logical_and(
        ttb_expanded_array, pet_threshold_array
    )  # remove expanded TTB voxels below PET threshold

    # loop through total segmentator tissue #s and use original TTB prediction in those labels
    totseg_multilabel = sitk.Resample(
        totseg_multilabel, ttb_image, sitk.TranslationTransform(3), sitk.sitkNearestNeighbor, 0
    )
    totseg_multilabel_array = sitk.GetArrayFromImage(totseg_multilabel)
    for totseg_value in totseg_non_expand_values:
        # paint the original TTB prediction into the totseg tissue regions - probably of most relevance for PSMA liver VOI
        ttb_rethresholded_array[totseg_multilabel_array == totseg_value] = ttb_original_array[
            totseg_multilabel_array == totseg_value
        ]

    # check if inferred normal array is included and if so set TTB voxels to background
    if normal_image is not None:
        normal_array = sitk.GetArrayFromImage(normal_image)
        ttb_rethresholded_array[normal_array > 0] = 0
    ttb_rethresholded_image = sitk.GetImageFromArray(
        ttb_rethresholded_array.astype("int16")
    )  # create output image & copy information
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
    log.info("Starting final postprocessing")

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
