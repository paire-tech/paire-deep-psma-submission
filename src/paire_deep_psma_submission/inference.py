import logging
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import SimpleITK as sitk
from typing_extensions import deprecated

log = logging.getLogger(__name__)

DATASET_MAPPING = {
    "PSMA": "801",
    "FDG": "802",
}

DEFAULT_EXPANSION_RADIUS_MM = 7.0
DEFAULT_IGNORED_ORGAN_IDS = [1, 2, 3, 5, 21]

def execute_lesions_segmentation(
    pt_image: sitk.Image,
    ct_image: sitk.Image,
    tracer_name: str = "PSMA",
    suv_threshold: float = 3.0,
    fold: str = "0",
    expansion_radius: float = 7.0,
) -> sitk.Image:
    log.info("Starting lesions segmentation for %s", tracer_name)

    image_filter = sitk.ResampleImageFilter()
    image_filter.SetReferenceImage(pt_image)
    image_filter.SetDefaultPixelValue(-1000)
    ct_image = image_filter.Execute(ct_image)
    pt_image = pt_image / suv_threshold

    with TemporaryDirectory(dir=Path.cwd(), prefix="tmp_") as tmp_dir:
        input_dir = Path(tmp_dir, "input")
        output_dir = Path(tmp_dir, "output")
        Path(input_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(pt_image, input_dir / "deep-psma_0000.nii.gz")
        sitk.WriteImage(ct_image, input_dir / "deep-psma_0001.nii.gz")

        log.info("Running nnU-Net inference for %s", tracer_name)
        subprocess.run(
            [
                "nnUNetv2_predict",
                "-i",
                input_dir.as_posix(),
                "-o",
                output_dir.as_posix(),
                "-d",
                DATASET_MAPPING[tracer_name],
                "-c",
                "3d_fullres",
                "-p",
                "nnUNetResEncUNetLPlans",
                "-f",
                str(fold),
                "-npp",
                "0",
                "-nps",
                "0",
                "--save_probabilities",
            ],
            check=True,
        )

        pred_image = sitk.ReadImage(output_dir / "deep-psma.nii.gz")
    if tracer_name == "FDG":
        pred_image = pred_image > .33
    elif tracer_name == "PSMA":
        pred_image = pred_image > .5
        

    pt_array = sitk.GetArrayFromImage(pt_image)
    tar = (pt_array >= 1.0).astype("int8")

    pred_ttb_ar = (sitk.GetArrayFromImage(pred_image) == 1).astype("int8")
    pred_norm_ar = (sitk.GetArrayFromImage(pred_image) == 2).astype("int8")

    # convert predicted TTB label to sitk format with spacing information to run grow/expansion function
    pred_ttb_label = sitk.GetImageFromArray(pred_ttb_ar)
    pred_ttb_label.CopyInformation(pred_image)

    # expand nnU-Net predicted disease region
    pred_ttb_label_expanded = expand_contract_label(pred_ttb_label, distance=expansion_radius)
    pred_ttb_ar_expanded = sitk.GetArrayFromImage(pred_ttb_label_expanded)  # get array from TTB expanded sitk image
    pred_ttb_ar_expanded = np.logical_and(pred_ttb_ar_expanded > 0, tar > 0)  # re-threshold expanded disease region

    output_ar = np.logical_and(pred_ttb_ar_expanded > 0, pred_norm_ar == 0).astype("int8")

    output_label = sitk.GetImageFromArray(output_ar)
    output_label.CopyInformation(pred_image)
    output_label = sitk.Resample(output_label, pt_image, sitk.TranslationTransform(3), sitk.sitkNearestNeighbor, 0)
    
    return output_label


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


@deprecated("This function is deprecated as it does not improve the performances.", category=FutureWarning)
def refine_fdg_prediction_from_psma_prediction(
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

    # Used to log some statistics about the FDG post-processing
    stats = []
    # Allocate memory for post-processed FDG prediction
    fdg_out_array = np.zeros_like(fdg_pred_array, dtype=np.uint8)

    for fdg_lesion_id in range(1, fdg_num_lesions + 1):
        fdg_lesion_mask = fdg_pred_array == fdg_lesion_id
        if not fdg_lesion_mask.any():
            continue

        # Get class labels from TotalSegmentator for this lesion
        fdg_totseg_labels = np.unique(fdg_totseg_array[fdg_lesion_mask])

        kept = any(label in psma_totseg_labels for label in fdg_totseg_labels if label != 0)
        # the idea is to remove lesions that are only in one total segmentators classes
        # that do not match any totalsegmentator of psma
        if kept:
            fdg_out_array[fdg_lesion_mask] = 1

        stats.append(
            {
                "lesion_id": fdg_lesion_id,
                "lesion_kept": kept,
                "lesion_volume": np.sum(fdg_lesion_mask) * np.prod(fdg_pred_image.GetSpacing()),
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
