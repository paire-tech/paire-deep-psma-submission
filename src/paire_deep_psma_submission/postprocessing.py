import logging
import math
import time
from typing import Any, Dict, Sequence

import numpy as np
import SimpleITK as sitk

log = logging.getLogger(__name__)


def predict_proba_once(
    x: Sequence[float],
    means: Sequence[float],
    stds: Sequence[float],
    coef: Sequence[float],
    intercept: float,
) -> float:
    # x = [kept(0/1), suvmax, volume]
    xn = [(xi - m) / s for xi, m, s in zip(x, means, stds)]
    z = intercept + sum(c * xi for c, xi in zip(coef, xn))
    return 1.0 / (1.0 + math.exp(-z))


def predict_label(x: Sequence[float], params: Dict[str, Any], threshold: float = 0.5) -> bool:
    p = predict_proba_once(x, params["means"], params["stds"], params["coef"], params["intercept"])
    return p >= threshold


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
        volume = np.sum(fdg_lesion_mask) * np.prod(fdg_pred_image.GetSpacing()) / 1000
        # Get class labels from TotalSegmentator for this lesion
        fdg_totseg_labels = np.unique(fdg_totseg_array[fdg_lesion_mask])
        if (len(fdg_totseg_labels)) == 1 and (fdg_totseg_labels.sum() == 0):
            is_predicted_true_positive = True
        else:
            is_in_psma = any(label in psma_totseg_labels for label in fdg_totseg_labels if label != 0)
            logvolume = np.log(volume + 1)
            if (len(fdg_totseg_labels)) == 1 and (fdg_totseg_labels.sum() == 0):
                is_predicted_true_positive = True
            else:
                is_in_psma = any(label in psma_totseg_labels for label in fdg_totseg_labels if label != 0)
                logvolume = np.log(volume + 1)
                params = {
                    "means": [0, 0, 0],
                    "stds": [1, 1, 1],
                    "coef": [3.345112414198748, 0.6979599985346014, 0.836256971157578],
                    "intercept": -5.405821569685443,
                }

                is_predicted_true_positive = predict_label(
                    [is_in_psma * 1.0, logvolume, suvmax],
                    params,
                    threshold=0.5,
                )

        if is_predicted_true_positive | (volume > 7.0):
            fdg_out_array[fdg_lesion_mask] = 1

        stats.append(
            {
                "lesion_id": fdg_lesion_id,
                "lesion_kept": is_predicted_true_positive,
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
