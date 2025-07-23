import logging
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import SimpleITK as sitk

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
                "-f",
                str(fold),
            ],
            check=True,
        )

        pred_image = sitk.ReadImage(output_dir / "deep-psma.nii.gz")

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
