import itertools
import logging
import time
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import monai.transforms as T
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.inferers import sliding_window_inference
from torch import Tensor

from .transforms import Divided, LogicalAndd, SITKChangeLabeld, SITKResampleToMatchd, SITKToTensord, Thresholdd, ToSITKd

log = logging.getLogger(__name__)

# special params used in dict-based transforms
PT_KEY = "ct"
CT_KEY = "pt"
ORGANS_KEY = "organs_segmentation"
PT_MASK_KEY = "pt_mask"
IMAGE_KEY = "image"
PRED_TTB_KEY = "pred_ttb"

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


@torch.inference_mode()
def execute_lesions_segmentation(
    ct_image: sitk.Image,
    pt_image: sitk.Image,
    organs_segmentation_image: sitk.Image,
    suv_threshold: float,
    model: nn.Module,
    device: str = "cpu",
    use_mixed_precision: bool = False,
) -> sitk.Image:
    # Preprocess the inputs
    log.debug("Starting preprocessing")
    tic = time.monotonic()
    image, pt_mask = preprocess(pt_image, ct_image, organs_segmentation_image, suv_threshold, return_pt_mask=True)
    pad_widths = [(0, 0)] + divisible_pad_widths(image.shape[1:], k=32)
    image = pad_tensor(image, pad_widths, mode="constant", value=0.0)
    log.debug("Preprocessing completed in %.2f seconds", time.monotonic() - tic)

    image = image.to(device)
    model = model.to(device)

    tic = time.monotonic()
    log.info("Starting inference on '%s' device with input %s", device, tuple(image.shape))
    model.eval()
    with torch.amp.autocast(device_type=device, enabled=use_mixed_precision, cache_enabled=False):
        logits = sliding_window_inference(
            inputs=image.unsqueeze(0),
            predictor=model,
            roi_size=[128, 96, 224],
            sw_batch_size=4,
            overlap=0.25,
            mode="constant",
        )

    pred_tensor = torch.argmax(logits.float(), dim=1, keepdim=True).squeeze(0)  # type: ignore[union-attr]
    log.debug("Inference completed in %.2f seconds", time.monotonic() - tic)

    # Postprocess the prediction
    tic = time.monotonic()
    log.debug("Starting postprocessing")
    pred_tensor = unpad_tensor(pred_tensor, pad_widths)
    pred_image = postprocess(
        pred_ttb=(pred_tensor == 1).detach().cpu(),  # TTB label
        pt_mask=pt_mask.detach().cpu(),
        # Pass the original PT profile
        spacing=pt_image.GetSpacing(),
        origin=pt_image.GetOrigin(),
        direction=pt_image.GetDirection(),
        metadata={k: pt_image.GetMetaData(k) for k in pt_image.GetMetaDataKeys()},
    )
    log.debug("Postprocessing completed in %.2f seconds", time.monotonic() - tic)

    # Clear CUDA memory if using GPU
    torch.cuda.empty_cache()

    return pred_image


def preprocess(
    pt_image: sitk.Image,
    ct_image: sitk.Image,
    organs_segmentation_image: sitk.Image,
    suv_threshold: float,
    return_pt_mask: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    transform = T.Compose(
        [
            SITKResampleToMatchd(
                keys=[CT_KEY, ORGANS_KEY],
                key_dst=PT_KEY,
                default_value=[-1000, 0],
                mode=["nearest", "nearest"],
            ),
            SITKChangeLabeld(keys=[ORGANS_KEY], mapping=ORGANS_MAPPING),
            SITKToTensord(keys=[PT_KEY, CT_KEY, ORGANS_KEY]),
            T.EnsureChannelFirstd(keys=[PT_KEY, CT_KEY, ORGANS_KEY], channel_dim="no_channel"),
            T.ScaleIntensityRanged(keys=CT_KEY, a_min=-1000, a_max=600, b_min=0.0, b_max=1.0, clip=True),
            Divided(keys=[PT_KEY], value=suv_threshold),
            Thresholdd(keys=[PT_KEY], dst_keys=[PT_MASK_KEY], threshold=1.0, above=True),
            T.ToTensord(keys=[PT_KEY, CT_KEY], dtype=torch.float32),
            T.ToTensord(keys=[ORGANS_KEY], dtype=torch.int32),
            T.AsDiscreted(keys=[ORGANS_KEY], to_onehot=9),
            T.ConcatItemsd(keys=[PT_KEY, CT_KEY, PT_MASK_KEY, ORGANS_KEY], name=IMAGE_KEY),
        ]
    )

    data = {PT_KEY: pt_image, CT_KEY: ct_image, ORGANS_KEY: organs_segmentation_image}
    data = transform(data)

    if return_pt_mask:
        return data[IMAGE_KEY], data[PT_MASK_KEY]
    return data[IMAGE_KEY]


def postprocess(
    pred_ttb: Tensor,
    pt_mask: Tensor,
    *,
    spacing: Optional[Sequence[float]] = None,
    origin: Optional[Sequence[float]] = None,
    direction: Optional[Sequence[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> sitk.Image:
    transform = T.Compose(
        [
            T.ToTensord(keys=[PRED_TTB_KEY], dtype=torch.float32),
            T.ImageFilterd(keys=[PRED_TTB_KEY], kernel="elliptical", kernel_size=3),
            T.AsDiscreted(keys=[PRED_TTB_KEY], threshold=0.5, dtype=torch.int16),
            T.ToTensord(keys=[PRED_TTB_KEY, PT_MASK_KEY], dtype=torch.int16),
            LogicalAndd(keys=[PRED_TTB_KEY], other_keys=[PT_MASK_KEY]),
            T.SqueezeDimd(keys=[PRED_TTB_KEY], dim=0),  # Remove the channel dimension
            ToSITKd(keys=[PRED_TTB_KEY], spacing=spacing, origin=origin, direction=direction, metadata=metadata),
        ]
    )

    data = {PRED_TTB_KEY: pred_ttb, PT_MASK_KEY: pt_mask}
    data = transform(data)
    return data[PRED_TTB_KEY]


def pad_tensor(
    x: Tensor,
    pad_widths: Sequence[tuple[int, int]],
    mode: str = "constant",
    value: float | None = None,
) -> Tensor:
    padding = list(itertools.chain.from_iterable(reversed(pad_widths)))  # [::-1]
    return F.pad(x, padding, mode=mode, value=value)


def unpad_tensor(x: Tensor, pad_widths: Sequence[tuple[int, int]]) -> Tensor:
    slices = [slice(pad[0], -pad[1] if pad[1] > 0 else None) if pad is not None else slice(None) for pad in pad_widths]
    return x[tuple(slices)]


def divisible_pad_widths(sizes: Sequence[int], k: int = 32) -> list[tuple[int, int]]:
    def calculate_padding(size: int, k: int = 32) -> tuple[int, int]:
        if size % k == 0:
            return (0, 0)

        pad_left = (k - size % k) // 2
        pad_right = k - size % k - pad_left
        return (pad_left, pad_right)

    return [calculate_padding(s, k) for s in sizes]
