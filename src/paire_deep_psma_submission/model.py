import logging
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
from monai.networks.nets import DynUNet

log = logging.getLogger(__name__)


def load_model(weights_dir: Union[str, Path], device: str = "cpu") -> nn.Module:
    model = DynUNet(
        spatial_dims=3,
        in_channels=12,
        out_channels=3,
        kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 2]],
        upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 2]],
        dropout=None,
        norm_name=("instance", {"affine": True}),
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supervision=False,
        res_block=True,
        trans_bias=False,
    )

    weights_paths = list(Path(weights_dir).glob("*.pth"))
    if len(weights_paths) != 1:
        raise ValueError(f"Expected exactly one weights file in {weights_dir}, found {len(weights_paths)}.")

    weights_path = weights_paths[0]
    log.debug("Loading model weights from %s", weights_path)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    log.info("Successfully loaded model from %s on '%s' device", weights_path, device)

    return model
