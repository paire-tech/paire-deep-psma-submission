import logging
from pathlib import Path
from typing import List, Union

import torch
from monai.networks.nets import DynUNet

log = logging.getLogger(__name__)


def load_model(weights_dir: Union[str, Path], prefix: str, device: str = "cpu") -> List[DynUNet]:
    model = DynUNet(
        spatial_dims=3,
        in_channels=12,
        out_channels=2,
        kernel_size=[[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 2]],
        upsample_kernel_size=[[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 2]],
        dropout=None,
        norm_name=("instance", {"affine": True}),
        act_name=("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        deep_supervision=False,
        res_block=True,
        trans_bias=False,
    ).to(device)

    weights_paths = list(Path(weights_dir).glob(f"{prefix}*.pth"))
    list_models = []
    for weights_path in weights_paths:
        log.info("Loading model weights from '%s'", weights_path)
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=True)
        log.info("Successfully loaded model from '%s' on '%s' device", weights_path, device)
        list_models.append(model)
    return list_models
