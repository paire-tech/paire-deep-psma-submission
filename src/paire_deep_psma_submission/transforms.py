import math
from typing import Any, Dict, Optional, Sequence, Union

import monai.transforms as T
import SimpleITK as sitk
import torch
from monai.config import KeysCollection
from monai.utils.misc import ensure_tuple_rep

SITK_INTERPOLATORS: Dict[str, int] = {
    "nearest": sitk.sitkNearestNeighbor,
    "linear": sitk.sitkLinear,
    "bspline": sitk.sitkBSpline,
    "bspline1": sitk.sitkBSpline1,
    "bspline2": sitk.sitkBSpline2,
    "bspline3": sitk.sitkBSpline3,
    "bspline4": sitk.sitkBSpline4,
    "bspline5": sitk.sitkBSpline5,
    "gaussian": sitk.sitkGaussian,
    "label_gaussian": sitk.sitkLabelGaussian,
    "label_linear": sitk.sitkLabelLinear,
    "hamming_windowed_sinc": sitk.sitkHammingWindowedSinc,
    "cosine_windowed_sinc": sitk.sitkCosineWindowedSinc,
    "welch_windowed_sinc": sitk.sitkWelchWindowedSinc,
    "lanczos_windowed_sinc": sitk.sitkLanczosWindowedSinc,
    "blackman_windowed_sinc": sitk.sitkBlackmanWindowedSinc,
    "bspline_resampler": sitk.sitkBSplineResampler,
    "bspline_resampler_order1": sitk.sitkBSplineResamplerOrder1,
    "bspline_resampler_order2": sitk.sitkBSplineResamplerOrder2,
    "bspline_resampler_order3": sitk.sitkBSplineResamplerOrder3,
    "bspline_resampler_order4": sitk.sitkBSplineResamplerOrder4,
    "bspline_resampler_order5": sitk.sitkBSplineResamplerOrder5,
}


class SITKToTensord(T.MapTransform):
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Dict) -> Dict:
        data = dict(data)
        for key in self.key_iterator(data):
            image = data[key]
            array = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
            data[key] = torch.from_numpy(array)

        return data


class SITKResampleSpacingd(T.MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        spacing: Union[float, Sequence[float]],
        default_value: Union[float, Sequence[float]] = 0,
        mode: Union[str, Sequence[str]] = "linear",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.spacing = ensure_tuple_rep(spacing, dim=3)
        self.default_value = ensure_tuple_rep(default_value, dim=len(self.keys))
        self.mode = ensure_tuple_rep(mode, dim=len(self.keys))

    def __call__(self, data: Dict) -> Dict:
        data = dict(data)
        for key, mode, default_value in self.key_iterator(data, self.mode, self.default_value):
            interpolator = SITK_INTERPOLATORS[mode]

            image = data[key]
            original_spacing = image.GetSpacing()
            original_size = image.GetSize()
            new_size = [math.floor(original_size[i] * original_spacing[i] / self.spacing[i]) for i in range(3)]

            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputSpacing(self.spacing)
            resampler.SetSize(tuple(new_size))
            resampler.SetOutputDirection(image.GetDirection())
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetDefaultPixelValue(default_value)
            resampler.SetInterpolator(interpolator)
            resampled_image = resampler.Execute(image)
            data[key] = resampled_image

        return data


class SITKResampleToMatchd(T.MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        key_dst: str,
        default_value: Union[float, Sequence[float]] = 0,
        mode: Union[str, Sequence[str]] = "linear",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.key_dst = key_dst
        self.default_value = ensure_tuple_rep(default_value, dim=len(self.keys))
        self.mode = ensure_tuple_rep(mode, dim=len(self.keys))

    def __call__(self, data: Dict) -> Dict:
        data = dict(data)
        image_dst = data[self.key_dst]
        for key, mode, default_value in self.key_iterator(data, self.mode, self.default_value):
            if key == self.key_dst:
                continue

            interpolator = SITK_INTERPOLATORS[mode]
            image_src = data[key]
            resampled_image = sitk.Resample(image_src, image_dst, sitk.Transform(), interpolator, default_value)
            data[key] = resampled_image
        return data


class SITKChangeLabeld(T.MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        mapping: Dict[int, int],
        dst_keys: Optional[KeysCollection] = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.mapping = mapping
        self.dst_keys = ensure_tuple_rep(dst_keys or self.keys, dim=len(self.keys))

    def __call__(self, data: Dict) -> Dict:
        data = dict(data)
        for key, dst_key in self.key_iterator(data, self.dst_keys):
            value = data[key]

            label_stats = sitk.LabelStatisticsImageFilter()
            label_stats.Execute(value, value)
            labels = sorted(label_stats.GetLabels())

            if not set(labels).issubset(self.mapping.keys()):
                missing_labels = [label for label in labels if label not in self.mapping]
                raise ValueError(
                    f"Not all labels are present in the mapping keys: {self.mapping.keys()}. "
                    "Please ensure that the mapping covers all labels in the data. "
                    f"Missing labels: {', '.join([f'{lbl}' for lbl in missing_labels])}."
                )

            mapping = {k: float(v) for k, v in self.mapping.items()}
            value = sitk.ChangeLabel(value, mapping)

            data[dst_key] = value
        return data


class SITKCastd(T.MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        dtype: Union[int, Sequence[int]],
        dst_keys: Optional[KeysCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.dtype = ensure_tuple_rep(dtype, dim=len(self.keys))
        self.dst_keys = ensure_tuple_rep(dst_keys or self.keys, dim=len(self.keys))

    def __call__(self, data: Dict) -> Dict:
        data = dict(data)
        for key, dst_key, dtype in self.key_iterator(data, self.dst_keys, self.dtype):
            data[dst_key] = sitk.Cast(data[key], dtype)
        return data


class ToSITKd(T.MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        spacing: Optional[Sequence[float]] = None,
        origin: Optional[Sequence[float]] = None,
        direction: Optional[Sequence[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        dst_keys: Optional[KeysCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.spacing = ensure_tuple_rep(spacing, dim=3) if spacing is not None else None
        self.origin = ensure_tuple_rep(origin, dim=3) if origin is not None else None
        self.direction = ensure_tuple_rep(direction, dim=9) if direction is not None else None
        self.metadata = metadata if metadata is not None else {}
        self.dst_keys = ensure_tuple_rep(dst_keys or self.keys, dim=len(self.keys))

    def __call__(self, data: Dict) -> Dict:
        data = dict(data)
        for key, dst_key in self.key_iterator(data, self.dst_keys):
            array = data[key].detach().cpu().numpy().transpose(2, 1, 0)
            image = sitk.GetImageFromArray(array)

            if self.spacing is not None:
                image.SetSpacing(self.spacing)
            if self.origin is not None:
                image.SetOrigin(self.origin)
            if self.direction is not None:
                image.SetDirection(self.direction)
            for k, v in self.metadata.items():
                image.SetMetaData(k, str(v))

            data[dst_key] = image

        return data


class Thresholdd(T.MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        threshold: Union[float, Sequence[float]],
        above: Union[bool, Sequence[bool]] = True,
        dst_keys: Optional[KeysCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.threshold = ensure_tuple_rep(threshold, dim=len(self.keys))
        self.above = ensure_tuple_rep(above, dim=len(self.keys))
        self.dst_keys = ensure_tuple_rep(dst_keys or self.keys, dim=len(self.keys))

    def __call__(self, data: Dict) -> Dict:
        data = dict(data)
        for key, dst_key, threshold, above in self.key_iterator(data, self.dst_keys, self.threshold, self.above):
            data[dst_key] = (data[key] >= threshold if above else data[key] < threshold) * 1
        return data


class Divided(T.MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        value: Union[int, float, Sequence[Union[int, float]]],
        dst_keys: Optional[KeysCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.value = ensure_tuple_rep(value, dim=len(self.keys))
        self.dst_keys = ensure_tuple_rep(dst_keys or self.keys, dim=len(self.keys))

    def __call__(self, data: Dict) -> Dict:
        data = dict(data)
        for key, dst_key, value in self.key_iterator(data, self.dst_keys, self.value):
            data[dst_key] = data[key] / value
        return data


class LogicalAndd(T.MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        other_keys: KeysCollection,
        dst_keys: Optional[KeysCollection] = None,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.other_keys = ensure_tuple_rep(other_keys, dim=len(self.keys))
        self.dst_keys = ensure_tuple_rep(dst_keys or self.keys, dim=len(self.keys))

    def __call__(self, data: Dict) -> Dict:
        data = dict(data)
        for key, dst_key, other_key in self.key_iterator(data, self.dst_keys, self.other_keys):
            if other_key not in data:
                available_keys = list(data.keys())
                raise KeyError(
                    f"Key {other_key!r} not found in data dictionary for transform {self.__class__.__name__}."
                    f" Available keys: {', '.join(available_keys)}."
                )

            data[dst_key] = data[key] & data[other_key]
        return data
