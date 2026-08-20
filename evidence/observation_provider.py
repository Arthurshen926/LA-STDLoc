"""Shared observation interface for real and Gaussian-rendered mapping views.

The provider deliberately owns no feature extraction policy.  It is a strict,
read-only adapter over an already materialized observation cache so that the
Track, triangulation, fusion, registry, and selection stages consume the same
data model independent of the RGB source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from features.multiview_fusion import PIXEL_CENTER_OFFSET


SourceKind = Literal["real_rgb", "gaussian_render"]


def _tensor(
    value,
    *,
    name: str,
    ndim: int | None = None,
    shape_tail: tuple[int, ...] | None = None,
) -> torch.Tensor:
    result = torch.as_tensor(value)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape={tuple(result.shape)}")
    if shape_tail is not None and tuple(result.shape[-len(shape_tail) :]) != shape_tail:
        raise ValueError(
            f"{name} must end in shape {shape_tail}, got shape={tuple(result.shape)}"
        )
    return result


@dataclass(frozen=True)
class ObservationView:
    """Validated observations for one frozen mapping camera."""

    camera_index: int
    image_name: str
    pose_w2c: torch.Tensor
    intrinsics: torch.Tensor
    keypoints: torch.Tensor
    descriptors: torch.Tensor
    detector_scores: torch.Tensor
    valid_mask: torch.Tensor | None
    alpha: torch.Tensor | None
    depth: torch.Tensor | None
    surface_support: torch.Tensor | None
    keypoint_validity: torch.Tensor | None
    keypoint_alpha: torch.Tensor | None
    keypoint_depth: torch.Tensor | None
    sequence_id: str | None
    pose_bin: int
    source_kind: SourceKind
    image_hw: tuple[int, int]
    pixel_center_offset: float = PIXEL_CENTER_OFFSET

    @property
    def physical_keypoints(self) -> torch.Tensor:
        """Keypoints in the pixel-center convention used by geometry/PnP."""

        return self.keypoints.float() + float(self.pixel_center_offset)


class ObservationProvider:
    """Strict cache-backed observation provider shared by Real and Render."""

    source_kind: SourceKind

    def __init__(
        self,
        payload: dict,
        *,
        source_kind: SourceKind,
        query_names: list[str] | tuple[str, ...] | None = None,
        query_bins: torch.Tensor | list[int] | None = None,
        validate_all: bool = True,
    ) -> None:
        if not isinstance(payload, dict):
            raise TypeError("observation cache payload must be a dictionary")
        records = payload.get("queries", payload)
        if not isinstance(records, dict) or not records:
            raise ValueError("observation cache must contain a non-empty queries map")
        names = (
            list(records)
            if query_names is None
            else [str(name) for name in query_names]
        )
        if len(names) != len(set(names)):
            raise ValueError("observation image names must be unique")
        missing = [name for name in names if name not in records]
        extra = [name for name in records if name not in set(names)]
        if missing or extra:
            raise ValueError(
                "observation cache order does not exactly cover records: "
                f"missing={missing[:3]} extra={extra[:3]}"
            )
        if query_bins is None:
            query_bins = payload.get("query_bins")
        if query_bins is not None:
            bins = _tensor(query_bins, name="query_bins", ndim=1).long()
            if bins.numel() != len(names):
                raise ValueError("query_bins must have one value per observation view")
        else:
            bins = None
        self._payload = payload
        self._records = records
        self._names = tuple(names)
        self._name_to_index = {name: index for index, name in enumerate(self._names)}
        self._query_bins = bins
        self.source_kind = source_kind
        self._validate_source_contract()
        if validate_all:
            for index in range(len(self)):
                self.build_view(index)

    @property
    def payload(self) -> dict:
        return self._payload

    @property
    def records(self) -> dict:
        return self._records

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    def __len__(self) -> int:
        return len(self._names)

    def _validate_source_contract(self) -> None:
        declared = self._payload.get("uses_source_mapping_rgb")
        if self.source_kind == "gaussian_render" and declared is not False:
            raise ValueError(
                "GaussianRenderObservationProvider requires "
                "uses_source_mapping_rgb=false"
            )
        if self.source_kind == "real_rgb" and declared is False:
            raise ValueError(
                "RealRGBObservationProvider refuses a source-image-free cache"
            )

    def _record(self, index_or_name: int | str) -> tuple[int, str, dict]:
        if isinstance(index_or_name, str):
            if index_or_name not in self._name_to_index:
                raise KeyError(index_or_name)
            index = self._name_to_index[index_or_name]
        else:
            index = int(index_or_name)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        name = self._names[index]
        record = self._records[name]
        if not isinstance(record, dict):
            raise TypeError(f"observation record {name!r} must be a dictionary")
        return index, name, record

    def build_view(self, index_or_name: int | str) -> ObservationView:
        index, name, record = self._record(index_or_name)
        keypoints = _tensor(
            record.get("native_keypoints"),
            name=f"{name}.native_keypoints",
            ndim=2,
            shape_tail=(2,),
        )
        descriptors = _tensor(
            record.get("native_descriptors"),
            name=f"{name}.native_descriptors",
            ndim=2,
        )
        scores = _tensor(
            record.get("native_scores"),
            name=f"{name}.native_scores",
            ndim=1,
        )
        row_count = int(keypoints.shape[0])
        if int(descriptors.shape[0]) != row_count or int(scores.shape[0]) != row_count:
            raise ValueError(f"{name} sparse observation rows are not aligned")
        if not keypoints.is_floating_point() or not descriptors.is_floating_point():
            raise ValueError(f"{name} keypoints/descriptors must be floating point")
        if not scores.is_floating_point():
            raise ValueError(f"{name} detector scores must be floating point")
        intrinsics = _tensor(record.get("native_K"), name=f"{name}.native_K", ndim=2)
        if tuple(intrinsics.shape) != (3, 3):
            raise ValueError(f"{name}.native_K must have shape [3,3]")
        pose = _tensor(record.get("pose_w2c"), name=f"{name}.pose_w2c", ndim=2)
        if tuple(pose.shape) != (4, 4):
            raise ValueError(f"{name}.pose_w2c must have shape [4,4]")
        image_hw_value = record.get("native_input_hw")
        if not isinstance(image_hw_value, (list, tuple)) or len(image_hw_value) != 2:
            if not isinstance(image_hw_value, torch.Tensor) or image_hw_value.shape != (
                2,
            ):
                raise ValueError(f"{name}.native_input_hw must contain [height,width]")
        image_hw_tensor = torch.as_tensor(image_hw_value)
        if image_hw_tensor.ndim != 1 or image_hw_tensor.shape[0] != 2:
            raise ValueError(f"{name}.native_input_hw must have exact shape [2]")
        height, width = (int(value) for value in image_hw_tensor.tolist())
        if height <= 0 or width <= 0:
            raise ValueError(f"{name}.native_input_hw must be positive")

        valid_mask = record.get("native_valid_mask")
        if valid_mask is not None:
            valid_mask = _tensor(valid_mask, name=f"{name}.native_valid_mask", ndim=2)
            if tuple(valid_mask.shape) != (height, width):
                raise ValueError(f"{name}.native_valid_mask does not match image size")
        depth = record.get("native_depth", record.get("native_rendered_depth"))
        if depth is not None:
            depth = _tensor(depth, name=f"{name}.native_depth", ndim=2)
            if tuple(depth.shape) != (height, width):
                raise ValueError(f"{name}.native_depth does not match image size")
        alpha = record.get("native_alpha", record.get("native_rendered_alpha"))
        if alpha is not None:
            alpha = _tensor(alpha, name=f"{name}.native_alpha", ndim=2)
            if tuple(alpha.shape) != (height, width):
                raise ValueError(f"{name}.native_alpha does not match image size")
        surface_support = record.get("native_surface_support")
        if surface_support is None:
            surface_support = record.get("native_appearance_reliability")
        if surface_support is not None:
            surface_support = _tensor(
                surface_support,
                name=f"{name}.native_surface_support",
                ndim=1,
            )
            if int(surface_support.shape[0]) != row_count:
                raise ValueError(f"{name}.native_surface_support rows are not aligned")
        keypoint_validity = record.get("native_valid_keypoint_mask")
        if keypoint_validity is not None:
            keypoint_validity = _tensor(
                keypoint_validity,
                name=f"{name}.native_valid_keypoint_mask",
                ndim=1,
            ).bool()
            if int(keypoint_validity.shape[0]) != row_count:
                raise ValueError(
                    f"{name}.native_valid_keypoint_mask rows are not aligned"
                )
        keypoint_alpha = record.get("native_alpha_at_keypoints")
        if keypoint_alpha is not None:
            keypoint_alpha = _tensor(
                keypoint_alpha,
                name=f"{name}.native_alpha_at_keypoints",
                ndim=1,
            ).float()
            if int(keypoint_alpha.shape[0]) != row_count:
                raise ValueError(
                    f"{name}.native_alpha_at_keypoints rows are not aligned"
                )
        keypoint_depth = record.get("native_depth_at_keypoints")
        if keypoint_depth is not None:
            keypoint_depth = _tensor(
                keypoint_depth,
                name=f"{name}.native_depth_at_keypoints",
                ndim=1,
            ).float()
            if int(keypoint_depth.shape[0]) != row_count:
                raise ValueError(
                    f"{name}.native_depth_at_keypoints rows are not aligned"
                )
        raw_sequence_id = record.get("sequence_id")
        sequence_id = None if raw_sequence_id is None else str(raw_sequence_id)
        pose_bin = (
            int(self._query_bins[index])
            if self._query_bins is not None
            else int(record.get("pose_bin", -1))
        )
        pixel_center_offset = float(
            record.get("pixel_center_offset", PIXEL_CENTER_OFFSET)
        )
        if pixel_center_offset != float(PIXEL_CENTER_OFFSET):
            raise ValueError(
                f"{name} uses unsupported pixel-center offset {pixel_center_offset}"
            )
        return ObservationView(
            camera_index=index,
            image_name=name,
            pose_w2c=pose,
            intrinsics=intrinsics,
            keypoints=keypoints,
            descriptors=descriptors,
            detector_scores=scores,
            valid_mask=valid_mask,
            alpha=alpha,
            depth=depth,
            surface_support=surface_support,
            keypoint_validity=keypoint_validity,
            keypoint_alpha=keypoint_alpha,
            keypoint_depth=keypoint_depth,
            sequence_id=sequence_id,
            pose_bin=pose_bin,
            source_kind=self.source_kind,
            image_hw=(height, width),
            pixel_center_offset=pixel_center_offset,
        )

    def track_inputs(self) -> dict:
        """Return the canonical inputs consumed by Track construction."""

        views = [self.build_view(index) for index in range(len(self))]
        return {
            "query_names": list(self._names),
            "descriptors": [view.descriptors.float() for view in views],
            "keypoints": [view.physical_keypoints for view in views],
            "detector_scores": [view.detector_scores.float() for view in views],
            "camera_K": torch.stack([view.intrinsics.float() for view in views]),
            "pose_w2c": torch.stack([view.pose_w2c.float() for view in views]),
            "image_hw": torch.tensor(
                [view.image_hw for view in views], dtype=torch.long
            ),
            "query_groups": [view.sequence_id for view in views],
        }


class RealRGBObservationProvider(ObservationProvider):
    def __init__(self, payload: dict, **kwargs) -> None:
        super().__init__(payload, source_kind="real_rgb", **kwargs)


class GaussianRenderObservationProvider(ObservationProvider):
    def __init__(self, payload: dict, **kwargs) -> None:
        super().__init__(payload, source_kind="gaussian_render", **kwargs)
