"""One immutable release operating point; explicit scene calibration is required."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ReleaseConfig:
    version: str = "compact-fast-v1"
    retention_fraction: float = 0.65
    minimum_matches: int = 256
    confidence: float = 0.999
    minimum_iterations: int = 250
    maximum_iterations: int = 30000
    nms_radius: int = 4
    descriptor_dimension: int = 256
    match_chunk_size: int = 8192
    pose_solves: int = 1
    cpu_threads: int = 4

    def resolved(self, *, keypoint_count, reprojection_error_px, mask_policy):
        if (
            not 1 <= keypoint_count <= 2048
            or not math.isfinite(reprojection_error_px)
            or reprojection_error_px <= 0
        ):
            raise ValueError("invalid resolved scene parameters")
        if mask_policy not in ["none", "explicit_valid_mask", "geometric_validity"]:
            raise ValueError("mask policy must be explicit")
        return dict(
            asdict(self),
            keypoint_count=keypoint_count,
            reprojection_error_px=reprojection_error_px,
            mask_policy=mask_policy,
        )


FROZEN = ReleaseConfig()
