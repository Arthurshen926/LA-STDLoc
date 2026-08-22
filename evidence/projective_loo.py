"""Query-local descriptor and pure-ray geometry replay for V6 feedback."""

from __future__ import annotations

import torch

from evidence.observation_provider import ObservationProvider
from evidence.tracks import fuse_projective_anchor_observations
from evidence.triangulation import robust_triangulate_associations


class LeaveOneQueryOutProjectiveMap:
    """Recompute every affected Anchor after removing one mapping camera."""

    def __init__(
        self,
        state: dict,
        observations: ObservationProvider,
        *,
        minimum_views: int = 3,
        minimum_view_bins: int = 2,
        minimum_parallax_deg: float = 1.0,
        maximum_reprojection_px: float = 2.0,
        descriptor_trim_fraction: float = 0.2,
    ) -> None:
        names = list(state.get("v6_mapping_query_names", ()))
        if names != list(observations.names):
            raise ValueError("V6 map and observation registries differ")
        construction = state.get("projective_anchor_construction", {})
        if construction.get("final_xyz_source") != "fixed_camera_robust_ray_triangulation":
            raise ValueError("LOO replay requires the V6 pure-ray map")
        csr = state.get("projective_anchor_observations", {})
        offsets = torch.as_tensor(csr.get("observation_offsets")).long()
        query = torch.as_tensor(csr.get("query_indices")).long()
        keypoint = torch.as_tensor(csr.get("keypoint_indices")).long()
        count = int(torch.as_tensor(state["anchor_ids"]).numel())
        if offsets.shape != (count + 1,) or int(offsets[-1]) != query.numel():
            raise ValueError("V6 map observation CSR is invalid")
        if query.shape != keypoint.shape:
            raise ValueError("V6 map observation CSR columns differ")
        self.state = state
        self.observations = observations
        self.offsets = offsets
        self.query = query
        self.keypoint = keypoint
        self.query_bins = torch.as_tensor(state["v6_mapping_query_bins"]).long()
        self.minimum_views = int(minimum_views)
        self.minimum_view_bins = int(minimum_view_bins)
        self.minimum_parallax_deg = float(minimum_parallax_deg)
        self.maximum_reprojection_px = float(maximum_reprojection_px)
        self.descriptor_trim_fraction = float(descriptor_trim_fraction)
        self.views = [observations.build_view(index) for index in range(len(observations))]
        anchor_for_observation = torch.repeat_interleave(
            torch.arange(count, dtype=torch.long), offsets[1:] - offsets[:-1]
        )
        self.anchor_for_observation = anchor_for_observation

    @torch.no_grad()
    def query_update(self, excluded_query: int) -> dict:
        excluded_query = int(excluded_query)
        if excluded_query < 0 or excluded_query >= len(self.observations):
            raise IndexError(excluded_query)
        removed = self.query == excluded_query
        affected = torch.unique(self.anchor_for_observation[removed], sorted=True)
        if affected.numel() == 0:
            return {
                "anchor_rows": affected,
                "valid": torch.empty(0, dtype=torch.bool),
                "anchor_xyz": torch.empty((0, 3)),
                "anchor_features": torch.empty(
                    (0, int(torch.as_tensor(self.state["anchor_features"]).shape[1]))
                ),
            }
        lookup = torch.full(
            (int(torch.as_tensor(self.state["anchor_ids"]).numel()),),
            -1,
            dtype=torch.long,
        )
        lookup[affected] = torch.arange(affected.numel())
        keep = (~removed) & (lookup[self.anchor_for_observation] >= 0)
        query = self.query[keep]
        keypoint = self.keypoint[keep]
        local_anchor = lookup[self.anchor_for_observation[keep]]
        counts = torch.tensor(
            [view.keypoints.shape[0] for view in self.views], dtype=torch.long
        )
        packed_offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        packed_uv = torch.cat([view.physical_keypoints for view in self.views])
        uv = packed_uv[packed_offsets[query] + keypoint]
        detector = torch.cat(
            [view.detector_scores.float() for view in self.views]
        )[packed_offsets[query] + keypoint]
        geometry = robust_triangulate_associations(
            landmark_count=int(affected.numel()),
            landmark_index=local_anchor,
            query_index=query,
            uv=uv,
            confidence=detector.clamp_min(1e-6),
            camera_K=torch.stack([view.intrinsics.float() for view in self.views]),
            pose_w2c=torch.stack([view.pose_w2c.float() for view in self.views]),
            query_bin=self.query_bins,
            rendered_depth=None,
            maximum_observations_per_landmark=32,
            minimum_views=self.minimum_views,
            minimum_view_bins=self.minimum_view_bins,
            huber_delta_px=2.0,
            iterations=3,
            minimum_parallax_deg=self.minimum_parallax_deg,
            parallax_quantile=0.75,
            maximum_reprojection_px=self.maximum_reprojection_px,
            maximum_condition_number=1e6,
            maximum_covariance_trace_m2=float("inf"),
            maximum_rendered_depth_residual_m=float("inf"),
            minimum_rendered_depth_observations=0,
            surface_support_enabled=False,
        )
        valid = torch.as_tensor(geometry["triangulated"]).bool()
        output_features = torch.as_tensor(self.state["anchor_features"])[affected].clone()
        packed_descriptor = torch.cat([view.descriptors.float() for view in self.views])
        selected_descriptor = packed_descriptor[packed_offsets[query] + keypoint]
        for local in torch.nonzero(valid, as_tuple=False).reshape(-1).tolist():
            rows = torch.nonzero(local_anchor == local, as_tuple=False).reshape(-1)
            output_features[local] = fuse_projective_anchor_observations(
                selected_descriptor[rows],
                self.query_bins[query[rows]],
                detector_weight=detector[rows],
                trim_fraction=self.descriptor_trim_fraction,
            )
        return {
            "anchor_rows": affected,
            "valid": valid,
            "anchor_xyz": torch.as_tensor(geometry["triangulated_xyz"]).float(),
            "anchor_features": output_features.float(),
            "contract": {
                "query_descriptor_loo": True,
                "query_geometry_loo": True,
                "gaussian_depth_used_for_final_xyz": False,
            },
        }
