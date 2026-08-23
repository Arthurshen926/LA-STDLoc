"""Query-local descriptor and pure-ray geometry replay for V6 feedback."""

from __future__ import annotations

from typing import Literal

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
        affected_anchor_policy: Literal["rebuild", "purge"] = "rebuild",
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
        if affected_anchor_policy not in {"rebuild", "purge"}:
            raise ValueError("affected_anchor_policy must be 'rebuild' or 'purge'")
        self.affected_anchor_policy = affected_anchor_policy
        if (
            self.affected_anchor_policy == "rebuild"
            and (
                state.get("provenance", {}).get("v6_compact_deployment_export")
                is True
                or (
                    isinstance(state.get("v6_descriptor_distillation"), dict)
                    and state.get("anchor_descriptor_residual") is None
                )
            )
        ):
            raise ValueError(
                "affected-Anchor rebuild requires the descriptor training checkpoint"
            )
        anchor_for_observation = torch.repeat_interleave(
            torch.arange(count, dtype=torch.long), offsets[1:] - offsets[:-1]
        )
        self.anchor_for_observation = anchor_for_observation
        if query.numel() and (
            int(query.min()) < 0 or int(query.max()) >= len(observations)
        ):
            raise ValueError("V6 observation query index is out of range")
        query_counts = torch.bincount(query, minlength=len(observations))
        self.query_observation_offsets = torch.cat(
            (query_counts.new_zeros(1), query_counts.cumsum(0))
        )
        self.query_observation_order = torch.argsort(query, stable=True)

        # Purged LOO is the scalable, conservative cross-validation policy:
        # an Anchor touched by any held-out camera is made unavailable.  It
        # cannot leak held-out descriptor or geometry evidence, and none of
        # the expensive per-query retriangulation tables are needed.
        if self.affected_anchor_policy == "purge":
            self.views = []
            return
        self.views = [observations.build_view(index) for index in range(len(observations))]
        view_counts = torch.tensor(
            [view.keypoints.shape[0] for view in self.views], dtype=torch.long
        )
        self.packed_offsets = torch.cat(
            (view_counts.new_zeros(1), view_counts.cumsum(0))
        )
        self.packed_uv = torch.cat(
            [view.physical_keypoints for view in self.views]
        )
        self.packed_detector = torch.cat(
            [view.detector_scores.float() for view in self.views]
        )
        self.packed_descriptor = torch.cat(
            [view.descriptors.float() for view in self.views]
        )
        self.camera_K = torch.stack(
            [view.intrinsics.float() for view in self.views]
        )
        self.pose_w2c = torch.stack(
            [view.pose_w2c.float() for view in self.views]
        )

    @torch.no_grad()
    def query_update(
        self,
        excluded_query: int,
        *,
        excluded_queries: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    ) -> dict:
        """Rebuild affected Anchors after removing a pose-neighborhood.

        ``excluded_query`` remains explicit because it is the query being
        localized.  ``excluded_queries`` may additionally contain nearby
        mapping cameras.  Passing no neighborhood preserves the historical
        single-image LOO behavior.
        """

        excluded_query = int(excluded_query)
        if excluded_query < 0 or excluded_query >= len(self.observations):
            raise IndexError(excluded_query)
        excluded = torch.as_tensor(
            [excluded_query] if excluded_queries is None else excluded_queries,
            dtype=torch.long,
        ).reshape(-1)
        excluded = torch.unique(
            torch.cat((excluded, torch.tensor([excluded_query], dtype=torch.long))),
            sorted=True,
        )
        if excluded.numel() == 0 or int(excluded.min()) < 0 or int(excluded.max()) >= len(
            self.observations
        ):
            raise IndexError("excluded query neighborhood is out of range")
        removed_parts = []
        for query_index in excluded.tolist():
            query_start = int(self.query_observation_offsets[query_index])
            query_stop = int(self.query_observation_offsets[query_index + 1])
            removed_parts.append(
                self.query_observation_order[query_start:query_stop]
            )
        removed_positions = torch.cat(removed_parts)
        affected = torch.unique(
            self.anchor_for_observation[removed_positions], sorted=True
        )
        if affected.numel() == 0:
            return {
                "anchor_rows": affected,
                "valid": torch.empty(0, dtype=torch.bool),
                "anchor_xyz": torch.empty((0, 3)),
                "anchor_features": torch.empty(
                    (0, int(torch.as_tensor(self.state["anchor_features"]).shape[1]))
                ),
                "excluded_queries": excluded,
                "contract": {
                    "query_descriptor_loo": True,
                    "query_geometry_loo": True,
                    "pose_neighborhood_loo": int(excluded.numel()) > 1,
                    "affected_anchor_policy": self.affected_anchor_policy,
                    "affected_anchors_rebuilt": False,
                    "gaussian_depth_used_for_final_xyz": False,
                },
            }
        if self.affected_anchor_policy == "purge":
            return {
                "anchor_rows": affected,
                "valid": torch.zeros(affected.numel(), dtype=torch.bool),
                "anchor_xyz": torch.as_tensor(self.state["anchor_xyz"])[
                    affected
                ].float(),
                "anchor_features": torch.as_tensor(
                    self.state["anchor_features"]
                )[affected].float(),
                "excluded_queries": excluded,
                "contract": {
                    "query_descriptor_loo": True,
                    "query_geometry_loo": True,
                    "pose_neighborhood_loo": int(excluded.numel()) > 1,
                    "affected_anchor_policy": "purge",
                    "affected_anchors_rebuilt": False,
                    "gaussian_depth_used_for_final_xyz": False,
                },
            }
        lengths = self.offsets[affected + 1] - self.offsets[affected]
        local_anchor = torch.repeat_interleave(
            torch.arange(affected.numel(), dtype=torch.long), lengths
        )
        group_offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
        starts = torch.repeat_interleave(self.offsets[affected], lengths)
        within = torch.arange(int(lengths.sum()), dtype=torch.long) - (
            torch.repeat_interleave(group_offsets[:-1], lengths)
        )
        positions = starts + within
        keep = ~torch.isin(self.query[positions], excluded)
        positions = positions[keep]
        local_anchor = local_anchor[keep]
        query = self.query[positions]
        keypoint = self.keypoint[positions]
        packed_rows = self.packed_offsets[query] + keypoint
        uv = self.packed_uv[packed_rows]
        detector = self.packed_detector[packed_rows]
        geometry = robust_triangulate_associations(
            landmark_count=int(affected.numel()),
            landmark_index=local_anchor,
            query_index=query,
            uv=uv,
            confidence=detector.clamp_min(1e-6),
            camera_K=self.camera_K,
            pose_w2c=self.pose_w2c,
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
        selected_descriptor = self.packed_descriptor[packed_rows]
        selected_counts = torch.bincount(
            local_anchor, minlength=int(affected.numel())
        )
        selected_offsets = torch.cat(
            (selected_counts.new_zeros(1), selected_counts.cumsum(0))
        )
        for local in torch.nonzero(valid, as_tuple=False).reshape(-1).tolist():
            rows = torch.arange(
                int(selected_offsets[local]),
                int(selected_offsets[local + 1]),
                dtype=torch.long,
            )
            fused = fuse_projective_anchor_observations(
                selected_descriptor[rows],
                self.query_bins[query[rows]],
                detector_weight=detector[rows],
                trim_fraction=self.descriptor_trim_fraction,
            )
            residuals = self.state.get("anchor_descriptor_residual")
            if residuals is not None:
                residual = torch.as_tensor(residuals)[affected[local]].float()
                residual = residual - torch.dot(residual, fused) * fused
                fused = torch.nn.functional.normalize(fused + residual, dim=0)
            output_features[local] = fused
        return {
            "anchor_rows": affected,
            "valid": valid,
            "anchor_xyz": torch.as_tensor(geometry["triangulated_xyz"]).float(),
            "anchor_features": output_features.float(),
            "excluded_queries": excluded,
            "contract": {
                "query_descriptor_loo": True,
                "query_geometry_loo": True,
                "pose_neighborhood_loo": int(excluded.numel()) > 1,
                "affected_anchor_policy": "rebuild",
                "affected_anchors_rebuilt": True,
                "gaussian_depth_used_for_final_xyz": False,
            },
        }
