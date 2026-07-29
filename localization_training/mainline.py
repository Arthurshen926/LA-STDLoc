"""Reusable facade for one alternating LaFGS localization-reconstruction round."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from localization_training.appearance_calibration import (
    calibrate_appearance_modes,
)
from localization_training.appearance_family import (
    build_appearance_mode_pool,
    materialize_family_artifact,
)
from localization_training.candidate_basin_teacher import (
    CandidateBasinConfig,
    build_candidate_basin_teacher,
)
from localization_training.confusion_evidence import (
    ConfusionGraphConfig,
    ConfusionViewPlanningConfig,
    ContrastiveEvidenceConfig,
    build_anchor_family_confusion_graph,
    build_contrastive_synthetic_record,
    pack_contrastive_synthetic_evidence,
    plan_confusion_conditioned_views,
)
from localization_training.failure_atlas import (
    FailureAtlasConfig,
    build_failure_atlas,
)
from localization_training.prototype_optimization import (
    ContrastivePrototypeOptimizationConfig,
    PrototypeOptimizationConfig,
    optimize_basin_prototypes,
    optimize_contrastive_prototypes,
)
from localization_training.shared_metric import SharedLowRankMetric


@dataclass
class LocalizationRoundState:
    anchor_map: dict
    metric: SharedLowRankMetric
    query_cache: dict
    complete_positive_teacher: dict
    dynamic_outcomes: dict
    query_bins: dict[str, int]

    def validate(self) -> None:
        anchor_count = int(
            torch.as_tensor(self.anchor_map["anchor_xyz"]).shape[0]
        )
        names = list(self.dynamic_outcomes["query_names"])
        if names != list(self.complete_positive_teacher["query_names"]):
            raise ValueError("mainline query registries differ")
        if int(self.dynamic_outcomes["anchor_count"]) != anchor_count:
            raise ValueError("dynamic outcomes do not align with mainline map")
        if (
            int(self.complete_positive_teacher["anchor_count"])
            != anchor_count
        ):
            raise ValueError("positive teacher does not align with mainline map")
        missing_cache = [name for name in names if name not in self.query_cache]
        if missing_cache:
            raise ValueError(
                f"mainline query cache misses {missing_cache[:3]}"
            )
        missing_bins = [name for name in names if name not in self.query_bins]
        if missing_bins:
            raise ValueError(
                f"mainline view-bin registry misses {missing_bins[:3]}"
            )


class AlternatingLocalizationReconstructor:
    """Compose method modules without filesystem or CLI dependencies."""

    def __init__(
        self, state: LocalizationRoundState, *, device: torch.device
    ):
        state.validate()
        self.state = state
        self.device = torch.device(device)

    def discover_appearance_families(
        self,
        *,
        basin_teacher: dict | None,
        synthetic_evidence: dict | None = None,
        **kwargs,
    ) -> dict:
        candidates, observations = build_appearance_mode_pool(
            self.state.anchor_map,
            self.state.complete_positive_teacher,
            self.state.query_cache,
            self.state.query_bins,
            self.state.metric,
            dynamic=self.state.dynamic_outcomes,
            basin_teacher=basin_teacher,
            synthetic_evidence=synthetic_evidence,
            device=self.device,
            **kwargs,
        )
        return materialize_family_artifact(
            self.state.anchor_map,
            candidates,
            config=kwargs,
            observation_payload=observations,
        )

    def calibrate_families(
        self,
        *,
        pool: dict,
        biases: torch.Tensor,
        base_family: dict | None = None,
        **kwargs,
    ) -> dict:
        return calibrate_appearance_modes(
            pool=pool,
            positives=self.state.complete_positive_teacher,
            dynamic=self.state.dynamic_outcomes,
            cache=self.state.query_cache,
            metric=self.state.metric,
            biases=biases,
            base_family=base_family,
            device=self.device,
            **kwargs,
        )

    def build_basin_teacher(
        self,
        *,
        family: dict,
        config: CandidateBasinConfig,
        progress=None,
    ) -> dict:
        return build_candidate_basin_teacher(
            state=self.state.anchor_map,
            metric=self.state.metric,
            family=family,
            dynamic=self.state.dynamic_outcomes,
            positives=self.state.complete_positive_teacher,
            cache=self.state.query_cache,
            config=config,
            device=self.device,
            progress=progress,
        )

    def optimize_prototypes(
        self,
        *,
        family: dict,
        basin_teacher: dict,
        config: PrototypeOptimizationConfig,
        **kwargs,
    ) -> tuple[dict, list[dict]]:
        return optimize_basin_prototypes(
            state=self.state.anchor_map,
            metric=self.state.metric,
            family=family,
            teacher=basin_teacher,
            cache=self.state.query_cache,
            config=config,
            device=self.device,
            **kwargs,
        )

    def build_failure_atlas(
        self,
        *,
        family: dict,
        basin_teacher: dict | None,
        config: FailureAtlasConfig,
        progress=None,
    ) -> dict:
        return build_failure_atlas(
            state=self.state.anchor_map,
            metric=self.state.metric,
            family=family,
            dynamic=self.state.dynamic_outcomes,
            positives=self.state.complete_positive_teacher,
            cache=self.state.query_cache,
            query_bins=self.state.query_bins,
            basin_teacher=basin_teacher,
            config=config,
            device=self.device,
            progress=progress,
        )

    def active_evidence_step(self) -> "ActiveEvidenceAcquisitionStep":
        return ActiveEvidenceAcquisitionStep(self)


class ActiveEvidenceAcquisitionStep:
    """First-class active evidence operation inside one reconstruction round."""

    def __init__(self, reconstructor: AlternatingLocalizationReconstructor):
        self.reconstructor = reconstructor

    @property
    def state(self) -> LocalizationRoundState:
        return self.reconstructor.state

    @property
    def device(self) -> torch.device:
        return self.reconstructor.device

    def build_confusion_atlas(
        self,
        *,
        family: dict,
        config: ConfusionGraphConfig,
        progress=None,
    ) -> dict:
        return build_anchor_family_confusion_graph(
            state=self.state.anchor_map,
            metric=self.state.metric,
            family=family,
            dynamic=self.state.dynamic_outcomes,
            positives=self.state.complete_positive_teacher,
            cache=self.state.query_cache,
            query_bins=self.state.query_bins,
            config=config,
            device=self.device,
            progress=progress,
        )

    def build_contrastive_evidence(
        self,
        *,
        family: dict,
        confusion_graph: dict,
        synthetic_records: list[dict],
        render_payloads: dict[str, dict],
        visibility_config,
        config: ContrastiveEvidenceConfig,
        source: dict,
    ) -> dict:
        records = []
        for record in synthetic_records:
            name = str(record["query_name"])
            payload = render_payloads[name]
            records.append(
                build_contrastive_synthetic_record(
                    record=record,
                    state=self.state.anchor_map,
                    metric=self.state.metric,
                    family=family,
                    confusion_graph=confusion_graph,
                    rendered_depth=payload["depth"],
                    alpha=payload["alpha"],
                    visibility_config=visibility_config,
                    config=config,
                    device=self.device,
                )
            )
        return pack_contrastive_synthetic_evidence(
            records, source=source, confusion_graph=confusion_graph
        )

    def propose_views(
        self,
        *,
        confusion_graph: dict,
        config: ConfusionViewPlanningConfig,
    ) -> list[dict]:
        return plan_confusion_conditioned_views(
            confusion_graph=confusion_graph,
            state=self.state.anchor_map,
            cache=self.state.query_cache,
            query_bins=self.state.query_bins,
            config=config,
        )

    def update_local_families(
        self,
        *,
        family: dict,
        contrastive_evidence: dict,
        config: ContrastivePrototypeOptimizationConfig,
        **kwargs,
    ) -> tuple[dict, list[dict]]:
        return optimize_contrastive_prototypes(
            state=self.state.anchor_map,
            metric=self.state.metric,
            family=family,
            evidence=contrastive_evidence,
            config=config,
            device=self.device,
            **kwargs,
        )

    def run_active_evidence_round(
        self,
        *,
        family: dict,
        synthetic_records: list[dict],
        render_payloads: dict[str, dict],
        visibility_config,
        confusion_config: ConfusionGraphConfig,
        evidence_config: ContrastiveEvidenceConfig,
        optimization_config: ContrastivePrototypeOptimizationConfig,
        source: dict,
        progress=None,
    ) -> dict:
        confusion_graph = self.build_confusion_atlas(
            family=family,
            config=confusion_config,
            progress=progress,
        )
        evidence = self.build_contrastive_evidence(
            family=family,
            confusion_graph=confusion_graph,
            synthetic_records=synthetic_records,
            render_payloads=render_payloads,
            visibility_config=visibility_config,
            config=evidence_config,
            source=source,
        )
        updated, history = self.update_local_families(
            family=family,
            contrastive_evidence=evidence,
            config=optimization_config,
            progress=progress,
        )
        return {
            "schema": "lafgs_active_evidence_round",
            "version": 1,
            "confusion_graph": confusion_graph,
            "contrastive_evidence": evidence,
            "updated_family": updated,
            "optimization_history": history,
        }
