from pathlib import Path

import pytest
import yaml

from lafgs.protocol import (
    REQUIRED_OFFLINE_CHAIN,
    RESEARCH_ONLY_COMPONENTS,
    load_mainline_protocol,
)
from lafgs.visualization.paper import (
    _resize_mask,
    build_method_overview,
    select_qualitative_query,
)


ROOT = Path(__file__).resolve().parents[1]


def test_release_protocol_freezes_method_and_sparse_deployment():
    protocol = load_mainline_protocol(
        ROOT / "configs" / "lafgs_paper_mainline.yaml"
    )
    assert protocol.offline_chain == REQUIRED_OFFLINE_CHAIN
    assert RESEARCH_ONLY_COMPONENTS <= protocol.excluded_components
    assert protocol.manifest()["deployment"]["pose_solves"] == 1


def test_release_protocol_rejects_research_component_in_default(tmp_path):
    source = yaml.safe_load(
        (ROOT / "configs" / "lafgs_paper_mainline.yaml").read_text()
    )
    source["method"]["excluded_by_default"].remove("viewpoint_completion")
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(source))
    with pytest.raises(ValueError, match="research-only"):
        load_mainline_protocol(path)


def test_query_selection_prefers_weak_prior_and_joint_a1_improvement():
    def row(te, precision):
        return {
            "sparse_TE": te,
            "sparse": {"sparse_diag_all_gt_precision_2px": precision},
        }

    a0 = {"a.png": row(40.0, 0.01), "b.png": row(25.0, 0.04)}
    a1 = {"a.png": row(5.0, 0.10), "b.png": row(20.0, 0.05)}
    prior = {
        "a.png": {"psnr_db": 10.0},
        "b.png": {"psnr_db": 18.0},
    }
    selected = select_qualitative_query(a0, a1, prior)
    assert selected["image_name"] == "a.png"
    assert selected["te_gain_cm"] == 35.0


def test_method_overview_is_manifest_driven(tmp_path):
    protocol = load_mainline_protocol(
        ROOT / "configs" / "lafgs_paper_mainline.yaml"
    )
    output = tmp_path / "overview.png"
    manifest = build_method_overview(protocol, output)
    assert output.is_file()
    assert len(manifest["sha256"]) == 64
    assert "One-shot sparse localization" in manifest["labels"]


def test_release_visualizer_reads_serialized_tensor_masks():
    torch = pytest.importorskip("torch")
    mask = torch.tensor([[1, 0], [0, 1]], dtype=torch.bool)
    resized = _resize_mask(mask, (4, 4))
    assert resized.shape == (4, 4)
    assert resized.dtype == bool
    assert resized[:2, :2].all()
    assert not resized[:2, 2:].any()
