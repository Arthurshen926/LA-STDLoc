import pytest
import torch

from common.artifact_lineage import audit_compact_artifact_lineage


def _artifacts(tmp_path):
    paths = {
        name: tmp_path / f"{name}.pt"
        for name in ("anchor_map", "function_graph", "teacher", "metric")
    }
    rows = torch.tensor([1, 3, 5])
    torch.save({"anchor_ids": torch.tensor([11, 17])}, paths["anchor_map"])
    torch.save(
        {
            "anchor_count": 2,
            "query_names": ["q"],
            "records": [{"query_rows": rows}],
        },
        paths["function_graph"],
    )
    torch.save(
        {
            "anchor_count": 2,
            "query_names": ["q"],
            "records": [{"query_rows": rows.clone()}],
        },
        paths["teacher"],
    )
    torch.save(
        {"landmark_indices": torch.tensor([11, 17]), "initial_metric_state": None},
        paths["metric"],
    )
    return paths


def _audit(paths):
    return audit_compact_artifact_lineage(
        anchor_map=paths["anchor_map"],
        function_graph=paths["function_graph"],
        complete_positive_teacher=paths["teacher"],
        metric_state=paths["metric"],
    )


def test_compact_artifact_lineage_requires_exact_ids_and_query_rows(tmp_path):
    paths = _artifacts(tmp_path)
    report = _audit(paths)
    assert report["valid"]
    assert report["anchor_count"] == 2
    assert report["query_count"] == 1

    metric = torch.load(paths["metric"], weights_only=False)
    metric["landmark_indices"] = torch.tensor([17, 11])
    torch.save(metric, paths["metric"])
    with pytest.raises(ValueError, match="metric landmark IDs"):
        _audit(paths)


def test_compact_artifact_lineage_rejects_equal_count_row_mismatch(tmp_path):
    paths = _artifacts(tmp_path)
    teacher = torch.load(paths["teacher"], weights_only=False)
    teacher["records"][0]["query_rows"] = torch.tensor([1, 4, 5])
    torch.save(teacher, paths["teacher"])
    with pytest.raises(ValueError, match="deployment rows"):
        _audit(paths)
