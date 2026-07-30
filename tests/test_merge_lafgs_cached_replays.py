import json

import pytest

from scripts.merge_lafgs_cached_replays import merge


def _payload(name, te, *, raw, inlier_precision, inliers, matches):
    return {
        "schema": "lafgs_cached_deployment_replay",
        "split": "mapping_replay",
        "map": "/map.pt",
        "metric_state": "/metric.pt",
        "anchor_count": 10,
        "results": [
            {
                "query": name,
                "te_cm": te,
                "re_deg": 0.1,
                "hypotheses": 100,
                "inlier_count": inliers,
                "match_count": matches,
                "raw_gt_precision_2px": raw,
                "inlier_gt_precision_2px": inlier_precision,
                "matching_ms": 2.0,
                "context_ms": 1.0,
                "ransac_ms": 7.0,
            }
        ],
    }


def test_merge_preserves_cleanliness_and_runtime_metrics(tmp_path):
    paths = []
    for index, payload in enumerate(
        (
            _payload("a", 4.0, raw=0.1, inlier_precision=0.5, inliers=2, matches=10),
            _payload("b", 6.0, raw=0.2, inlier_precision=0.7, inliers=4, matches=10),
        )
    ):
        path = tmp_path / f"{index}.json"
        path.write_text(json.dumps(payload))
        paths.append(str(path))
    output = merge(paths)
    assert output["raw_gt_precision_2px_percent"] == pytest.approx(15.0)
    assert output["inlier_gt_precision_2px_percent"] == pytest.approx(60.0)
    assert output["mean_solver_inlier_ratio_percent"] == pytest.approx(30.0)
    assert output["total_ms_per_query"] == pytest.approx(10.0)
