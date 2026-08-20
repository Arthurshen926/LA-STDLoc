import json
import os
from pathlib import Path
import time

import torch

from scripts.probe_rendered_rgb_track_map import _gather_query_keypoints
from scripts.run_render_track_single_scene import (
    MAPPING_STAGE_NAMES,
    collect_scene_build_timing,
)


def test_collect_scene_build_timing_separates_current_work_from_cache_hits(
    tmp_path: Path,
) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    invocation_started = time.time()
    stages = (*MAPPING_STAGE_NAMES, "test_seed2026")
    for index, stage in enumerate(stages):
        path = logs / f"{stage}.timing.json"
        path.write_text(json.dumps({"returncode": 0, "seconds": index + 0.5}))
        if stage in {"base", "selector"}:
            current = invocation_started + 1.0
            os.utime(path, (current, current))
            continue
        old = invocation_started - 10.0
        os.utime(path, (old, old))

    payload = collect_scene_build_timing(
        tmp_path,
        invocation_started_unix=invocation_started,
        evaluation_stage="test_seed2026",
    )
    assert payload["schema"] == "lafgs_v4_render_track_build_timing"
    assert payload["executed_stages"] == ["base", "selector"]
    assert set(payload["cache_hit_stages"]) == set(stages) - {"base", "selector"}
    assert payload["current_mapping_subprocess_seconds"] == 0.5 + 5.5


def test_ragged_keypoint_gather_is_exact() -> None:
    keypoints = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[5.0, 6.0]]),
        torch.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
    ]
    query = torch.tensor([2, 0, 1, 2, 0])
    row = torch.tensor([1, 0, 0, 2, 1])
    expected = torch.stack([keypoints[int(q)][int(k)] for q, k in zip(query, row)])
    assert torch.equal(_gather_query_keypoints(keypoints, query, row), expected)
