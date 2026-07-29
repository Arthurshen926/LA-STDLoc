import json
import subprocess
import sys


def test_compose_policy_replay_replaces_queries(tmp_path):
    base = {
        "split": "crossfold_mapping",
        "anchor_count": 3,
        "results": [
            {"query": "a", "te_cm": 10.0, "re_deg": 1.0, "hypotheses": 100},
            {"query": "b", "te_cm": 20.0, "re_deg": 2.0, "hypotheses": 100},
        ],
    }
    continuation = {
        "results": [
            {"query": "b", "te_cm": 2.0, "re_deg": 0.2, "hypotheses": 300}
        ]
    }
    base_path = tmp_path / "base.json"
    continuation_path = tmp_path / "continuation.json"
    output_path = tmp_path / "output.json"
    base_path.write_text(json.dumps(base))
    continuation_path.write_text(json.dumps(continuation))
    subprocess.run(
        [
            sys.executable,
            "scripts/compose_lafgs_policy_replay.py",
            "--base",
            str(base_path),
            "--continuation",
            str(continuation_path),
            "--output",
            str(output_path),
        ],
        check=True,
    )
    output = json.loads(output_path.read_text())
    assert output["continuation_query_count"] == 1
    assert output["median_te_cm"] == 6.0
    assert output["mean_hypotheses"] == 200.0
