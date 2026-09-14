import json
import sys

import pytest

from common.hashing import sha256_file
from scripts.materialize_v14_certified_view import main


def _observer(tmp_path, *, source_role: str, observer_role: str):
    certified = tmp_path / f"{source_role}.json"
    certified.write_text(
        json.dumps(
            {
                "view_role": source_role,
                "uses_test_queries": False,
                "map_mutation_count": 0,
                "records": [],
            }
        )
    )
    observer = tmp_path / f"{observer_role}.json"
    observer.write_text(
        json.dumps(
            {
                "schema": "lafgs_v9_no_loo_causal_feedback_batch",
                "version": 2,
                "uses_test_queries": False,
                "loo_used": False,
                "accepted_query_row_policy": "v2_row_valid_only",
                "role": observer_role,
                "input": {
                    "certified_batch": str(certified),
                    "certified_batch_sha256": sha256_file(certified),
                },
                "records": [],
            }
        )
    )
    return observer


def test_certified_view_cannot_relabel_control_as_confirmation(
    tmp_path, monkeypatch
) -> None:
    observer = _observer(
        tmp_path,
        source_role="feedback_query",
        observer_role="heldout_control",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_v14_certified_view.py",
            "--observer-batch",
            str(observer),
            "--view-role",
            "confirmation_query",
            "--output",
            str(tmp_path / "view.json"),
        ],
    )

    with pytest.raises(ValueError, match="cannot materialize"):
        main()


def test_certified_view_accepts_explicit_confirmation_observer(
    tmp_path, monkeypatch
) -> None:
    observer = _observer(
        tmp_path,
        source_role="confirmation_query",
        observer_role="confirmation_observer",
    )
    output = tmp_path / "view.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "materialize_v14_certified_view.py",
            "--observer-batch",
            str(observer),
            "--view-role",
            "confirmation_query",
            "--output",
            str(output),
        ],
    )

    main()

    payload = json.loads(output.read_text())
    assert payload["observer_role"] == "confirmation_observer"
    assert payload["source_view_role"] == "confirmation_query"
