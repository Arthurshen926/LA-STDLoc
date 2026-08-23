from scripts.make_v6_sequence_block_split import build_sequence_block_split


def test_sequence_block_split_never_splits_a_sequence() -> None:
    names = ["seq1/a.png", "seq1/b.png", "seq2/a.png", "seq3/a.png"]
    feedback = {
        "schema": "self_localization_feedback_v1",
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "query_names": names,
        "records": [
            {"failure_layers": ["L3"]},
            {"failure_layers": []},
            {"failure_layers": ["L4"]},
            {"failure_layers": ["L3", "L4"]},
        ],
    }
    split = build_sequence_block_split(
        feedback,
        source_feedback_sha256="f" * 64,
        modulus=2,
        training_remainders=[1],
    )
    assert split["training_query_indices"] == [0, 1, 3]
    assert split["validation_query_indices"] == [2]
    assert split["training_summary"]["sequences"] == ["seq1", "seq3"]
    assert split["training_summary"]["failure_layer_counts"]["L3"] == 2
    assert split["validation_summary"]["failure_layer_counts"]["L4"] == 1

    explicit = build_sequence_block_split(
        feedback,
        source_feedback_sha256="f" * 64,
        validation_sequences=["seq2"],
    )
    assert explicit["training_query_indices"] == [0, 1, 3]
    assert explicit["validation_query_indices"] == [2]
    assert explicit["rule"]["validation_sequences"] == ["seq2"]
