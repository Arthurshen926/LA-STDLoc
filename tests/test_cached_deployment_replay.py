from pathlib import Path


def test_cached_replay_uses_masked_function_graph_rows():
    source = Path("scripts/evaluate_lafgs_map_on_query_cache.py").read_text()
    assert 'graph["records"][query_index]["query_rows"]' in source
    assert 'cached["native_descriptors"]' in source
    assert "ransac_seed=int(args.seed)" in source
    assert ".max(dim=1)" in source
    assert "partial replay identity does not match current run" in source
    assert "os.replace(temporary, path)" in source


def test_clean_runner_uses_formal_stage_a_1000_path():
    source = Path("scripts/run_lafgs_v10_clean_rebuild.sh").read_text()
    assert 'LAFGS_STAGE_A_STEPS=1000' in source
    assert 'stage_a_combined_1000' in source
    assert 'stage_a_2500' not in source
