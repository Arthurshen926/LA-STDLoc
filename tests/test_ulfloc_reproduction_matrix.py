import pickle

import numpy as np

from scripts.run_ulfloc_reproduction_matrix import materialize_torch_masks


def test_ulfloc_staged_masks_preserve_native_resolution_by_default(tmp_path):
    shared = np.asarray([[True, False, True], [False, True, False]], dtype=np.bool_)
    source = tmp_path / "masks.pkl"
    with source.open("wb") as stream:
        pickle.dump({"frame.png": (shared, shared, shared)}, stream)
    output = tmp_path / "staged.pkl"
    report = materialize_torch_masks(source, output)
    with output.open("rb") as stream:
        payload = pickle.load(stream)
    assert tuple(payload["frame.png"][0].shape) == (2, 3)
    assert report["source_shape"] == [2, 3]
    assert report["staged_shape"] == [2, 3]
    assert report["longest_edge"] is None


def test_ulfloc_mask_adapter_can_still_materialize_a_requested_render_size(tmp_path):
    shared = np.ones((4, 8), dtype=np.bool_)
    source = tmp_path / "masks.pkl"
    with source.open("wb") as stream:
        pickle.dump({"frame.png": (shared, shared, shared)}, stream)
    report = materialize_torch_masks(source, tmp_path / "staged.pkl", longest_edge=4)
    assert report["staged_shape"] == [2, 4]
