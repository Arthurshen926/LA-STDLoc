from scripts.slim_lafgs_query_cache import SPARSE_KEYS


def test_sparse_cache_keeps_micro_anchor_geometry_fields():
    required = {
        "pose_w2c",
        "native_keypoints",
        "native_descriptors",
        "native_K",
        "native_depth",
    }

    assert required.issubset(SPARSE_KEYS)
