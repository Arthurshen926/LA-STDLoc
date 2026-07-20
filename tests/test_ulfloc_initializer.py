import torch


def test_native_sparse_descriptor_sampling_matches_native_dense_sampler():
    from encoders.sp_encoder.export_image_embeddings import sample_descriptors
    from localization_training.ulf_initializer import (
        grid_index_to_physical,
        sample_dense_descriptors_at_image_uv,
    )

    torch.manual_seed(7)
    dense = torch.randn(1, 8, 4, 5)
    dense = torch.nn.functional.normalize(dense, dim=1)
    keypoints = torch.tensor([[0.0, 0.0], [7.0, 9.0], [31.0, 24.0]])
    sparse = sample_descriptors(keypoints[None], dense)[0].T
    fused = sample_dense_descriptors_at_image_uv(
        dense,
        grid_index_to_physical(keypoints),
        (32, 40),
    )
    cosine = (sparse * fused).sum(dim=1)
    assert torch.allclose(cosine, torch.ones_like(cosine), atol=1e-6)


def test_surface_normal_and_geometry_weight_use_surfel_local_z_axis():
    from localization_training.ulf_initializer import (
        geometry_view_weights,
        surface_normals_from_rotation,
    )

    rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    normals = surface_normals_from_rotation(rotation, torch.tensor([[1.0, 1.0]]))
    assert torch.allclose(normals, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-6)
    fronto_parallel = geometry_view_weights(
        torch.tensor([[0.0, 0.0, 0.0]]), normals, torch.tensor([0.0, 0.0, 2.0])
    )
    grazing = geometry_view_weights(
        torch.tensor([[0.0, 0.0, 0.0]]), normals, torch.tensor([2.0, 0.0, 0.0])
    )
    assert torch.allclose(fronto_parallel, torch.ones(1), atol=1e-6)
    assert torch.allclose(grazing, torch.zeros(1), atol=1e-6)


def test_consensus_nearest_keypoint_distance_is_exact_without_faiss():
    from localization_training.ulf_initializer import nearest_keypoint_distance

    projected = torch.tensor([[1.0, 1.0], [3.0, 4.0], [10.0, 0.0]])
    keypoints = torch.tensor([[1.0, 1.0], [4.0, 4.0]])
    distance = nearest_keypoint_distance(projected, keypoints, chunk_size=1)
    assert torch.allclose(distance, torch.tensor([0.0, 1.0, (52.0) ** 0.5]))


def test_pixel_center_projection_returns_feature_grid_index():
    from localization_training.direct_landmark_teacher import project_landmarks_to_query

    xyz = torch.tensor([[0.0, 0.0, 1.0]])
    K = torch.tensor([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]])
    pose = torch.eye(4)
    uv, _, valid = project_landmarks_to_query(
        xyz,
        K,
        pose,
        height=10,
        width=10,
        pixel_center_offset=0.5,
    )
    assert valid.tolist() == [True]
    assert torch.allclose(uv, torch.tensor([[4.5, 4.5]]))
