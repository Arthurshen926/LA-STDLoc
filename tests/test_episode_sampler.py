import os
import tempfile
import unittest

import torch


class EpisodeSamplerTest(unittest.TestCase):
    def test_pose_noise_and_sparse_cache_round_trip(self):
        from localization_training.episode_sampler import (
            SparsePoseCache,
            apply_pose_noise,
            sample_noise_from_distribution,
        )

        pose = torch.eye(4)
        noisy = apply_pose_noise(pose, torch.tensor([0.1, -0.2, 0.3, 0.01, 0.02, -0.03]))
        self.assertEqual(noisy.shape, (4, 4))
        self.assertFalse(torch.allclose(noisy, pose))

        errors = {
            "translation": torch.tensor([0.01, 0.2, 0.4]),
            "rotation_deg": torch.tensor([0.5, 2.0, 5.0]),
        }
        sampled = sample_noise_from_distribution(errors, quantile=0.5, generator=torch.Generator().manual_seed(0))
        self.assertEqual(sampled.shape, (6,))
        self.assertLessEqual(torch.linalg.norm(sampled[:3]).item(), 0.21)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cache.pt")
            cache = SparsePoseCache(path)
            cache.update("image_a", noisy, inliers=12, ae=1.5, te=23.0)
            cache.save()
            restored = SparsePoseCache(path)
            restored.load()
            item = restored.get("image_a")
            self.assertEqual(item["inliers"], 12)
            self.assertTrue(torch.allclose(item["pose_w2c"], noisy))

            cache.update(
                "image_b",
                noisy,
                inliers=12,
                ae=2.0,
                te=20.0,
                dense_pose_w2c=torch.eye(4),
                dense_inliers=18,
                dense_ae=1.0,
                dense_te=10.0,
            )
            self.assertEqual(cache.get("image_b")["dense_inliers"], 18)
            self.assertEqual(cache.get("image_b")["dense_te"], 10.0)

    def test_mixed_mode_respects_sparse_probability(self):
        from localization_training.episode_sampler import EpisodeSampler, SparsePoseCache

        class Camera:
            image_name = "image_a"
            world_view_transform = torch.eye(4)

        sparse_pose = torch.eye(4)
        sparse_pose[0, 3] = 1.0
        cache = SparsePoseCache("unused.pt")
        cache.update("image_a", sparse_pose, inliers=20, ae=0.1, te=1.0)

        sampler = EpisodeSampler(
            sparse_pose_cache=cache,
            query_mode="mixed",
            mixed_sparse_probability=0.0,
            error_distribution={"translation": torch.tensor([0.0]), "rotation_deg": torch.tensor([0.0])},
        )
        episode = sampler.sample(Camera(), generator=torch.Generator().manual_seed(0))
        self.assertEqual(episode.source, "noise")
        self.assertTrue(torch.allclose(episode.pose_init_w2c, torch.eye(4)))
        self.assertEqual(episode.sparse_meta["inliers"], 20)

        sampler = EpisodeSampler(
            sparse_pose_cache=cache,
            query_mode="mixed",
            mixed_sparse_probability=1.0,
        )
        episode = sampler.sample(Camera(), generator=torch.Generator().manual_seed(0))
        self.assertEqual(episode.source, "sparse")
        self.assertTrue(torch.allclose(episode.pose_init_w2c, sparse_pose))

    def test_sparse_mode_prefers_teacher_cache_key(self):
        from localization_training.episode_sampler import EpisodeSampler, SparsePoseCache

        class Camera:
            image_name = "synthetic/000000.png"
            teacher_cache_key = "synthetic_rgb:synthetic/000000.png"
            world_view_transform = torch.eye(4)

        sparse_pose = torch.eye(4)
        sparse_pose[1, 3] = 2.0
        cache = SparsePoseCache("unused.pt")
        cache.update(Camera.teacher_cache_key, sparse_pose, inliers=24, ae=0.2, te=2.0)

        sampler = EpisodeSampler(sparse_pose_cache=cache, query_mode="sparse")
        episode = sampler.sample(Camera(), generator=torch.Generator().manual_seed(0))

        self.assertEqual(episode.source, "sparse")
        self.assertTrue(torch.allclose(episode.pose_init_w2c, sparse_pose))
        self.assertEqual(episode.sparse_meta["inliers"], 24)

    def test_sparse_mode_uses_stage_failed_cache_by_default(self):
        from localization_training.episode_sampler import EpisodeSampler, SparsePoseCache

        class Camera:
            image_name = "synthetic/000000.png"
            teacher_cache_key = "synthetic_rgb:synthetic/000000.png"
            world_view_transform = torch.eye(4)

        bad_sparse_pose = torch.eye(4)
        bad_sparse_pose[0, 3] = 10.0
        cache = SparsePoseCache("unused.pt")
        cache.update(Camera.teacher_cache_key, bad_sparse_pose, inliers=24, ae=90.0, te=1000.0)
        cache.items[Camera.teacher_cache_key]["failure_stage"] = "sparse_failure"

        sampler = EpisodeSampler(
            sparse_pose_cache=cache,
            query_mode="sparse",
            error_distribution={"translation": torch.tensor([0.0]), "rotation_deg": torch.tensor([0.0])},
        )
        episode = sampler.sample(Camera(), generator=torch.Generator().manual_seed(0))

        self.assertEqual(episode.source, "sparse")
        self.assertTrue(torch.allclose(episode.pose_init_w2c, bad_sparse_pose))

    def test_sparse_mode_can_opt_in_to_reject_cache_stage_sparse_failure(self):
        from localization_training.episode_sampler import EpisodeSampler, SparsePoseCache

        class Camera:
            image_name = "synthetic/000000.png"
            teacher_cache_key = "synthetic_rgb:synthetic/000000.png"
            world_view_transform = torch.eye(4)

        bad_sparse_pose = torch.eye(4)
        bad_sparse_pose[0, 3] = 10.0
        cache = SparsePoseCache("unused.pt")
        cache.update(Camera.teacher_cache_key, bad_sparse_pose, inliers=24, ae=90.0, te=1000.0)
        cache.items[Camera.teacher_cache_key]["failure_stage"] = "sparse_failure"

        sampler = EpisodeSampler(
            sparse_pose_cache=cache,
            query_mode="sparse",
            error_distribution={"translation": torch.tensor([0.0]), "rotation_deg": torch.tensor([0.0])},
            exclude_sparse_failure_stages=True,
        )
        episode = sampler.sample(Camera(), generator=torch.Generator().manual_seed(0))

        self.assertEqual(episode.source, "noise")
        self.assertTrue(torch.allclose(episode.pose_init_w2c, torch.eye(4)))

    def test_noise_mode_preserves_sparse_meta_for_advantage_gates(self):
        from localization_training.episode_sampler import EpisodeSampler, SparsePoseCache

        class Camera:
            image_name = "image_a"
            world_view_transform = torch.eye(4)

        sparse_pose = torch.eye(4)
        sparse_pose[0, 3] = 1.0
        cache = SparsePoseCache("unused.pt")
        cache.update(
            "image_a",
            sparse_pose,
            inliers=20,
            ae=2.0,
            te=20.0,
            dense_pose_w2c=torch.eye(4),
            dense_inliers=30,
            dense_ae=1.0,
            dense_te=10.0,
        )

        sampler = EpisodeSampler(
            sparse_pose_cache=cache,
            query_mode="noise",
            error_distribution={"translation": torch.tensor([0.0]), "rotation_deg": torch.tensor([0.0])},
        )
        episode = sampler.sample(Camera(), generator=torch.Generator().manual_seed(0))

        self.assertEqual(episode.source, "noise")
        self.assertTrue(torch.allclose(episode.pose_init_w2c, torch.eye(4)))
        self.assertEqual(episode.sparse_meta["te"], 20.0)
        self.assertEqual(episode.sparse_meta["dense_te"], 10.0)

    def test_empirical_noise_sampling_uses_observed_error_magnitudes(self):
        from localization_training.episode_sampler import sample_noise_from_distribution

        errors = {
            "translation": torch.tensor([0.01, 0.2, 0.4]),
            "rotation_deg": torch.tensor([0.5, 2.0, 5.0]),
        }
        sampled = sample_noise_from_distribution(
            errors,
            sampling="empirical",
            generator=torch.Generator().manual_seed(4),
        )
        sampled_t = torch.linalg.norm(sampled[:3])
        candidates = torch.tensor([0.01, 0.2, 0.4])
        self.assertLess(torch.min(torch.abs(candidates - sampled_t)).item(), 1e-6)

    def test_support_query_split_is_deterministic_and_disjoint(self):
        from localization_training.episode_sampler import split_support_query_cameras

        class Camera:
            def __init__(self, name):
                self.image_name = name

        cameras = [Camera(f"image_{idx}") for idx in range(10)]
        support_a, query_a = split_support_query_cameras(cameras, query_ratio=0.3, seed=7)
        support_b, query_b = split_support_query_cameras(cameras, query_ratio=0.3, seed=7)

        self.assertEqual([cam.image_name for cam in query_a], [cam.image_name for cam in query_b])
        self.assertEqual(len(query_a), 3)
        self.assertEqual(len(support_a), 7)
        self.assertTrue({cam.image_name for cam in support_a}.isdisjoint({cam.image_name for cam in query_a}))

    def test_support_query_sequence_block_holds_out_whole_sequences(self):
        from localization_training.episode_sampler import split_support_query_cameras

        class Camera:
            def __init__(self, name):
                self.image_name = name

        cameras = [
            Camera(f"seq0/frame{idx:05d}.png") for idx in range(4)
        ] + [
            Camera(f"seq1/frame{idx:05d}.png") for idx in range(4)
        ]

        support, query = split_support_query_cameras(cameras, query_ratio=0.5, seed=0, mode="sequence_block")

        query_sequences = {cam.image_name.split("/")[0] for cam in query}
        self.assertEqual(len(query_sequences), 1)
        self.assertTrue({cam.image_name for cam in support}.isdisjoint({cam.image_name for cam in query}))

    def test_support_query_temporal_block_holds_out_contiguous_frames(self):
        from localization_training.episode_sampler import split_support_query_cameras

        class Camera:
            def __init__(self, name):
                self.image_name = name

        cameras = [Camera(f"frame{idx:05d}.png") for idx in range(10)]

        _, query = split_support_query_cameras(cameras, query_ratio=0.3, seed=0, mode="temporal_block")

        query_indices = [int(cam.image_name[5:10]) for cam in query]
        self.assertEqual(len(query_indices), 3)
        self.assertEqual(query_indices, list(range(query_indices[0], query_indices[0] + len(query_indices))))

    def test_support_query_split_keeps_one_support_and_query_for_extreme_ratios(self):
        from localization_training.episode_sampler import split_support_query_cameras

        class Camera:
            def __init__(self, name):
                self.image_name = name

        cameras = [Camera("a"), Camera("b")]
        support, query = split_support_query_cameras(cameras, query_ratio=0.99, seed=0)

        self.assertEqual(len(support), 1)
        self.assertEqual(len(query), 1)

    def test_interpolated_novel_view_samples_between_adjacent_cameras(self):
        from localization_training.episode_sampler import sample_interpolated_novel_view

        class Camera:
            def __init__(self, name, x):
                self.image_name = name
                self.FoVx = 0.7
                self.FoVy = 0.6
                self.world_view_transform = torch.eye(4)
                self.world_view_transform[3, 0] = -x

        cameras = [Camera("frame_000.png", 0.0), Camera("frame_001.png", 2.0)]

        synthetic = sample_interpolated_novel_view(
            cameras,
            generator=torch.Generator().manual_seed(0),
            alpha_min=0.5,
            alpha_max=0.5,
        )

        self.assertEqual(synthetic.image_name, "synthetic/frame_000.png__frame_001.png__0.500")
        self.assertEqual(synthetic.source, "synthetic_interpolate")
        self.assertAlmostEqual(synthetic.FoVx, 0.7)
        self.assertAlmostEqual(synthetic.FoVy, 0.6)
        pose_w2c = synthetic.world_view_transform.transpose(0, 1)
        center = torch.linalg.inv(pose_w2c)[:3, 3]
        self.assertTrue(torch.allclose(center, torch.tensor([1.0, 0.0, 0.0]), atol=1e-5))

    def test_interpolated_novel_view_respects_pair_weights(self):
        from localization_training.episode_sampler import sample_interpolated_novel_view

        class Camera:
            def __init__(self, name, x):
                self.image_name = name
                self.FoVx = 0.7
                self.FoVy = 0.6
                self.world_view_transform = torch.eye(4)
                self.world_view_transform[3, 0] = -x

        cameras = [Camera(str(index), float(index)) for index in range(4)]
        synthetic = sample_interpolated_novel_view(
            cameras,
            generator=torch.Generator().manual_seed(0),
            alpha_min=0.5,
            alpha_max=0.5,
            pair_weights=torch.tensor([0.0, 1.0, 0.0]),
        )

        self.assertEqual(synthetic.train_index_a, 1)
        self.assertEqual(synthetic.train_index_b, 2)

    def test_spatial_offset_novel_view_moves_off_train_trajectory(self):
        from localization_training.episode_sampler import sample_spatial_offset_novel_view

        class Camera:
            def __init__(self, image_name, center):
                c2w = torch.eye(4)
                c2w[:3, 3] = torch.tensor(center, dtype=torch.float32)
                w2c = torch.linalg.inv(c2w)
                self.image_name = image_name
                self.world_view_transform = w2c.transpose(0, 1)
                self.FoVx = 0.8
                self.FoVy = 0.6
                self.image_width = 8
                self.image_height = 6

        cameras = [
            Camera("seq/frame000.png", [0.0, 0.0, 0.0]),
            Camera("seq/frame001.png", [1.0, 0.0, 0.0]),
            Camera("seq/frame002.png", [2.0, 0.0, 0.0]),
        ]

        view = sample_spatial_offset_novel_view(
            cameras,
            generator=torch.Generator().manual_seed(3),
            min_offset_ratio=2.0,
            max_offset_ratio=2.0,
            yaw_deg=0.0,
            height_offset_ratio=0.0,
        )
        pose_w2c = view.world_view_transform.transpose(0, 1)
        center = torch.linalg.inv(pose_w2c)[:3, 3]

        self.assertEqual(view.source, "synthetic_spatial_offset")
        self.assertGreaterEqual(view.nearest_train_distance, 1.9)
        self.assertGreater(torch.linalg.norm(center[1:]).item(), 1.9)
        self.assertIn("spatial_offset", view.image_name)


if __name__ == "__main__":
    unittest.main()
