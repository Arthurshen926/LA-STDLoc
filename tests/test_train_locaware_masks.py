import unittest

import torch


class TrainLocawareMaskTest(unittest.TestCase):
    def test_resize_bool_mask_matches_target_hw(self):
        from train_locaware import _resize_bool_mask

        mask = torch.ones(1, 8, 8, dtype=torch.bool)
        resized = _resize_bool_mask(mask, (4, 6))

        self.assertEqual(resized.shape, (1, 4, 6))
        self.assertEqual(resized.dtype, torch.bool)

    def test_geometry_anchor_refreshes_after_topology_point_count_change(self):
        from train_locaware import _refresh_geometry_anchor_if_point_count_changed

        class FakeGaussians:
            def __init__(self, n):
                self._xyz = torch.ones(n, 3)
                self._scaling = torch.ones(n, 3)
                self._rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)

            @property
            def get_xyz(self):
                return self._xyz

        old_anchor = {
            "xyz": torch.zeros(2, 3),
            "scaling": torch.zeros(2, 3),
            "rotation": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        }

        refreshed = _refresh_geometry_anchor_if_point_count_changed(FakeGaussians(3), old_anchor)
        self.assertEqual(refreshed["xyz"].shape[0], 3)
        self.assertTrue(torch.equal(refreshed["xyz"], torch.ones(3, 3)))

        same = _refresh_geometry_anchor_if_point_count_changed(FakeGaussians(3), refreshed)
        self.assertIs(same, refreshed)


if __name__ == "__main__":
    unittest.main()
