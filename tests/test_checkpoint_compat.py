import unittest

import torch


class CheckpointCompatTest(unittest.TestCase):
    def test_checkpoint_helpers_accept_tuple_and_dict_formats(self):
        from train_locaware import _is_locaware_checkpoint, _split_checkpoint_payload

        tuple_payload = ("model", 7)
        self.assertFalse(_is_locaware_checkpoint(tuple_payload))
        model_params, iteration, loc_state = _split_checkpoint_payload(tuple_payload)
        self.assertEqual(model_params, "model")
        self.assertEqual(iteration, 7)
        self.assertIsNone(loc_state)

        dict_payload = {
            "version": 2,
            "iteration": 11,
            "model_params": "model-v2",
            "localization_state": {"loc_opacity": torch.zeros(2, 1)},
        }
        self.assertTrue(_is_locaware_checkpoint(dict_payload))
        model_params, iteration, loc_state = _split_checkpoint_payload(dict_payload)
        self.assertEqual(model_params, "model-v2")
        self.assertEqual(iteration, 11)
        self.assertIn("loc_opacity", loc_state)


if __name__ == "__main__":
    unittest.main()
