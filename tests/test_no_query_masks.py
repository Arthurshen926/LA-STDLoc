from unittest.mock import patch
from anygsloc.data.datasets import ColmapDataset


def test_no_mask_policy_does_not_read_mask_files(tmp_path):
    with patch.object(ColmapDataset, '_load_test_names', return_value=frozenset()), \
         patch.object(ColmapDataset, '_load_cameras', return_value=[]), \
         patch.object(ColmapDataset, '_load_masks', side_effect=AssertionError('mask file accessed')):
        dataset = ColmapDataset(tmp_path, load_masks=False)
        assert dataset._masks is None


def test_explicit_mask_loading_is_preserved(tmp_path):
    with patch.object(ColmapDataset, '_load_test_names', return_value=frozenset()), \
         patch.object(ColmapDataset, '_load_cameras', return_value=[]), \
         patch.object(ColmapDataset, '_load_masks', return_value={'sentinel': 1}) as loader:
        dataset = ColmapDataset(tmp_path, load_masks=True)
        loader.assert_called_once()
        assert dataset._masks == {'sentinel': 1}
