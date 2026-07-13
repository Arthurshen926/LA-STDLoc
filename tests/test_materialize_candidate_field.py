import pickle
from argparse import Namespace

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from scripts.materialize_candidate_field import materialize_candidate_field


def test_materialize_candidate_field_only_changes_selected_loc_features(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "output"
    point_cloud = source / "point_cloud" / "iteration_7"
    detector = source / "detector_field"
    point_cloud.mkdir(parents=True)
    detector.mkdir()

    data = np.zeros(
        4,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("opacity", "f4"),
            ("loc_0", "f4"),
            ("loc_1", "f4"),
        ],
    )
    data["x"] = np.arange(4, dtype=np.float32)
    data["opacity"] = np.linspace(0.1, 0.4, 4, dtype=np.float32)
    data["loc_0"] = [0.1, 0.2, 0.3, 0.4]
    data["loc_1"] = [0.5, 0.6, 0.7, 0.8]
    PlyData([PlyElement.describe(data, "vertex")]).write(
        point_cloud / "point_cloud.ply"
    )
    torch.save({"loc_opacity": torch.ones(4, 1)}, point_cloud / "loc_state.pt")
    (source / "cfg_args").write_text(
        repr(Namespace(model_path=str(source), source_path="/dataset", gaussian_type="2dgs"))
    )
    (source / "cameras.json").write_text("[]\n")
    (source / "input.ply").write_bytes(b"input")

    indices = torch.tensor([1, 3])
    features = torch.tensor([[0.9, -0.1], [-0.4, 0.7]], dtype=torch.float32)
    state = {
        "iteration": 5,
        "landmark_indices": indices,
        "landmark_features": features,
    }
    torch.save(state, detector / "5_candidate_teacher_state.pt")
    torch.save({"weight": torch.ones(1)}, detector / "5_detector.pth")
    torch.save({"landmark_indices": indices}, detector / "landmark_meta.pt")
    with open(detector / "sampled_idx.pkl", "wb") as handle:
        pickle.dump(indices.numpy(), handle)

    manifest = materialize_candidate_field(
        str(source),
        "detector_field/5_candidate_teacher_state.pt",
        str(output),
        iteration=7,
        detector_folder="detector_field",
    )

    result = PlyData.read(output / "point_cloud" / "iteration_7" / "point_cloud.ply")
    output_data = result["vertex"].data
    np.testing.assert_array_equal(output_data["x"], data["x"])
    np.testing.assert_array_equal(output_data["opacity"], data["opacity"])
    np.testing.assert_allclose(output_data["loc_0"][[1, 3]], features[:, 0].numpy())
    np.testing.assert_allclose(output_data["loc_1"][[1, 3]], features[:, 1].numpy())
    np.testing.assert_array_equal(output_data["loc_0"][[0, 2]], data["loc_0"][[0, 2]])
    np.testing.assert_array_equal(output_data["loc_1"][[0, 2]], data["loc_1"][[0, 2]])
    assert manifest["geometry_fields_exact"] is True
    assert manifest["unselected_localization_fields_exact"] is True
    assert (output / "detector_field" / "5_detector.pth").is_file()
