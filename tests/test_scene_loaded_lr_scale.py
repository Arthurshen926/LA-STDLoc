from types import SimpleNamespace

from scene import Scene
from scene.dataset_readers import sceneLoadTypeCallbacks


def test_loaded_scene_restores_spatial_lr_scale(tmp_path, monkeypatch):
    source = tmp_path / "scene"
    (source / "sparse").mkdir(parents=True)
    model = tmp_path / "model"
    point_cloud = model / "point_cloud" / "iteration_7"
    point_cloud.mkdir(parents=True)

    scene_info = SimpleNamespace(
        train_cameras=[],
        test_cameras=[],
        nerf_normalization={"radius": 12.5},
        loc_feature_dim=256,
        ply_path=str(tmp_path / "unused.ply"),
    )
    monkeypatch.setitem(
        sceneLoadTypeCallbacks,
        "Colmap",
        lambda *args, **kwargs: scene_info,
    )

    class FakeGaussians:
        spatial_lr_scale = 0.0

        def load_ply(self, path, loc_feature_dim=None):
            self.loaded_ply = path
            self.loaded_loc_feature_dim = loc_feature_dim

        def load_localization_state(self, path):
            self.loaded_loc_state = path

    args = SimpleNamespace(
        model_path=str(model),
        source_path=str(source),
        feature_type="sp",
        images="processed",
        eval=False,
        longest_edge=640,
    )
    gaussians = FakeGaussians()

    Scene(args, gaussians, load_iteration=7, preload_cameras=False)

    assert gaussians.spatial_lr_scale == 12.5
    assert gaussians.loaded_loc_feature_dim == 256
