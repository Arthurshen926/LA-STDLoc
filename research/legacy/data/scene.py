"""Read-only mapping scene for localization-map reconstruction."""

from __future__ import annotations

from pathlib import Path

from data.cameras import load_camera
from data.datasets import ColmapDataset
from priors.models import FrozenGaussianModel


class FrozenScene:
    """Load mapping cameras and one frozen external Gaussian checkpoint."""

    def __init__(
        self,
        args,
        gaussians: FrozenGaussianModel,
        *,
        load_iteration: int,
        load_test_cameras: bool = False,
    ) -> None:
        if load_iteration is None:
            raise ValueError("A frozen Gaussian iteration is required")
        self.args = args
        self.model_path = str(args.model_path)
        self.source_path = str(args.source_path)
        self.loaded_iter = int(load_iteration)
        self.gaussians = gaussians
        self.dataset = ColmapDataset(self.source_path, images=args.images)
        point_cloud = (
            Path(self.model_path)
            / "point_cloud"
            / f"iteration_{self.loaded_iter}"
            / "point_cloud.ply"
        )
        if not point_cloud.is_file():
            raise FileNotFoundError(f"Frozen Gaussian PLY is missing: {point_cloud}")
        gaussians.load_ply(point_cloud, loc_feature_dim=256)
        self._mapping = self._load_split("mapping")
        self._test = self._load_split("test") if load_test_cameras else []

    def _load_split(self, split: str):
        return [
            load_camera(
                self.dataset,
                record,
                uid=index,
                resolution=self.args.resolution,
                data_device=self.args.data_device,
            )
            for index, record in enumerate(self.dataset.split(split))
        ]

    def getTrainCameras(self):
        return list(self._mapping)

    def getTestCameras(self):
        return list(self._test)
