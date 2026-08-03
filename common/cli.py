"""Shared command-line fields for a frozen Gaussian scene."""

from __future__ import annotations

import os
import argparse


class SceneArguments:
    """Register and extract the external-prior and image-dataset arguments."""

    def __init__(self, parser) -> None:
        group = parser.add_argument_group("scene")
        group.add_argument("--sh_degree", type=int, default=3)
        group.add_argument("--source_path", required=True)
        group.add_argument("--feature_type", default="sp", choices=("sp", "superpoint"))
        group.add_argument("--gaussian_type", default="3dgs", choices=("2dgs", "3dgs"))
        group.add_argument("--model_path", required=True)
        group.add_argument("--images", default="processed")
        group.add_argument("--resolution", type=int, default=1)
        group.add_argument(
            "--white_background", action=argparse.BooleanOptionalAction, default=True
        )
        group.add_argument("--longest_edge", type=int, default=0)
        group.add_argument("--data_device", default="cpu")
        group.add_argument(
            "--norm_before_render", action=argparse.BooleanOptionalAction, default=True
        )

    @staticmethod
    def extract(args):
        args.source_path = os.path.abspath(args.source_path)
        args.model_path = os.path.abspath(args.model_path)
        return args


ModelParams = SceneArguments
