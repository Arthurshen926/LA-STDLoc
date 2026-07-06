import sys
import tempfile
import unittest
from argparse import ArgumentParser
from pathlib import Path
from unittest import mock


class CombinedArgsTest(unittest.TestCase):
    def test_missing_cfg_args_falls_back_to_command_line(self):
        from arguments import get_combined_args

        parser = ArgumentParser()
        parser.add_argument("--model_path")
        parser.add_argument("--source_path")

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "new_model"
            model_path.mkdir()
            argv = [
                "prog",
                "--model_path",
                str(model_path),
                "--source_path",
                "/data/scene",
            ]
            with mock.patch.object(sys, "argv", argv):
                args = get_combined_args(parser)

        self.assertEqual(args.model_path, str(model_path))
        self.assertEqual(args.source_path, "/data/scene")


if __name__ == "__main__":
    unittest.main()
