import re
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_locaware_cambridge_full.sh"


class FullRunScriptArgsTest(unittest.TestCase):
    def _command_blocks(self, command_name):
        text = SCRIPT.read_text()
        pattern = re.compile(rf'"\$PYTHON" {re.escape(command_name)} \\\n(?P<body>.*?)(?=\n"\$PYTHON"|\nif |\nelse|\nfi|\Z)', re.S)
        return [match.group("body") for match in pattern.finditer(text)]

    def test_eval_commands_do_not_receive_training_only_args(self):
        for command_name in ("stdloc.py", "cache_sparse_poses.py"):
            with self.subTest(command=command_name):
                blocks = self._command_blocks(command_name)
                self.assertTrue(blocks, f"{command_name} is not invoked by the full run script")
                for block in blocks:
                    self.assertIn('"${DATA_ARGS[@]}"', block)
                    self.assertNotIn('"${TRAIN_ARGS[@]}"', block)
                    self.assertNotIn('"${COMMON_ARGS[@]}"', block)

    def test_la_training_phases_are_resume_safe(self):
        text = SCRIPT.read_text()
        for phase_end in ("FEATURE_END", "GEOMETRY_END", "TOPOLOGY_END", "CLOSED_LOOP_END"):
            with self.subTest(phase_end=phase_end):
                self.assertIn(f'if ! point_cloud_exists "$LA_MODEL" "${phase_end}"; then', text)


if __name__ == "__main__":
    unittest.main()
