"""Legacy imports are permitted only for explicit regression references."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "research/legacy"))
