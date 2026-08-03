"""Deterministic runtime helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_output(quiet: bool) -> None:
    # The historical implementation wrapped stdout to append timestamps.  It
    # changed logs only, not computation; release CLIs keep normal stdout.
    if quiet:
        return
