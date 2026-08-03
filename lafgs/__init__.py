"""Stable public API for the frozen LaFGS paper mainline.

The historical research modules remain importable, but new release tooling
should depend on this package rather than on experiment scripts.
"""

from lafgs.protocol import MainlineProtocol, load_mainline_protocol

__all__ = ["MainlineProtocol", "load_mainline_protocol"]
