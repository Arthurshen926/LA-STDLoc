"""Exact CPU tensor identities for scientific artifact contracts."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from typing import Any

import torch


def tensor_bytes(value: Any) -> bytes:
    """Return contiguous raw tensor bytes without dtype coercion."""
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    # Viewing a zero-dimensional tensor as bytes is illegal in torch.  Flatten
    # first; shape is framed separately by every caller that forms an identity.
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_bitwise_equal(left: Any, right: Any) -> bool:
    """Compare dtype, shape, layout, and raw bytes (including signed zero)."""
    left_tensor = torch.as_tensor(left).detach().cpu().contiguous()
    right_tensor = torch.as_tensor(right).detach().cpu().contiguous()
    return (
        left_tensor.dtype == right_tensor.dtype
        and left_tensor.layout == right_tensor.layout
        and tuple(left_tensor.shape) == tuple(right_tensor.shape)
        and tensor_bytes(left_tensor) == tensor_bytes(right_tensor)
    )


def tensor_identity(value: Any) -> dict:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    raw = tensor_bytes(tensor)
    return {
        "dtype": str(tensor.dtype),
        "layout": str(tensor.layout),
        "shape": list(tensor.shape),
        "byte_count": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "framed_sha256": hashlib.sha256(
            json.dumps(
                {
                    "dtype": str(tensor.dtype),
                    "layout": str(tensor.layout),
                    "shape": list(tensor.shape),
                    "bytes_sha256": hashlib.sha256(raw).hexdigest(),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("ascii")
        ).hexdigest(),
    }


def recursive_bitwise_equal(left: Any, right: Any) -> bool:
    """Strict recursive equality with raw tensor/float representations."""
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and tensor_bitwise_equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(recursive_bitwise_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(recursive_bitwise_equal(a, b) for a, b in zip(left, right))
        )
    if isinstance(left, float) or isinstance(right, float):
        if type(left) is not float or type(right) is not float:
            return False
        if math.isnan(left) or math.isnan(right):
            return math.isnan(left) and math.isnan(right) and struct.pack(
                ">d", left
            ) == struct.pack(">d", right)
        return struct.pack(">d", left) == struct.pack(">d", right)
    return type(left) is type(right) and left == right
