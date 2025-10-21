"""
Bit-level compression helpers for observation payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np


@dataclass(frozen=True)
class PackedArray:
    """Container for bit-packed arrays with original shape and dtype metadata."""

    data: bytes
    shape: Tuple[int, ...]
    dtype: str

    def unpack(self) -> np.ndarray:
        return unpack_packed_array(self)


def pack_array_bits(array: np.ndarray) -> PackedArray:
    """Pack a boolean/uint8 array of 0/1 values into a byte representation."""
    if array.dtype != np.uint8:
        array = array.astype(np.uint8, copy=False)
    packed = np.packbits(array.reshape(-1), bitorder="little")
    return PackedArray(data=packed.tobytes(), shape=tuple(array.shape), dtype="uint8")


def unpack_packed_array(packed: PackedArray) -> np.ndarray:
    """Reconstruct an array from a PackedArray."""
    arr = np.frombuffer(packed.data, dtype=np.uint8)
    unpacked = np.unpackbits(arr, bitorder="little")
    required = int(np.prod(packed.shape))
    unpacked = unpacked[:required]
    return unpacked.reshape(packed.shape).astype(np.uint8)


def decode_packed_structure(obj: Any) -> Any:
    """Recursively decode PackedArray instances within nested structures."""
    if isinstance(obj, PackedArray):
        return unpack_packed_array(obj)
    if isinstance(obj, dict):
        return {k: decode_packed_structure(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [decode_packed_structure(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(decode_packed_structure(v) for v in obj)
    return obj
