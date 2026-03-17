# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tiling class utilities for PyPTO Language DSL."""

from dataclasses import dataclass
from dataclasses import is_dataclass

from pypto.pypto_core import DataType

_PYTHON_TYPE_TO_DTYPE: dict[type, DataType] = {
    int: DataType.INDEX,
    float: DataType.FP32,
    bool: DataType.BOOL,
}

_DTYPE_DEFAULT: dict[type, int | float | bool] = {int: 0, float: 0.0, bool: False}


class ArrayInstance:
    """Runtime fixed-size array value for tiling class fields.

    Construct via the Array[T, N] alias:
        Array[int, 60]([0] * 60)   # from a list
        Array[int, 3](0, 1, 2)     # variadic args
        Array[int, 4]()            # zero-initialized
    """

    def __init__(self, alias: "_ArrayAlias", values: list) -> None:
        if len(values) != alias.size:
            raise ValueError(
                f"Array[{alias.dtype.__name__}, {alias.size}] requires exactly "
                f"{alias.size} elements, got {len(values)}"
            )
        self._values = list(values)
        self._alias = alias

    def __getitem__(self, idx: int):
        return self._values[idx]

    def __setitem__(self, idx: int, val) -> None:
        self._values[idx] = val

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __repr__(self) -> str:
        return f"Array[{self._alias.dtype.__name__}, {self._alias.size}]{self._values!r}"


@dataclass(frozen=True)
class _ArrayAlias:
    """Internal: stores dtype and size for an Array[T, N] annotation."""

    dtype: type  # int, float, or bool
    size: int

    def __call__(self, *args) -> ArrayInstance:
        """Create an ArrayInstance with the given values.

        Args:
            *args: No args (zero-initialized), one iterable arg, or N scalar args.

        Returns:
            ArrayInstance with validated size.
        """
        if len(args) == 0:
            values = [_DTYPE_DEFAULT[self.dtype]] * self.size
        elif len(args) == 1 and hasattr(args[0], "__iter__"):
            values = list(args[0])
        else:
            values = list(args)
        return ArrayInstance(self, values)


class Array:
    """Fixed-length homogeneous array type for tiling class fields.

    Usage:
        offsets: Array[int, 4]    # 4 × INT32 params
        scales: Array[float, 2]   # 2 × FP32 params
    """

    def __class_getitem__(cls, args: tuple[type, int]) -> _ArrayAlias:
        dtype, size = args
        if dtype not in _PYTHON_TYPE_TO_DTYPE:
            raise TypeError(f"Array element type must be int, float, or bool, got {dtype!r}")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"Array size must be a positive integer, got {size!r}")
        return _ArrayAlias(dtype=dtype, size=size)


@dataclass(frozen=True)
class ScalarFieldInfo:
    """Field info for a scalar tiling field (int, float, or bool)."""

    dtype: DataType


@dataclass(frozen=True)
class ArrayFieldInfo:
    """Field info for a fixed-length array tiling field (Array[T, N])."""

    dtype: DataType
    size: int


FieldInfo = ScalarFieldInfo | ArrayFieldInfo


def _is_valid_field_annotation(ann: object) -> bool:
    return ann in _PYTHON_TYPE_TO_DTYPE or isinstance(ann, _ArrayAlias)


def is_tiling_class(cls: object) -> bool:
    """Return True if cls is a user-defined tiling class.

    A tiling class is a dataclass with at least one field,
    all annotated as int, float, bool, or Array[T, N].

    Args:
        cls: Object to check

    Returns:
        True if cls is a valid tiling class
    """
    if not isinstance(cls, type):
        return False
    if not is_dataclass(cls):
        return False
    annotations = getattr(cls, "__annotations__", {})
    if not annotations:
        return False
    return all(_is_valid_field_annotation(v) for v in annotations.values())


def get_tiling_fields(cls: type) -> dict[str, FieldInfo]:
    """Return ordered {field_name: FieldInfo} for a validated tiling class.

    Args:
        cls: A tiling class (validated by is_tiling_class)

    Returns:
        Ordered dict mapping field names to their FieldInfo (ScalarFieldInfo or ArrayFieldInfo)

    Raises:
        ValueError: If cls is not a valid tiling class
    """
    if not is_tiling_class(cls):
        raise ValueError(
            f"Not a valid tiling class: {cls!r}. All fields must be annotated as "
            "int, float, bool, or Array[T, N]."
        )
    result: dict[str, FieldInfo] = {}
    for name, ann in cls.__annotations__.items():
        if ann in _PYTHON_TYPE_TO_DTYPE:
            result[name] = ScalarFieldInfo(dtype=_PYTHON_TYPE_TO_DTYPE[ann])
        else:  # _ArrayAlias
            result[name] = ArrayFieldInfo(dtype=_PYTHON_TYPE_TO_DTYPE[ann.dtype], size=ann.size)
    return result


__all__ = [
    "is_tiling_class",
    "get_tiling_fields",
    "Array",
    "ArrayInstance",
    "ScalarFieldInfo",
    "ArrayFieldInfo",
    "FieldInfo",
]
