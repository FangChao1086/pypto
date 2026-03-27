# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Debug operations for PyPTO IR."""

from collections.abc import Sequence
from typing import Any

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, ConstBool, ConstInt, Expr, ScalarType, Span, TensorType, TileType

from ..utils import _get_span_or_capture, _normalize_expr, _to_make_tuple

_PRINTF_FLAGS = set("-+ #0")
_INTEGER_CONVERSIONS = {"d", "i", "u", "x"}
_FLOAT_CONVERSIONS = {"f"}
_SUPPORTED_CONVERSIONS = _INTEGER_CONVERSIONS | _FLOAT_CONVERSIONS
_INTEGER_DTYPES = tuple(
    dtype
    for dtype_name in (
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "UINT8",
        "UINT16",
        "UINT32",
        "UINT64",
    )
    if hasattr(DataType, dtype_name)
    for dtype in (getattr(DataType, dtype_name),)
)

_SIGNED_INTEGER_DTYPES = tuple(
    dtype
    for dtype_name in ("INT8", "INT16", "INT32", "INT64")
    if hasattr(DataType, dtype_name)
    for dtype in (getattr(DataType, dtype_name),)
)

_UNSIGNED_INTEGER_DTYPES = tuple(
    dtype
    for dtype_name in ("UINT8", "UINT16", "UINT32", "UINT64")
    if hasattr(DataType, dtype_name)
    for dtype in (getattr(DataType, dtype_name),)
)


def _is_integer_dtype(dtype: DataType) -> bool:
    if hasattr(dtype, "IsInt"):
        return dtype.IsInt()
    return dtype in _INTEGER_DTYPES


def _is_signed_integer_dtype(dtype: DataType) -> bool:
    if hasattr(dtype, "IsSignedInt"):
        return dtype.IsSignedInt()
    return dtype in _SIGNED_INTEGER_DTYPES


def _is_unsigned_integer_dtype(dtype: DataType) -> bool:
    if hasattr(dtype, "IsUnsignedInt"):
        return dtype.IsUnsignedInt()
    return dtype in _UNSIGNED_INTEGER_DTYPES


def _is_index_dtype(dtype: DataType) -> bool:
    return dtype == DataType.INDEX


def _is_bool_dtype(dtype: DataType) -> bool:
    return dtype == DataType.BOOL


def _scan_printf_format(format_str: str) -> list[str]:
    specs: list[str] = []
    i = 0
    while i < len(format_str):
        if format_str[i] != "%":
            i += 1
            continue
        if i + 1 < len(format_str) and format_str[i + 1] == "%":
            raise ValueError("printf does not support literal '%%'")

        j = i + 1
        while j < len(format_str) and format_str[j] in _PRINTF_FLAGS:
            j += 1
        while j < len(format_str) and format_str[j].isdigit():
            j += 1
        if j < len(format_str) and format_str[j] == ".":
            j += 1
            if j >= len(format_str) or not format_str[j].isdigit():
                raise ValueError("printf precision must be followed by digits")
            while j < len(format_str) and format_str[j].isdigit():
                j += 1
        if j >= len(format_str):
            raise ValueError("printf format string ends with an incomplete conversion")

        conversion = format_str[j]
        if conversion not in _SUPPORTED_CONVERSIONS:
            raise ValueError(f"printf does not support conversion '%{conversion}'")

        specs.append(format_str[i : j + 1])
        i = j + 1

    return specs


def dump_tensor_(
    tensor: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    shapes: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    span: Span | None = None,
) -> Call:
    """Print a tensor or tensor window for debugging."""
    actual_span = _get_span_or_capture(span)
    tensor_type = tensor.type
    if not isinstance(tensor_type, TensorType):
        raise TypeError(
            f"debug.dump_tensor requires TensorType input, but got {type(tensor_type).__name__}"
        )

    if (offsets is None) != (shapes is None):
        raise ValueError("debug.dump_tensor offsets and shapes must be provided together")

    rank = len(tensor_type.shape)
    if offsets is None and shapes is None:
        offsets_tuple = _to_make_tuple([0] * rank, actual_span)
        shapes_tuple = _to_make_tuple(tensor_type.shape, actual_span)
    else:
        if tensor_type.tensor_view is not None and tensor_type.tensor_view.stride:
            last_stride = tensor_type.tensor_view.stride[-1]
            if not isinstance(last_stride, ConstInt):
                raise NotImplementedError(
                    "debug.dump_tensor windowed mode requires the innermost stride to be statically 1"
                )
            if last_stride.value != 1:
                raise ValueError(
                    "debug.dump_tensor windowed mode requires innermost stride == 1, "
                    f"got {last_stride.value}"
                )
        offsets_tuple = _to_make_tuple(offsets, actual_span)
        shapes_tuple = _to_make_tuple(shapes, actual_span)

    if len(offsets_tuple.elements) != rank or len(shapes_tuple.elements) != rank:
        raise ValueError(
            f"debug.dump_tensor offsets/shapes must match tensor rank {rank}, got "
            f"{len(offsets_tuple.elements)} offsets and {len(shapes_tuple.elements)} shapes"
        )

    for idx, offset_expr in enumerate(offsets_tuple.elements):
        if not isinstance(offset_expr, ConstInt):
            raise NotImplementedError(
                "debug.dump_tensor currently only supports static offsets, "
                f"got dynamic offset at axis {idx}"
            )

    for idx, shape_expr in enumerate(shapes_tuple.elements):
        if not isinstance(shape_expr, ConstInt):
            raise NotImplementedError(
                "debug.dump_tensor currently only supports static shapes, "
                f"got dynamic shape at axis {idx}"
            )
        if shape_expr.value <= 0:
            raise ValueError(
                f"debug.dump_tensor shape at axis {idx} must be positive, got {shape_expr.value}"
            )

    return _ir_core.create_op_call(
        "debug.dump_tensor", [tensor, offsets_tuple, shapes_tuple], {}, actual_span
    )


def dump_tile_(
    tile: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    shapes: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    span: Span | None = None,
) -> Call:
    """Print a tile or tile window for debugging."""
    actual_span = _get_span_or_capture(span)
    tile_type = tile.type
    if not isinstance(tile_type, TileType):
        raise TypeError(f"debug.dump_tile requires TileType input, but got {type(tile_type).__name__}")
    if (offsets is None) != (shapes is None):
        raise ValueError("debug.dump_tile offsets and shapes must be provided together")

    rank = len(tile_type.shape)
    if offsets is None and shapes is None:
        return _ir_core.create_op_call("debug.dump_tile", [tile], {}, actual_span)

    offsets_tuple = _to_make_tuple(offsets, actual_span)
    shapes_tuple = _to_make_tuple(shapes, actual_span)
    if len(offsets_tuple.elements) != rank or len(shapes_tuple.elements) != rank:
        raise ValueError(
            f"debug.dump_tile offsets/shapes must match tile rank {rank}, got "
            f"{len(offsets_tuple.elements)} offsets and {len(shapes_tuple.elements)} shapes"
        )

    for idx, offset_expr in enumerate(offsets_tuple.elements):
        if not isinstance(offset_expr, ConstInt):
            raise NotImplementedError(
                "debug.dump_tile currently only supports static offsets, "
                f"got dynamic offset at axis {idx}"
            )

    for idx, shape_expr in enumerate(shapes_tuple.elements):
        if not isinstance(shape_expr, ConstInt):
            raise NotImplementedError(
                "debug.dump_tile currently only supports static shapes, "
                f"got dynamic shape at axis {idx}"
            )
        if shape_expr.value <= 0:
            raise ValueError(
                f"debug.dump_tile shape at axis {idx} must be positive, got {shape_expr.value}"
            )

    return _ir_core.create_op_call(
        "debug.dump_tile", [tile, offsets_tuple, shapes_tuple], {}, actual_span
    )


def printf_(format_str: str, *args: int | float | Expr, span: Span | None = None) -> Call:
    """Print scalar values using a compile-time format string."""
    actual_span = _get_span_or_capture(span)
    if not isinstance(format_str, str):
        raise TypeError(f"debug.printf requires string format literal, but got {type(format_str).__name__}")

    normalized_args = []
    raw_bool_args: list[bool] = []
    for arg in args:
        if isinstance(arg, Expr):
            normalized_args.append(arg)
            raw_bool_args.append(False)
        elif isinstance(arg, bool):
            normalized_args.append(ConstBool(arg, actual_span))
            raw_bool_args.append(True)
        else:
            normalized_args.append(
                _normalize_expr(arg, actual_span, int_dtype=DataType.INT64, float_dtype=DataType.FP32)
            )
            raw_bool_args.append(False)

    specs = _scan_printf_format(format_str)
    if len(specs) != len(normalized_args):
        raise ValueError(
            f"printf format expects {len(specs)} scalar arguments, but got {len(normalized_args)}"
        )

    for idx, (spec, arg, is_raw_bool) in enumerate(zip(specs, normalized_args, raw_bool_args, strict=True)):
        scalar_type = arg.type
        if not isinstance(scalar_type, ScalarType):
            raise TypeError(
                f"debug.printf argument {idx} requires ScalarType input, but got {type(scalar_type).__name__}"
            )

        conversion = spec[-1]
        if conversion in {"d", "i"} and not (
            _is_signed_integer_dtype(scalar_type.dtype)
            or _is_bool_dtype(scalar_type.dtype)
            or _is_index_dtype(scalar_type.dtype)
        ):
            raise TypeError(
                f"debug.printf conversion '{spec}' requires signed integer, bool, or index scalar, but got {scalar_type.dtype}"
            )
        if conversion == "u" and not (
            _is_unsigned_integer_dtype(scalar_type.dtype)
            or _is_bool_dtype(scalar_type.dtype)
            or _is_index_dtype(scalar_type.dtype)
            or is_raw_bool
        ):
            raise TypeError(
                f"debug.printf conversion '{spec}' requires unsigned integer, bool, or index scalar, but got {scalar_type.dtype}"
            )
        if conversion == "x" and not (
            _is_unsigned_integer_dtype(scalar_type.dtype) or _is_index_dtype(scalar_type.dtype)
        ):
            raise TypeError(
                f"debug.printf conversion '{spec}' requires unsigned integer or index scalar, but got {scalar_type.dtype}"
            )
        if conversion == "x" and is_raw_bool:
            raise TypeError(f"debug.printf conversion '{spec}' does not support bool scalars")
        if conversion in _FLOAT_CONVERSIONS and scalar_type.dtype != DataType.FP32:
            raise TypeError(
                f"debug.printf conversion '{spec}' requires FP32 scalar, but got {scalar_type.dtype}"
            )

    kwargs: dict[str, Any] = {"format": format_str}
    return _ir_core.create_op_call("debug.printf", normalized_args, kwargs, actual_span)


__all__ = ["dump_tensor_", "dump_tile_", "printf_"]
