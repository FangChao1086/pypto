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


def _normalize_printf_args(
    args: Sequence[int | float | Expr | bool], actual_span: Span
) -> tuple[list[Expr], list[bool]]:
    normalized_args: list[Expr] = []
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

    return normalized_args, raw_bool_args


def _validate_printf_arguments(
    format_str: str, normalized_args: Sequence[Expr], raw_bool_args: Sequence[bool], *, op_name: str
) -> None:
    specs = _scan_printf_format(format_str)
    if len(specs) != len(normalized_args):
        raise ValueError(
            f"{op_name} format expects {len(specs)} scalar arguments, but got {len(normalized_args)}"
        )

    for idx, (spec, arg, is_raw_bool) in enumerate(zip(specs, normalized_args, raw_bool_args, strict=True)):
        scalar_type = arg.type
        if not isinstance(scalar_type, ScalarType):
            raise TypeError(
                f"debug.{op_name} argument {idx} requires ScalarType input, but got {type(scalar_type).__name__}"
            )

        conversion = spec[-1]
        if conversion in {"d", "i"} and not (
            _is_signed_integer_dtype(scalar_type.dtype)
            or _is_bool_dtype(scalar_type.dtype)
            or _is_index_dtype(scalar_type.dtype)
        ):
            raise TypeError(
                f"debug.{op_name} conversion '{spec}' requires signed integer, bool, or index scalar, but got {scalar_type.dtype}"
            )
        if conversion == "u" and not (
            _is_unsigned_integer_dtype(scalar_type.dtype)
            or _is_bool_dtype(scalar_type.dtype)
            or _is_index_dtype(scalar_type.dtype)
            or is_raw_bool
        ):
            raise TypeError(
                f"debug.{op_name} conversion '{spec}' requires unsigned integer, bool, or index scalar, but got {scalar_type.dtype}"
            )
        if conversion == "x" and not (
            _is_unsigned_integer_dtype(scalar_type.dtype) or _is_index_dtype(scalar_type.dtype)
        ):
            raise TypeError(
                f"debug.{op_name} conversion '{spec}' requires unsigned integer or index scalar, but got {scalar_type.dtype}"
            )
        if conversion == "x" and is_raw_bool:
            raise TypeError(f"debug.{op_name} conversion '{spec}' does not support bool scalars")
        if conversion in _FLOAT_CONVERSIONS and scalar_type.dtype != DataType.FP32:
            raise TypeError(
                f"debug.{op_name} conversion '{spec}' requires FP32 scalar, but got {scalar_type.dtype}"
            )


def dump_tensor(
    tensor: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    shapes: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    span: Span | None = None,
) -> Call:
    """Print a tensor or tensor window for debugging.

    Supported forms:
    - Full tensor dump: ``dump_tensor(tensor)``
    - Full dumps with dynamic tensor shapes
    - Window dump with dynamic ``offsets``
    - Window dump with dynamic ``shapes``

    Backend support:
    - PTO: full dump and window dump support dynamic ``offsets``/``shapes``
    - CCE: full dump and window dump support dynamic ``offsets``/``shapes``

    Current limitation:
    - Windowed dumps still require the innermost tensor stride to be statically 1
    """
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

    for idx, shape_expr in enumerate(shapes_tuple.elements):
        if isinstance(shape_expr, ConstInt) and shape_expr.value <= 0:
            raise ValueError(
                f"debug.dump_tensor shape at axis {idx} must be positive, got {shape_expr.value}"
            )

    return _ir_core.create_op_call(
        "debug.dump_tensor", [tensor, offsets_tuple, shapes_tuple], {}, actual_span
    )


def dump_tile(
    tile: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    shapes: Sequence[int | Expr] | _ir_core.MakeTuple | None = None,
    span: Span | None = None,
) -> Call:
    """Print a tile or tile window for debugging.

    Supported forms:
    - Full tile dump: ``dump_tile(tile)``
    - Full dump of tiles with dynamic valid-shape
    - Window dump with dynamic ``offsets`` on PTO and CCE
    - Window dump with dynamic ``shapes`` on CCE

    Backend support:
    - PTO: window ``shapes`` must still be static
    - CCE: window ``shapes`` may be dynamic

    Current limitations:
    - Tile windows are currently 2D-only
    - Tile printing requires a printable ``vec`` tile; CCE window dumps handle
      this by copying the requested window into a temporary vec tile before
      printing
    """
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

    for idx, shape_expr in enumerate(shapes_tuple.elements):
        if isinstance(shape_expr, ConstInt) and shape_expr.value <= 0:
            raise ValueError(
                f"debug.dump_tile shape at axis {idx} must be positive, got {shape_expr.value}"
            )

    return _ir_core.create_op_call(
        "debug.dump_tile", [tile, offsets_tuple, shapes_tuple], {}, actual_span
    )


def printf(format_str: str, *args: int | float | Expr, span: Span | None = None) -> Call:
    """Print scalar values using a compile-time format string."""
    actual_span = _get_span_or_capture(span)
    if not isinstance(format_str, str):
        raise TypeError(f"debug.printf requires string format literal, but got {type(format_str).__name__}")

    normalized_args, raw_bool_args = _normalize_printf_args(args, actual_span)
    _validate_printf_arguments(format_str, normalized_args, raw_bool_args, op_name="printf")

    kwargs: dict[str, Any] = {"format": format_str}
    return _ir_core.create_op_call("debug.printf", normalized_args, kwargs, actual_span)


def assert_(
    condition: bool | Expr,
    format_str: str | None = None,
    *args: int | float | Expr | bool,
    condition_text: str | None = None,
    span: Span | None = None,
) -> Call:
    """Abort execution when a scalar boolean condition is false."""
    actual_span = _get_span_or_capture(span)
    if isinstance(condition, Expr):
        condition_expr = condition
    elif isinstance(condition, bool):
        condition_expr = ConstBool(condition, actual_span)
    else:
        raise TypeError(
            "debug.assert requires a scalar bool condition, "
            f"but got {type(condition).__name__}"
        )

    condition_type = condition_expr.type
    if not isinstance(condition_type, ScalarType) or condition_type.dtype != DataType.BOOL:
        raise TypeError(
            "debug.assert requires a scalar bool condition, "
            f"but got {type(condition_type).__name__}({getattr(condition_type, 'dtype', condition_type)})"
        )

    if condition_text is None:
        condition_text = "condition"
    if not isinstance(condition_text, str):
        raise TypeError(
            f"debug.assert requires string condition_text metadata, but got {type(condition_text).__name__}"
        )

    normalized_args: list[Expr] = []
    if format_str is None:
        format_value = ""
    else:
        if not isinstance(format_str, str):
            raise TypeError(f"debug.assert requires string literal format, but got {type(format_str).__name__}")
        normalized_args, raw_bool_args = _normalize_printf_args(args, actual_span)
        _validate_printf_arguments(format_str, normalized_args, raw_bool_args, op_name="assert")
        format_value = format_str

    kwargs: dict[str, Any] = {"condition_text": condition_text, "format": format_value}
    return _ir_core.create_op_call("debug.assert", [condition_expr, *normalized_args], kwargs, actual_span)


def trap(*, span: Span | None = None) -> Call:
    """Abort execution by inserting a trap."""
    actual_span = _get_span_or_capture(span)
    return _ir_core.create_op_call("debug.trap", [], {}, actual_span)


__all__ = ["assert_", "dump_tensor", "dump_tile", "printf", "trap"]
