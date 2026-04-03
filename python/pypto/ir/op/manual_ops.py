# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Manual (non-SSA) IR operations for PyPTO.

IR-level functions for manual ops with special-case argument handling.
Called from the parser via:
    if hasattr(ir_op.manual, op_name):
        return getattr(ir_op.manual, op_name)(*args, **kwargs, span=span)
"""

from collections.abc import Sequence

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Call, Expr, Span, ConstInt

from ..utils import _get_span_or_capture, _normalize_expr, _to_make_tuple


def _compute_absolute_offsets(
    tile_offsets: _ir_core.MakeTuple,
    tile_shape: list,
    tile_dims: list[int],
    span: "Span",
) -> _ir_core.MakeTuple:
    """Convert tile-relative offsets to absolute offsets.

    For each dimension i in tile_offsets:
    - If i is in tile_dims: absolute_offset = tile_offset * tile_shape[tile_idx]
    - Otherwise: absolute_offset = tile_offset (used directly)

    Args:
        tile_offsets: MakeTuple of per-dimension tile-relative offsets.
        tile_shape: Tile shape dimensions (list of Expr).
        tile_dims: Which tensor dimensions correspond to tile dimensions.
        span: Source span for created expressions.

    Returns:
        MakeTuple of absolute offsets.
    """
    offsets = []
    for i, tile_offset in enumerate(tile_offsets.elements):
        if i in tile_dims:
            tile_idx = tile_dims.index(i)
            shape = tile_shape[tile_idx]
            if isinstance(tile_offset, ConstInt) and isinstance(shape, ConstInt):
                offset = ConstInt(tile_offset.value * shape.value, DataType.INT64, span)
            else:
                offset = _ir_core.Mul(tile_offset, shape, DataType.INT64, span)
            offsets.append(offset)
        else:
            offsets.append(tile_offset)
    return _ir_core.MakeTuple(offsets, span)


def load(
    out: Expr,
    tensor: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    span: Span | None = None,
    layout: str | None = None,
) -> Call:
    """Build manual.load IR call.

    Args:
        out: Pre-allocated destination tile.
        tensor: Source tensor expression.
        offsets: Offsets tuple or sequence (absolute offsets).
        span: Optional source span.
        layout: Tensor memory layout, "ND" (row-major) or "DN" (column-major).

    Returns:
        Call expression for manual.load.
    """
    actual_span = _get_span_or_capture(span)
    offsets_tuple = _to_make_tuple(offsets, actual_span)
    kwargs: dict = {}
    if layout is not None:
        kwargs["layout"] = layout
    return _ir_core.create_op_call(
        "manual.load", [tensor, offsets_tuple, out], kwargs, actual_span
    )


def move(
    out: Expr,
    src: Expr,
    span: Span | None = None,
    acc_to_vec_mode: str | None = None,
) -> Call:
    """Build manual.move IR call.

    Args:
        out: Pre-allocated destination tile.
        src: Source tile expression.
        span: Optional source span.
        acc_to_vec_mode: AccToVecMode string for Acc->Vec transfers.

    Returns:
        Call expression for manual.move.
    """
    actual_span = _get_span_or_capture(span)
    kwargs: dict = {}
    if acc_to_vec_mode is not None:
        kwargs["acc_to_vec_mode"] = acc_to_vec_mode
    return _ir_core.create_op_call(
        "manual.move", [src, out], kwargs, actual_span
    )


def insert(
    out: Expr,
    src: Expr,
    index_row: "int | Expr" = 0,
    index_col: "int | Expr" = 0,
    offset: "int | Expr | None" = None,
    span: Span | None = None,
) -> Call:
    """Build manual.insert IR call.

    Args:
        out: Pre-allocated destination tile.
        src: Source sub-tile expression.
        index_row: Row index where insertion begins.
        index_col: Column index where insertion begins.
        offset: Optional byte offset for destination tile base address.
        span: Optional source span.

    Returns:
        Call expression for manual.insert.
    """
    actual_span = _get_span_or_capture(span)
    row_expr = _normalize_expr(index_row)
    col_expr = _normalize_expr(index_col)
    if offset is not None:
        offset_expr = _normalize_expr(offset)
        return _ir_core.create_op_call(
            "manual.insert", [src, row_expr, col_expr, offset_expr, out], {}, actual_span
        )
    return _ir_core.create_op_call(
        "manual.insert", [src, row_expr, col_expr, out], {}, actual_span
    )


def store(
    out: Expr,
    tile: Expr,
    offsets: Sequence[int | Expr] | _ir_core.MakeTuple,
    span: Span | None = None,
) -> Call:
    """Build manual.store IR call.

    Args:
        out: Out tensor.
        tile: Source tile expression.
        offsets: Offsets tuple or sequence.
        span: Optional source span.

    Returns:
        Call expression for manual.store.
    """
    actual_span = _get_span_or_capture(span)
    offsets_tuple = _to_make_tuple(offsets, actual_span)
    return _ir_core.create_op_call(
        "manual.store", [tile, offsets_tuple, out], {}, actual_span,
    )


def load_tile(
    out: Expr,
    tensor: Expr,
    tile_offsets: _ir_core.MakeTuple,
    span: Span | None = None,
    layout: str | None = None,
    tile_dims: list[int] | None = None,
) -> Call:
    """Build manual.load with abs_offsets = tile_offsets * shapes for last dims.

    For N-D tensor with M-D tile (N >= M):
    - First (N-M) offsets are used directly (batch, head indices, etc.)
    - Last M offsets are multiplied by tile shape

    Args:
        out: Pre-allocated destination tile.
        tensor: Source tensor.
        tile_offsets: MakeTuple of offsets. First (N-M) are direct offsets,
            last M are tile-relative offsets.
        span: Optional source span.
        layout: Optional layout string ("dn" for column-major).
        tile_dims: Optional list of dimension indices that tile corresponds to.
            If None, defaults to last M dimensions (BNSD layout).
            For BSND layout [B, S, N, D] with tile [TS, TD], use tile_dims=[1, 3].

    Returns:
        Call expression for manual.load with absolute offsets.
    """
    actual_span = _get_span_or_capture(span)
    tensor_type = tensor.type
    tile_shape = out.type.shape
    tensor_ndim = len(tensor_type.shape)
    tile_ndim = len(tile_shape)

    if tile_dims is None:
        tile_dims = list(range(tensor_ndim - tile_ndim, tensor_ndim))

    offsets_tuple = _compute_absolute_offsets(
        tile_offsets, list(tile_shape), tile_dims, actual_span
    )
    kwargs = {}
    if layout is not None:
        kwargs["layout"] = layout
    if tile_dims != list(range(tensor_ndim - tile_ndim, tensor_ndim)):
        kwargs["tile_dims"] = ",".join(str(d) for d in tile_dims)
    return _ir_core.create_op_call(
        "manual.load", [tensor, offsets_tuple, out], kwargs, actual_span
    )


def store_tile(
    out: Expr,
    tile: Expr,
    tile_offsets: _ir_core.MakeTuple,
    span: Span | None = None,
    tile_dims: list[int] | None = None,
) -> Call:
    """Build manual.store with abs_offsets = tile_offsets * shapes for last dims.

    For N-D tensor with M-D tile (N >= M):
    - First (N-M) offsets are used directly (batch, head indices, etc.)
    - Last M offsets are multiplied by tile shape

    Args:
        output_tensor: Destination tensor.
        tile: Source tile.
        tile_offsets: MakeTuple of offsets. First (N-M) are direct offsets,
            last M are tile-relative offsets.
        valid_shapes: MakeTuple of tile shapes (used for offset computation only).
        span: Optional source span.
        tile_dims: Optional list of dimension indices that tile corresponds to.
            If None, defaults to last M dimensions (BNSD layout).
            For BSND layout [B, S, N, D] with tile [TS, TD], use tile_dims=[1, 3].

    Returns:
        Call expression for manual.store with absolute offsets.
    """
    actual_span = _get_span_or_capture(span)
    tensor_type = out.type
    tile_shape = tile.type.shape
    tensor_ndim = len(tensor_type.shape)
    tile_ndim = len(tile_shape)

    if tile_dims is None:
        tile_dims = list(range(tensor_ndim - tile_ndim, tensor_ndim))

    offsets_tuple = _compute_absolute_offsets(
        tile_offsets, list(tile_shape), tile_dims, actual_span
    )

    kwargs = {}
    if tile_dims != list(range(tensor_ndim - tile_ndim, tensor_ndim)):
        kwargs["tile_dims"] = ",".join(str(d) for d in tile_dims)
    return _ir_core.create_op_call(
        "manual.store", [tile, offsets_tuple, out], kwargs, actual_span
    )


def sync(
    tile: Expr | None = None,
    *,
    sync_type: str,
    event_id: int = 0,
    direction: str | None = None,
    pipeline: str | None = None,
    span: Span | None = None,
) -> Call:
    """Build manual.sync IR call for buffer synchronization.

    This is used for buffer lock/free operations in FlashAttention patterns.
    Supports different synchronization types and fine-grained pipeline control.

    Args:
        tile: Optional tile expression for buffer-level sync
        sync_type: Type of synchronization ("inner_core_sync", "cross_core_sync_forward", etc.)
        event_id: Event identifier for cross-core sync
        direction: Optional direction ("lock", "free", "allocate", "record")
        pipeline: Optional pipeline type for fine-grained sync
        span: Optional source span

    Returns:
        Call expression for manual.sync

    Examples:
        # Buffer lock for MTE2 pipeline
        sync(tile=buf, sync_type="inner_core_sync", direction="lock", pipeline="PIPE_MTE2")

        # Buffer free for V pipeline
        sync(tile=buf, sync_type="inner_core_sync", direction="free", pipeline="PIPE_V")
    """
    actual_span = _get_span_or_capture(span)
    kwargs = {
        "sync_type": sync_type,
        "event_id": event_id,
    }
    if direction is not None:
        kwargs["direction"] = direction
    if pipeline is not None:
        kwargs["pipeline"] = pipeline

    args = []
    if tile is not None:
        args.append(tile)

    return _ir_core.create_op_call("manual.sync", args, kwargs, actual_span)
