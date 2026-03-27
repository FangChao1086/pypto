# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Sync emission helpers: emit sync_src / sync_dst IR ops via the builder."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .data_structures import BackwardDep, SyncPair

if TYPE_CHECKING:
    from pypto.ir import IRBuilder
    from pypto.pypto_core import ir


def emit_sync_pair(builder: IRBuilder, pair: SyncPair, span: ir.Span) -> None:
    """Emit sync IR for a forward dependency.

    V→V pairs on a2/a3 use ``bar_v`` (vector pipeline barrier) because
    hardware does not support ``set_flag[V, V]``.  All other cross-pipe
    pairs emit the standard ``sync_src`` + ``sync_dst`` pair.
    """
    from pypto.pypto_core.ir import PipeType
    from pypto.ir.op import system_ops

    if pair.set_pipe == PipeType.V and pair.wait_pipe == PipeType.V:
        # a2/a3: V pipeline does not guarantee intra-pipe ordering;
        # bar_v drains all pending V operations at this point.
        builder.eval_stmt(system_ops.bar_v(span=span), span)
        return

    src_expr = system_ops.sync_src(
        set_pipe=pair.set_pipe, wait_pipe=pair.wait_pipe,
        event_id=pair.event_id, span=span,
    )
    builder.eval_stmt(src_expr, span)

    dst_expr = system_ops.sync_dst(
        set_pipe=pair.set_pipe, wait_pipe=pair.wait_pipe,
        event_id=pair.event_id, span=span,
    )
    builder.eval_stmt(dst_expr, span)


def emit_backward_sync_src(builder: IRBuilder, dep: BackwardDep, span: ir.Span) -> None:
    """Emit ``sync_src`` (set_flag) for a backward dependency."""
    from pypto.ir.op import system_ops

    expr = system_ops.sync_src(
        set_pipe=dep.last_pipe, wait_pipe=dep.first_pipe,
        event_id=dep.event_id, span=span,
    )
    builder.eval_stmt(expr, span)


def emit_backward_sync_dst(builder: IRBuilder, dep: BackwardDep, span: ir.Span) -> None:
    """Emit ``sync_dst`` (wait_flag) for a backward dependency."""
    from pypto.ir.op import system_ops

    expr = system_ops.sync_dst(
        set_pipe=dep.last_pipe, wait_pipe=dep.first_pipe,
        event_id=dep.event_id, span=span,
    )
    builder.eval_stmt(expr, span)
