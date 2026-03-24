# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""AST pre-scan for backward (cross-iteration) loop dependencies."""

from __future__ import annotations

import ast
from collections.abc import Callable
from typing import Any

from pypto.pypto_core.ir import PipeType

from .data_structures import BackwardDep
from .event_allocator import EventIdAllocator
from .op_metadata import _OP_TILE_ACCESS, _OP_TO_PIPE


def prescan_loop_backward_deps(
    body_stmts: list[ast.stmt],
    scope_lookup: Callable[[str], Any],
    event_allocator: EventIdAllocator | None = None,
    loop_depth: int = 0,
) -> list[BackwardDep]:
    """Pre-scan a loop body AST to detect backward (cross-iteration) deps.

    Walks the AST to find ``plm.{op}(...)`` calls, determines their pipe
    type, extracts tile argument names, and tracks the first/last pipe per
    tile.  If they differ, a backward dependency exists.

    Args:
        body_stmts: The list of AST statement nodes in the loop body.
        scope_lookup: Callable that maps a variable name to its ``ir.Var``
            (or ``None``).  Used to check whether an arg is a tile.
        event_allocator: Optional allocator for backward event IDs.
        loop_depth: Current loop nesting depth.

    Returns:
        List of :class:`BackwardDep` for tiles whose first and last
        accessing pipelines differ.
    """
    # tile_name → (first_pipe, last_pipe)
    tile_pipe_map: dict[str, tuple[PipeType, PipeType]] = {}
    # tiles created inside the loop body via make_tile
    local_tile_names: set[str] = set()

    def _scan_stmts(stmts: list[ast.stmt]) -> None:
        for stmt in stmts:
            _scan_stmt(stmt)

    def _scan_stmt(stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Assign):
            _scan_assign(stmt)
        elif isinstance(stmt, ast.Expr):
            _scan_expr_stmt(stmt)
        elif isinstance(stmt, ast.For):
            _scan_stmts(stmt.body)
        elif isinstance(stmt, ast.If):
            _scan_stmts(stmt.body)
            if stmt.orelse:
                _scan_stmts(stmt.orelse)
        elif isinstance(stmt, ast.With):
            _scan_stmts(stmt.body)

    def _scan_assign(stmt: ast.Assign) -> None:
        if not isinstance(stmt.value, ast.Call):
            return
        op_name = _extract_plm_op_name(stmt.value)
        if op_name is None:
            return
        if op_name == "make_tile":
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    local_tile_names.add(target.id)
            return
        _process_op(op_name, stmt.value)

    def _scan_expr_stmt(stmt: ast.Expr) -> None:
        if not isinstance(stmt.value, ast.Call):
            return
        op_name = _extract_plm_op_name(stmt.value)
        if op_name is not None and op_name != "make_tile":
            _process_op(op_name, stmt.value)

    def _extract_plm_op_name(call: ast.Call) -> str | None:
        func = call.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "plm"
        ):
            return func.attr
        return None

    def _process_op(op_name: str, call: ast.Call) -> None:
        pipe = _OP_TO_PIPE.get(op_name)
        if pipe is None:
            if op_name == "move":
                pipe = PipeType.V  # conservative default for pre-scan
            elif op_name in ("store", "store_tile"):
                pipe = PipeType.MTE3  # conservative default (Vec store)
            else:
                return
        access = _OP_TILE_ACCESS.get(op_name)
        if access is None:
            return
        all_indices = set(access.read_indices) | set(access.write_indices)
        for idx in all_indices:
            if idx < len(call.args):
                name = _extract_tile_name(call.args[idx])
                if name is not None and _is_tile(name):
                    if name in tile_pipe_map:
                        first_pipe, _ = tile_pipe_map[name]
                        tile_pipe_map[name] = (first_pipe, pipe)
                    else:
                        tile_pipe_map[name] = (pipe, pipe)

    def _extract_tile_name(arg: ast.expr) -> str | None:
        if isinstance(arg, ast.Name):
            return arg.id
        return None

    def _is_tile(name: str) -> bool:
        if name in local_tile_names:
            return True
        var = scope_lookup(name)
        if var is None:
            return False
        var_type = getattr(var, "type", None)
        if var_type is None:
            return False
        # Use isinstance when the ir module is available, fall back to name check
        try:
            from pypto.pypto_core import ir as _ir
            return isinstance(var_type, _ir.TileType)
        except (ImportError, AttributeError):
            return type(var_type).__name__ == "TileType"

    _scan_stmts(body_stmts)

    deps = []
    for name, (first, last) in tile_pipe_map.items():
        if first != last:
            eid = 0
            if event_allocator is not None:
                eid = event_allocator.backward_event_id(last, first, loop_depth)
            deps.append(BackwardDep(
                first_pipe=first, last_pipe=last, tile_name=name,
                event_id=eid, loop_depth=loop_depth,
            ))
    return deps
