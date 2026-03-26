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
    tile_tuple_registry: dict[str, list[str]] | None = None,
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
        tile_tuple_registry: Optional mapping of tuple variable names to their
            constituent tile names (for double-buffer subscript resolution).

    Returns:
        List of :class:`BackwardDep` for tiles whose first and last
        accessing pipelines differ.
    """
    _tile_tuples = tile_tuple_registry or {}

    # tile_name → (first_pipe, last_pipe)
    tile_pipe_map: dict[str, tuple[PipeType, PipeType]] = {}
    # tiles created inside the loop body via make_tile
    local_tile_names: set[str] = set()
    # tiles that were accessed via tuple subscript (DB pattern)
    tiles_from_tuples: set[str] = set()
    # max tuple size seen (for n_slots)
    max_tuple_size: int = 1

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

    def _resolve_prescan_move_pipe(call: ast.Call) -> PipeType:
        """Resolve move pipe by examining tile arg memory spaces via scope_lookup."""
        from .op_metadata import get_move_pipe

        target_ms = _get_tile_arg_ms(call.args[0]) if len(call.args) >= 1 else None
        src_ms = _get_tile_arg_ms(call.args[1]) if len(call.args) >= 2 else None
        return get_move_pipe(src_ms, target_ms)

    def _resolve_prescan_store_pipe(call: ast.Call) -> PipeType:
        """Resolve store/store_tile pipe by examining source tile memory space.

        DSL convention: ``plm.store(tensor, tile, ...)`` / ``plm.store_tile(tensor, tile, ...)``.
        arg[1] is the source tile.  Acc → FIX, Vec → MTE3.
        """
        from .op_metadata import get_store_pipe

        src_ms = _get_tile_arg_ms(call.args[1]) if len(call.args) >= 2 else None
        return get_store_pipe(src_ms)

    def _get_tile_arg_ms(arg: ast.expr) -> "MemorySpace | None":
        """Get memory space of a tile argument (Name or Subscript)."""
        tile_name: str | None = None
        if isinstance(arg, ast.Name):
            tile_name = arg.id
        elif isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name):
            tuple_name = arg.value.id
            if tuple_name in _tile_tuples:
                tile_name = _tile_tuples[tuple_name][0]
        if tile_name is None:
            return None
        var = scope_lookup(tile_name)
        if var is None:
            return None
        var_type = getattr(var, "type", None)
        if var_type is None:
            return None
        memref = getattr(var_type, "memref", None)
        if memref is not None:
            return getattr(memref, "memory_space_", None)
        return None

    def _process_op(op_name: str, call: ast.Call) -> None:
        pipe = _OP_TO_PIPE.get(op_name)
        if pipe is None:
            if op_name == "move":
                pipe = _resolve_prescan_move_pipe(call)
            elif op_name in ("store", "store_tile"):
                pipe = _resolve_prescan_store_pipe(call)
            else:
                return
        access = _OP_TILE_ACCESS.get(op_name)
        if access is None:
            return
        all_indices = set(access.read_indices) | set(access.write_indices)
        for idx in all_indices:
            if idx < len(call.args):
                names = _extract_tile_names(call.args[idx])
                for name in names:
                    if _is_tile(name):
                        if name in tile_pipe_map:
                            first_pipe, _ = tile_pipe_map[name]
                            tile_pipe_map[name] = (first_pipe, pipe)
                        else:
                            tile_pipe_map[name] = (pipe, pipe)

    def _extract_tile_names(arg: ast.expr) -> list[str]:
        """Extract tile name(s) from an AST expression.

        Returns a list of names: single-element for ``ast.Name``,
        all tuple members for ``tile_buf[idx]`` subscripts.
        """
        nonlocal max_tuple_size
        if isinstance(arg, ast.Name):
            return [arg.id]
        # DB: tile_buf[buf_idx] → expand to all tiles in the tuple
        if isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name):
            tuple_name = arg.value.id
            if tuple_name in _tile_tuples:
                names = list(_tile_tuples[tuple_name])
                tiles_from_tuples.update(names)
                max_tuple_size = max(max_tuple_size, len(names))
                return names
        return []

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

    # Collect backward deps, deduplicated by (first_pipe, last_pipe).
    # Multiple tiles with the same pipe pair only need one sync pair,
    # because hardware sync is per-pipe, not per-tile.
    # If any contributing tile came from a tuple subscript, mark n_slots
    # so backward sync can use per-slot event IDs for DB pipelining.
    seen_pipe_pairs: set[tuple[PipeType, PipeType]] = set()
    deps = []
    for name, (first, last) in tile_pipe_map.items():
        if first != last:
            pipe_key = (first, last)
            if pipe_key in seen_pipe_pairs:
                continue
            seen_pipe_pairs.add(pipe_key)
            n_slots = max_tuple_size if name in tiles_from_tuples else 1
            eid = 0
            if event_allocator is not None:
                eid = event_allocator.backward_event_id(last, first, loop_depth, n_slots=n_slots)
            deps.append(BackwardDep(
                first_pipe=first, last_pipe=last, tile_name=name,
                event_id=eid, loop_depth=loop_depth, n_slots=n_slots,
            ))

    return deps


def _collapse_transitive_backward_deps(
    deps: list[BackwardDep],
    event_allocator: EventIdAllocator | None,
    loop_depth: int,
) -> list[BackwardDep]:
    """Collapse transitive backward deps within the same data-flow chain.

    Given (first=MTE2, last=MTE1) and (first=MTE1, last=M), the pipeline
    chain for a single tile is MTE2→MTE1→M.  Waiting for M→MTE2 subsumes
    MTE1→MTE2.  Result: single dep (first=MTE2, last=M).

    Only merges when dep_a.last_pipe == dep_b.first_pipe, meaning the
    intermediate pipe connects the same data flow (e.g., load→move→matmul
    on the same tile buffers).  Does NOT merge unrelated deps like
    FIX→M (tile_c store) with M→MTE2 (tile_a load), because they protect
    different tiles.
    """
    if len(deps) <= 1:
        return deps

    # Try to find a single chain: dep whose last_pipe matches another's first_pipe.
    # Only merge if exactly one pair connects (simple linear chain).
    by_last: dict[PipeType, int] = {}  # last_pipe → index
    for i, d in enumerate(deps):
        by_last[d.last_pipe] = i

    # Walk from each dep and see if its first_pipe is another dep's last_pipe
    subsumed: set[int] = set()
    replacements: dict[int, BackwardDep] = {}  # index → merged dep

    for i, dep_a in enumerate(deps):
        if i in subsumed:
            continue
        # dep_a: first=A, last=B.  Is there dep_b with last=A?
        # If so, dep_b: first=X, last=A means chain X→A→B, merge to first=X, last=B? No.
        # We want: dep_a.last_pipe == dep_b.first_pipe
        # dep_a: first=MTE2, last=MTE1.  dep_b: first=MTE1, last=M.
        # dep_a.last_pipe=MTE1 == dep_b.first_pipe=MTE1 → merge to first=MTE2, last=M.
        for j, dep_b in enumerate(deps):
            if j in subsumed or j == i:
                continue
            if dep_a.last_pipe == dep_b.first_pipe:
                # Chain: dep_a.first → dep_a.last=dep_b.first → dep_b.last
                # Merged: first=dep_a.first, last=dep_b.last
                n_slots = max(dep_a.n_slots, dep_b.n_slots)
                eid = 0
                if event_allocator is not None:
                    eid = event_allocator.backward_event_id(
                        dep_b.last_pipe, dep_a.first_pipe, loop_depth, n_slots=n_slots,
                    )
                merged = BackwardDep(
                    first_pipe=dep_a.first_pipe, last_pipe=dep_b.last_pipe,
                    tile_name=dep_a.tile_name, event_id=eid,
                    loop_depth=loop_depth, n_slots=n_slots,
                )
                subsumed.add(i)
                subsumed.add(j)
                replacements[i] = merged
                break  # only one merge per dep

    if not subsumed:
        return deps

    result = []
    for i, dep in enumerate(deps):
        if i in replacements:
            result.append(replacements[i])
        elif i not in subsumed:
            result.append(dep)
    return result
