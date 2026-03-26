# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Kernel decorator for PyPTO Frontend.

The @kernel decorator provides a simplified API that wraps a single function
into a KernelDef, deferring AST parsing to compile time.

Usage:
    import pypto.frontend as fe

    @fe.kernel
    def vector_add(
        x: fe.Tensor[[1024], fe.FP32],
        y: fe.Tensor[[1024], fe.FP32],
    ) -> fe.Tensor[[1024], fe.FP32]:
        tile_x = fe.load(x, [0], [1024])
        tile_y = fe.load(y, [0], [1024])
        result = fe.add(tile_x, tile_y)
        out = fe.create([1024], dtype=fe.FP32)
        return fe.store(result, [0], [1024], out)

    # vector_add is now a KernelDef; parsing happens at fe.compile() time
    compiled = fe.compile(vector_add, arch="a3")
"""

import ast
import sys
import inspect
import textwrap
from typing import Any, Callable, Optional

from pypto.pypto_core import ir
from pypto.language.parser.ast_parser import ASTParser
from pypto.language.parser.decorator import (
    _calculate_col_offset,
    _parse_ast_tree,
    _find_ast_node,
    _attach_source_lines_to_error,
    _extract_function_type_from_decorator,
    KernelFunction,
)
from pypto.language.parser.diagnostics import ParserError, ParserSyntaxError


class KernelDef:
    """Lazy kernel definition — captures source/AST/closure at decoration time,
    defers AST parsing to compile time.

    Call :meth:`parse` to trigger parsing and obtain an ``ir.Program``.
    ``fe.compile()`` calls this automatically.

    Args:
        func: Original Python function.
        source_file: Path to the source file.
        source_lines: Dedented source lines for the parser.
        source_lines_raw: Raw (non-dedented) source lines for error reporting.
        line_offset: Line number offset in the original file.
        col_offset: Column indentation offset.
        func_def: AST FunctionDef node.
        closure_vars: Captured caller scope for name resolution.
        name: Optional program name.
        func_type: IR function type (Opaque, Orchestration, InCore).
        strict_ssa: Whether to enforce SSA.
        auto_sync: Whether to enable automatic sync insertion.
        meta_data: Optional metadata.
        helper_funcs: List of ``ir.Function`` from ``@pl.func`` helpers.
    """

    def __init__(
        self,
        func: Callable,
        source_file: str,
        source_lines: list[str],
        source_lines_raw: list[str],
        line_offset: int,
        col_offset: int,
        func_def: ast.FunctionDef,
        closure_vars: dict[str, Any],
        name: str | None,
        func_type: ir.FunctionType,
        strict_ssa: bool,
        auto_sync: bool,
        meta_data: Any,
        helper_funcs: list,
    ) -> None:
        self._func = func
        self._source_file = source_file
        self._source_lines = source_lines
        self._source_lines_raw = source_lines_raw
        self._line_offset = line_offset
        self._col_offset = col_offset
        self._func_def = func_def
        self._closure_vars = closure_vars
        self._name = name
        self._func_type = func_type
        self._strict_ssa = strict_ssa
        self._auto_sync = auto_sync
        self._meta_data = meta_data
        self._helper_funcs = helper_funcs

    def parse(self, npu_arch: str | None = None) -> ir.Program:
        """Parse the kernel AST and return an ``ir.Program``.

        Args:
            npu_arch: Target architecture (e.g. ``"a3"``, ``"a5"``).
                Controls same-pipeline sync behaviour when ``auto_sync=True``.
                ``"a2"``/``"a3"`` (dav-2201) require intra-pipe sync;
                ``"a5"`` (dav-3510) does not.

        Returns:
            ir.Program containing the parsed kernel function.
        """
        program_name = self._name if self._name is not None else self._func.__name__

        try:
            parser = ASTParser(
                self._source_file,
                self._source_lines,
                self._line_offset,
                self._col_offset,
                strict_ssa=self._strict_ssa,
                closure_vars=self._closure_vars,
                auto_sync=self._auto_sync,
                npu_arch=npu_arch,
            )

            try:
                ir_func = parser.parse_function(self._func_def, func_type=self._func_type)
            except ParserError:
                raise
            except Exception as e:
                raise ParserSyntaxError(
                    f"Failed to parse kernel function '{self._func.__name__}': {e}",
                    hint="Check your function definition for errors",
                ) from e

            implicit_funcs = list(parser.external_funcs.values())

            starting_line = self._line_offset + 1
            program_span = ir.Span(self._source_file, starting_line, self._col_offset)
            return ir.Program(
                self._helper_funcs + implicit_funcs + [ir_func],
                program_name,
                program_span,
            )

        except ParserError as e:
            _attach_source_lines_to_error(e, self._source_file, self._source_lines_raw)
            raise


def _call_meta_and_capture_env(meta_fn):
    """Run meta_fn() and capture its local namespace (for types etc.). Returns (return_value, env dict)."""
    env = {}

    if meta_fn is None:
        return None, env
    def trace(frame, event, arg):
        if event == "return":
            env.clear()
            env.update(frame.f_locals)
        return trace

    old_trace = sys.gettrace()
    sys.settrace(trace)
    try:
        result = meta_fn()
    finally:
        sys.settrace(old_trace)
    return result, env


def kernel(
    func: Optional[Callable] = None,
    meta_data=None,
    *,
    name: Optional[str] = None,
    type: ir.FunctionType = ir.FunctionType.Opaque,
    strict_ssa: bool = False,
    auto_sync: bool = False,
) -> "KernelDef":
    """Decorator that captures a DSL function for deferred compilation.

    The decorated function becomes a :class:`KernelDef` — a lazy definition
    that records the source code, AST, and closure at decoration time but
    defers parsing until ``fe.compile()`` is called.  This allows the target
    architecture (which controls sync behaviour) to be specified once at
    compile time.

    Args:
        func: Python function to capture.
        name: Optional program name (defaults to function name).
        type: Function type (Opaque, Orchestration, or InCore).
        strict_ssa: If True, enforce SSA (single assignment per variable).
        auto_sync: If True, enable automatic intra-pipeline synchronization
            at compile time.

    Returns:
        KernelDef that can be passed to ``fe.compile()``.

    Example:
        >>> @fe.kernel
        ... def my_kernel(x: fe.Tensor[[64], fe.FP32]) -> fe.Tensor[[64], fe.FP32]:
        ...     tile = fe.load(x, [0], [64])
        ...     result = fe.add(tile, tile)
        ...     return fe.store(result, [0], [64], x)
        >>> isinstance(my_kernel, KernelDef)
        True

        >>> @fe.kernel(auto_sync=True)
        ... def auto_sync_kernel(x: fe.Tensor[[64], fe.FP32]) -> fe.Tensor[[64], fe.FP32]:
        ...     tile = fe.load(x, [0], [64])
        ...     result = fe.add(tile, tile)
        ...     return fe.store(result, [0], [64], x)
    """

    # Capture caller's scope so the parser can resolve names like `pl`, `plm`, etc.
    # Must be captured here (not inside _decorator) to get the correct call-site frame.
    caller_frame = sys._getframe(1)
    closure_vars = {**caller_frame.f_globals, **caller_frame.f_locals}

    def _decorator(f: Callable) -> KernelDef:
        # Get source code and file information
        source_file = inspect.getfile(f)
        source_lines_raw, starting_line = inspect.getsourcelines(f)
        source_code = "".join(source_lines_raw)

        # Calculate indentation offset before dedenting
        col_offset = _calculate_col_offset(source_lines_raw)

        # Remove leading indentation so ast.parse() can parse it
        source_code = textwrap.dedent(source_code)
        source_lines = source_code.split("\n")

        # Calculate line offset
        line_offset = starting_line - 1

        try:
            tree = _parse_ast_tree(source_code, "function")
            func_def = _find_ast_node(tree, ast.FunctionDef, f.__name__, "function")
        except ParserError as e:
            _attach_source_lines_to_error(e, source_file, source_lines_raw)
            raise

        # Collect @pl.func helper functions from closure
        helper_funcs = [
            val.ir_function
            for val in closure_vars.values()
            if isinstance(val, KernelFunction)
        ]

        return KernelDef(
            func=f,
            source_file=source_file,
            source_lines=source_lines,
            source_lines_raw=source_lines_raw,
            line_offset=line_offset,
            col_offset=col_offset,
            func_def=func_def,
            closure_vars=closure_vars,
            name=name,
            func_type=type,
            strict_ssa=strict_ssa,
            auto_sync=auto_sync,
            meta_data=meta_data,
            helper_funcs=helper_funcs,
        )

    # Support both @fe.kernel and @fe.kernel(name=..., type=...)
    if func is None:
        return _decorator  # type: ignore[return-value]
    else:
        return _decorator(func)


__all__ = ["kernel", "KernelDef"]
