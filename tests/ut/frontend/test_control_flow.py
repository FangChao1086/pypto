# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""End-to-end MLIR emission tests for control-flow constructs.

Each test compiles a @fe.kernel to PTO MLIR and checks that the emitted
MLIR contains the expected scf.* operations.

Coverage
--------
- Plain if/else (no loop)         → scf.if (side-effect only, no result)
- For loop (basic)                → scf.for, no inner scf.if
- For loop + continue             → scf.for + inner scf.if (inverted continue guard)
- For loop + break                → scf.for + scf.if + i1 iter_arg (_can_continue)
- While loop (basic)              → scf.while + scf.condition, no inner scf.if
- While loop + continue           → scf.while + inner scf.if (continue guard)
- While loop + break              → scf.while + scf.if + i1 iter_arg (_can_continue)
"""

import pytest

import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm
from pypto import backend
from pypto.backend import BackendType
from pypto.pypto_core.codegen import PTOCodegen

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _compile_to_mlir(kernel_def_or_prog) -> str:
    """Compile a KernelDef or ir.Program to PTO MLIR without running external tools."""
    from pypto.frontend.kernel import KernelDef
    if isinstance(kernel_def_or_prog, KernelDef):
        prog = kernel_def_or_prog.parse()
    else:
        prog = kernel_def_or_prog
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)
    codegen = PTOCodegen()
    result = codegen.generate(prog)
    return result if isinstance(result, str) else "".join(result.values())


# ---------------------------------------------------------------------------
# Kernels — all at module level so @fe.kernel can capture source lines
# ---------------------------------------------------------------------------

# --- 1. Plain if/else (no loop) ---
@fe.kernel
def _if_else_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
    b: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    tile_type_b = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_b = plm.make_tile(tile_type_b, addr=0x4000, size=16384)
    tile_type_c = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_c = plm.make_tile(tile_type_c, addr=0x8000, size=16384)
    plm.load(tile_a, a, [0, 0])
    plm.load(tile_b, b, [0, 0])
    flag = 1
    if flag == 1:
        plm.add(tile_c, tile_a, tile_b)
    else:
        plm.mul(tile_c, tile_a, tile_b)
    return b


# --- 2. For loop (basic, no break/continue) ---
@fe.kernel
def _for_basic_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
    b: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    tile_type_b = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_b = plm.make_tile(tile_type_b, addr=0x4000, size=16384)
    tile_type_c = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_c = plm.make_tile(tile_type_c, addr=0x8000, size=16384)
    plm.load(tile_a, a, [0, 0])
    plm.load(tile_b, b, [0, 0])
    for i in pl.range(5):
        plm.add(tile_c, tile_a, tile_b)
    return b


# --- 3. For loop + continue ---
@fe.kernel
def _for_continue_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
    b: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    tile_type_b = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_b = plm.make_tile(tile_type_b, addr=0x4000, size=16384)
    tile_type_c = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_c = plm.make_tile(tile_type_c, addr=0x8000, size=16384)
    plm.load(tile_a, a, [0, 0])
    plm.load(tile_b, b, [0, 0])
    for i in pl.range(5):
        if i == 2:
            continue
        plm.add(tile_c, tile_a, tile_b)
    return b


# --- 4. For loop + break ---
@fe.kernel
def _for_break_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
    b: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    tile_type_b = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_b = plm.make_tile(tile_type_b, addr=0x4000, size=16384)
    tile_type_c = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_c = plm.make_tile(tile_type_c, addr=0x8000, size=16384)
    plm.load(tile_a, a, [0, 0])
    plm.load(tile_b, b, [0, 0])
    for i in pl.range(5):
        if i == 2:
            break
        plm.add(tile_c, tile_a, tile_b)
    return b


# --- 5. While loop (basic, no break/continue) ---
@fe.kernel
def _while_basic_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
    b: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    tile_type_b = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_b = plm.make_tile(tile_type_b, addr=0x4000, size=16384)
    tile_type_c = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_c = plm.make_tile(tile_type_c, addr=0x8000, size=16384)
    plm.load(tile_a, a, [0, 0])
    plm.load(tile_b, b, [0, 0])
    i = 0
    while i < 5:
        plm.add(tile_c, tile_a, tile_b)
        i = i + 1
    return b


# --- 6. While loop + continue ---
@fe.kernel
def _while_continue_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
    b: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    tile_type_b = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_b = plm.make_tile(tile_type_b, addr=0x4000, size=16384)
    tile_type_c = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_c = plm.make_tile(tile_type_c, addr=0x8000, size=16384)
    plm.load(tile_a, a, [0, 0])
    plm.load(tile_b, b, [0, 0])
    i = 0
    while i < 5:
        if i == 2:
            i = i + 1
            continue
        plm.add(tile_c, tile_a, tile_b)
        i = i + 1
    return b


# --- 7. While loop + break ---
@fe.kernel
def _while_break_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
    b: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    tile_type_b = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_b = plm.make_tile(tile_type_b, addr=0x4000, size=16384)
    tile_type_c = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_c = plm.make_tile(tile_type_c, addr=0x8000, size=16384)
    plm.load(tile_a, a, [0, 0])
    plm.load(tile_b, b, [0, 0])
    i = 0
    while i < 5:
        if i == 2:
            break
        plm.add(tile_c, tile_a, tile_b)
        i = i + 1
    return b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_plain_if_else():
    """if/else with side-effect tile ops → scf.if, no loop constructs.

    Expected MLIR structure:
        %cmp = arith.cmpi eq, %flag, %c1
        scf.if %cmp {
            pto.tadd ...
        } else {
            pto.tmul ...
        }
    """
    mlir = _compile_to_mlir(_if_else_kernel)
    print("\n=== test_plain_if_else MLIR ===")
    print(mlir)

    # Control flow structure
    assert "scf.if" in mlir, "Expected scf.if for the if/else branch"
    assert "scf.for" not in mlir, "No for loop in this kernel"
    assert "scf.while" not in mlir, "No while loop in this kernel"

    # Both branch operations must be present
    assert "pto.tadd" in mlir, "Expected pto.tadd in then-branch"
    assert "pto.tmul" in mlir, "Expected pto.tmul in else-branch"

    # No break/continue flags: no i1 type for loop control
    assert "arith.constant 1 : i1" not in mlir, "No _can_continue flag expected"


def test_for_basic():
    """for i in pl.range(5) with tile op → scf.for, no inner scf.if.

    Expected MLIR structure:
        scf.for %i = %c0 to %c5 step %c1 {
            pto.tadd ...
        }
    """
    mlir = _compile_to_mlir(_for_basic_kernel)
    print("\n=== test_for_basic MLIR ===")
    print(mlir)

    # Outer loop structure
    assert "scf.for" in mlir, "Expected scf.for for the range loop"
    assert "scf.while" not in mlir, "No while loop: basic for needs no while"

    # No break/continue → no inner scf.if for control flow
    assert "scf.if" not in mlir, "No if/else in kernel body, no scf.if expected"

    # Body op
    assert "pto.tadd" in mlir, "Expected pto.tadd in loop body"


def test_for_continue():
    """for + continue → scf.for with an inner scf.if (inverted continue guard).

    LowerBreakContinue rewrites:
        if i == 2: continue
        plm.add(...)
    to:
        if i != 2:    # condition inverted
            plm.add(...)

    Expected MLIR structure:
        scf.for %i = %c0 to %c5 step %c1 {
            %cmp = arith.cmpi ne, %i, %c2
            scf.if %cmp {
                pto.tadd ...
            }
        }
    """
    mlir = _compile_to_mlir(_for_continue_kernel)
    print("\n=== test_for_continue MLIR ===")
    print(mlir)

    # Outer loop: still a for (continue does NOT need a while)
    assert "scf.for" in mlir, "Expected scf.for — continue in for stays a for"
    assert "scf.while" not in mlir, "Continue in for should not introduce scf.while"

    # Inner guard: scf.if from the inverted continue condition
    assert "scf.if" in mlir, "Expected scf.if for continue guard"

    # tadd is guarded inside the scf.if — must be present
    assert "pto.tadd" in mlir, "Expected pto.tadd inside the continue guard"

    # No break flag
    assert "i1" not in mlir, "No _can_continue i1 flag for plain continue-in-for"


def test_for_break():
    """for + break → scf.for with i1 iter_arg (_can_continue) and nested scf.if.

    LowerBreakContinue rewrites the for loop to carry _can_continue=true and
    guards the body behind scf.if(_can_continue).

    Expected MLIR structure:
        scf.for %i = ... iter_args(%cont = %true) -> (i1) {
            %res = scf.if %cont -> (i1) {
                %flag, %cont2 = scf.if (%i == 2) -> (i1, i1) {
                    scf.yield %false, %false  // break path
                } else {
                    pto.tadd ...
                    scf.yield %true, %true
                }
                scf.yield %cont2 : i1
            } else {
                scf.yield %cont : i1
            }
            scf.yield %res : i1
        }
    """
    mlir = _compile_to_mlir(_for_break_kernel)
    print("\n=== test_for_break MLIR ===")
    print(mlir)

    # Still a for loop at the outer level
    assert "scf.for" in mlir, "Expected scf.for — break-in-for stays a for"
    assert "scf.while" not in mlir, "Break in for should not introduce scf.while"

    # Nested scf.if for the _can_continue guard and the break condition
    assert "scf.if" in mlir, "Expected scf.if for break guards"
    assert mlir.count("scf.if") >= 2, "Expected at least 2 scf.if (outer guard + break site)"

    # _can_continue is an i1 iter_arg on the scf.for
    assert "i1" in mlir, "Expected i1 type for _can_continue iter_arg"

    # false constant for the break path (emitted as arith.constant 0 : i1)
    assert "arith.constant 0 : i1" in mlir, "Expected arith.constant 0 : i1 for break path"

    # tadd is inside the else (non-break) path
    assert "pto.tadd" in mlir, "Expected pto.tadd in the non-break path"


def test_while_basic():
    """while i < 5 with loop var update → scf.while + scf.condition, no inner scf.if.

    ConvertToSSA promotes i to an iter_arg.

    Expected MLIR structure:
        scf.while (%i = %c0) : (index) -> (index) {
            %cond = arith.cmpi slt, %i, %c5
            scf.condition(%cond) %i : index
        } do {
            ^bb0(%i: index):
            pto.tadd ...
            %new_i = arith.addi %i, %c1
            scf.yield %new_i : index
        }
    """
    mlir = _compile_to_mlir(_while_basic_kernel)
    print("\n=== test_while_basic MLIR ===")
    print(mlir)

    # Outer loop structure
    assert "scf.while" in mlir, "Expected scf.while for the natural while loop"
    assert "scf.condition" in mlir, "Expected scf.condition in scf.while before-region"
    assert "scf.for" not in mlir, "No for loop in this kernel"

    # No break/continue → no inner scf.if for control flow
    assert "scf.if" not in mlir, "No if/else in kernel body, no scf.if expected"

    # Loop counter update and body op
    assert "arith.addi" in mlir, "Expected arith.addi for i = i + 1"
    assert "pto.tadd" in mlir, "Expected pto.tadd in loop body"

    # No break flag
    assert "i1" not in mlir, "No _can_continue i1 flag for basic while"


def test_while_continue():
    """while + continue → scf.while with inner scf.if (continue guard).

    LowerBreakContinue rewrites:
        if i == 2: i = i + 1; continue
        plm.add(...)
        i = i + 1
    to:
        if i != 2:    # condition inverted, remaining stmts become the else-free then-body
            plm.add(...)
            i = i + 1

    Expected MLIR structure:
        scf.while (%i = ...) : (index) -> (index) {
            scf.condition(%cond) %i
        } do {
            ...
            %guard = arith.cmpi ne, %i, %c2
            %new_i = scf.if %guard -> (index) {
                pto.tadd ...
                scf.yield %updated_i
            } else {
                scf.yield %i_from_continue_path
            }
            scf.yield %new_i
        }
    """
    mlir = _compile_to_mlir(_while_continue_kernel)
    print("\n=== test_while_continue MLIR ===")
    print(mlir)

    # Outer loop
    assert "scf.while" in mlir, "Expected scf.while"
    assert "scf.condition" in mlir, "Expected scf.condition"

    # Inner scf.if from the continue guard
    assert "scf.if" in mlir, "Expected inner scf.if for continue guard"

    # tadd is inside the scf.if then-branch (non-continue path)
    assert "pto.tadd" in mlir, "Expected pto.tadd inside the non-continue branch"

    # No break flag (continue only)
    assert "i1" not in mlir, "No _can_continue i1 flag for plain continue-in-while"


def test_while_break():
    """while + break → scf.while with i1 iter_arg (_can_continue) and inner scf.if.

    LowerBreakContinue rewrites the while loop to use _can_continue=true as the sole
    before-region condition; the original while condition is checked inside the do-region.

    Expected MLIR structure:
        scf.while (%cont = %true, %i = %c0) : (i1, index) -> (i1, index) {
            scf.condition(%cont) %cont, %i
        } do {
            ^bb0(%cont: i1, %i: index):
            %orig_cond = arith.cmpi slt, %i, %c5
            %res_cont, %res_i = scf.if %orig_cond -> (i1, index) {
                %flag, %cont2 = scf.if (%i == 2) -> (i1, i1) {
                    scf.yield %false, %false    // break path
                } else {
                    pto.tadd ...
                    scf.yield %updated_i, %true
                }
                scf.yield %cont2, %flag
            } else {
                scf.yield %false, %i    // loop condition false → stop
            }
            scf.yield %res_cont, %res_i
        }
    """
    mlir = _compile_to_mlir(_while_break_kernel)
    print("\n=== test_while_break MLIR ===")
    print(mlir)

    # Outer loop
    assert "scf.while" in mlir, "Expected scf.while"
    assert "scf.condition" in mlir, "Expected scf.condition"

    # _can_continue is an i1 iter_arg — appears in the scf.while type signature
    assert "i1" in mlir, "Expected i1 for _can_continue iter_arg"

    # Inner scf.if for the original while condition and the break condition
    assert "scf.if" in mlir, "Expected scf.if for break guard"
    assert mlir.count("scf.if") >= 2, "Expected at least 2 scf.if (orig-cond guard + break site)"

    # false constant for the break/stop path (emitted as arith.constant 0 : i1)
    assert "arith.constant 0 : i1" in mlir, "Expected arith.constant 0 : i1 for break path"

    # tadd is in the non-break path
    assert "pto.tadd" in mlir, "Expected pto.tadd in the non-break path"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
