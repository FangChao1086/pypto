# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend tests for the @fe.kernel decorator with manual (non-SSA) plm.* ops.

Each test kernel is compiled to PTO MLIR and the output is checked for
correct pto.alloc_tile / pto.tload / pto.tadd / pto.tmul patterns.
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

def _compile_to_mlir(prog) -> str:
    """Compile an ir.Program to PTO MLIR without running external tools."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)
    codegen = PTOCodegen()
    result = codegen.generate(prog)
    return result if isinstance(result, str) else "".join(result.values())


# ---------------------------------------------------------------------------
# Test kernels — defined at module level so @fe.kernel runs at import time
# ---------------------------------------------------------------------------

# Kernel 1: two loads only (baseline)
@fe.kernel
def load_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
    b: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(
        shape=[64, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[32, 64],
        blayout=1,
        slayout=2,
        fractal=1,
        pad=2,
    )
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    plm.load(tile_a, a, [0, 0])

    tile_type_b = plm.TileType(
        shape=[64, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[32, 64],
        blayout=2,
        slayout=1,
        fractal=1,
        pad=2,
    )
    tile_b = plm.make_tile(tile_type_b, addr=0x4000, size=16384)
    plm.load(tile_b, b, [0, 0])
    return b


# Kernel 2: load two tiles and add them
@fe.kernel
def add_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
    b: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(
        shape=[64, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    plm.load(tile_a, a, [0, 0])

    tile_type_b = plm.TileType(
        shape=[64, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    tile_b = plm.make_tile(tile_type_b, addr=0x4000, size=16384)
    plm.load(tile_b, b, [0, 0])

    tile_type_c = plm.TileType(
        shape=[64, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    tile_c = plm.make_tile(tile_type_c, addr=0x8000, size=16384)
    plm.add(tile_c, tile_a, tile_b)
    return b


# Kernel 3: load two tiles and multiply them
@fe.kernel
def mul_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
    b: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(
        shape=[64, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    plm.load(tile_a, a, [0, 0])

    tile_type_b = plm.TileType(
        shape=[64, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    tile_b = plm.make_tile(tile_type_b, addr=0x4000, size=16384)
    plm.load(tile_b, b, [0, 0])

    tile_type_c = plm.TileType(
        shape=[64, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    tile_c = plm.make_tile(tile_type_c, addr=0x8000, size=16384)
    plm.mul(tile_c, tile_a, tile_b)
    return b


# Kernel 4: unary neg
@fe.kernel(auto_sync=True)
def neg_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(
        shape=[64, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    plm.load(tile_a, a, [0, 0])

    tile_type_b = plm.TileType(
        shape=[64, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Vec,
    )
    tile_b = plm.make_tile(tile_type_b, addr=0x4000, size=16384)
    plm.neg(tile_b, tile_a)
    return a


# ---------------------------------------------------------------------------
# Test functions (run with: python test_kernel.py)
# ---------------------------------------------------------------------------
@fe.jit()
def test_load_kernel_with_jit():
    compiled_kernel = fe.compile(load_kernel)
    return

def test_load_kernel():
    """Two loads: verify alloc_tile and tload patterns."""
    mlir = _compile_to_mlir(load_kernel)
    print("\n=== test_load_kernel MLIR ===")
    print(mlir)

    assert "pto.alloc_tile" in mlir, "Expected pto.alloc_tile"
    assert "dtype=f16" in mlir, "Expected f16 dtype"
    assert "rows=64, cols=128" in mlir, "Expected 64x128 tile shape"
    # tile_b is at 0x4000 = 16384 — should carry explicit addr constant
    assert "v_row=32, v_col=64" in mlir, "Expected v_row=32, v_col=64 for valid_shape"
    assert "arith.constant 16384 : i64" in mlir, "Expected arith.constant 16384 for tile_b addr"
    assert "blayout=col_major" in mlir, "Expected blayout=col_major"
    assert "slayout=row_major" in mlir, "Expected slayout=row_major"
    assert "fractal=1" in mlir, "Expected fractal=0"
    assert "pad=2" in mlir, "Expected pad=0"
    assert "pto.partition_view" in mlir, "Expected pto.partition_view for load"
    assert "pto.tload" in mlir, "Expected pto.tload"
    assert mlir.count("pto.tload") == 2, f"Expected 2 tloads, got {mlir.count('pto.tload')}"


def test_add_kernel():
    """Two loads + add: verify tadd and three alloc_tile."""
    mlir = _compile_to_mlir(add_kernel)
    print("\n=== test_add_kernel MLIR ===")
    print(mlir)

    # 3 tiles allocated
    n_alloc = mlir.count("pto.alloc_tile")
    assert n_alloc == 3, f"Expected 3 pto.alloc_tile, got {n_alloc}"
    # tile_b at 0x4000 = 16384
    assert "arith.constant 16384 : i64" in mlir, "Expected arith.constant 16384 for tile_b addr"
    # tile_c at 0x8000 = 32768
    assert "arith.constant 32768 : i64" in mlir, "Expected arith.constant 32768 for tile_c addr"
    # 2 loads and 1 add
    assert mlir.count("pto.tload") == 2, f"Expected 2 tloads, got {mlir.count('pto.tload')}"
    assert "pto.tadd" in mlir, "Expected pto.tadd"


def test_mul_kernel():
    """Two loads + mul: verify tmul."""
    mlir = _compile_to_mlir(mul_kernel)
    print("\n=== test_mul_kernel MLIR ===")
    print(mlir)

    assert mlir.count("pto.alloc_tile") == 3, "Expected 3 pto.alloc_tile"
    assert mlir.count("pto.tload") == 2, "Expected 2 pto.tload"
    assert "pto.tmul" in mlir, "Expected pto.tmul"


def test_neg_kernel():
    """Load + unary neg: verify tneg."""
    mlir = _compile_to_mlir(neg_kernel)
    print("\n=== test_neg_kernel MLIR ===")
    print(mlir)

    assert mlir.count("pto.alloc_tile") == 2, "Expected 2 pto.alloc_tile"
    assert "pto.tload" in mlir, "Expected pto.tload"
    assert "pto.tneg" in mlir, "Expected pto.tneg"


# ---------------------------------------------------------------------------
# Auto-inline: plain Python helper (no decorator)
# ---------------------------------------------------------------------------

def plain_offset(base, stride):
    """Regular Python function with no DSL type annotations."""
    return base + stride


def test_auto_func_kernel():
    """Unannotated plain_offset is auto-inlined: arith.addi appears, no func.func."""

    @fe.kernel
    def _kernel(
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
        idx: pl.Scalar[pl.INDEX] = plain_offset(1, 2)
        plm.add(tile_c, tile_a, tile_b)
        return b

    mlir = _compile_to_mlir(_kernel)
    print("\n=== test_auto_func_kernel MLIR ===")
    print(mlir)

    assert "arith.addi" in mlir or "arith.add" in mlir, "Expected inline arithmetic from plain_offset"
    assert "func.func @plain_offset" not in mlir, "Should not emit func.func for auto-inlined fn"


# ---------------------------------------------------------------------------
# @pl.func: helper function generating func.call in MLIR
# ---------------------------------------------------------------------------

def compute_offset(base: pl.Scalar[pl.INDEX], stride: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
    """Helper function compiled to a separate func.func definition."""
    return base + stride + pl.block.get_block_idx()


@fe.kernel
def func_call_kernel(
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
    offset = compute_offset(0, 64)   # → func.call @compute_offset
    plm.add(tile_c, tile_a, tile_b)
    return b


def test_func_kernel():
    """func_call_kernel."""
    mlir = _compile_to_mlir(func_call_kernel)
    print("\n=== func_call_kernel MLIR ===")
    print(mlir)

    assert "func.func @compute_offset" in mlir, "Expected func @compute_offset"


# ---------------------------------------------------------------------------
# @pl.func with Tile params
# ---------------------------------------------------------------------------

# @pl.func
def tile_add_helper(
    src_a: pl.Tile[[64, 128], pl.FP16],
    src_b: pl.Tile[[64, 128], pl.FP16],
    dst: pl.Tile[[64, 128], pl.FP16],
) -> pl.Scalar[pl.INDEX]:
    plm.add(dst, src_a, src_b)
    return 0


@fe.kernel
def tile_param_kernel(
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
    result = tile_add_helper(tile_a, tile_b, tile_c)   # → func.call with tile_buf params
    return b


def test_tile_param_kernel():
    """@pl.func with Tile params → func.func with !pto.tile_buf<...> signature."""
    mlir = _compile_to_mlir(tile_param_kernel)
    print("\n=== tile_param_kernel MLIR ===")
    print(mlir)

    assert "func.func @tile_add_helper" in mlir, "Expected helper func definition"
    assert "!pto.tile_buf" in mlir, "Expected !pto.tile_buf in helper signature"
    assert "func.call @tile_add_helper" in mlir, "Expected func.call in kernel"


if __name__ == "__main__":
    test_load_kernel_with_jit()
    test_load_kernel()
    test_add_kernel()
    test_func_kernel()
    test_auto_func_kernel()
    test_tile_param_kernel()
    test_mul_kernel()
    test_neg_kernel()
    print("\nAll tests passed!")
