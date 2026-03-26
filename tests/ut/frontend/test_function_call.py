# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Test cases for @pl.func and function call support.

Tests cover:
1. @pl.func decorator and KernelFunction
2. func.call generation in PTO MLIR
3. Implicit compilation of annotated plain Python functions (no decorator needed)
4. Scalar helper functions with proper type annotations
5. Multiple function calls and code reuse
6. Mixed usage of @pl.func and @pl.inline
7. Control flow with function calls
8. Block operations in helper functions
9. Nested function calls
10. Different scalar types (int, float)
"""

import os
import shutil
import subprocess
import tempfile

import pytest
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm
from pypto import backend
from pypto.backend import BackendType
from pypto.pypto_core.codegen import PTOCodegen
from pypto.language.parser.diagnostics import UnsupportedFeatureError


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


def _validate_mlir_with_ptoas(mlir: str) -> None:
    """Validate MLIR string by running ptoas. Skips if ptoas is not available."""
    ptoas_root = os.environ.get("PTOAS_ROOT")
    ptoas_bin = os.path.join(ptoas_root, "ptoas") if ptoas_root else shutil.which("ptoas")
    if not ptoas_bin:
        pytest.skip("ptoas not available — skipping MLIR validation")

    with tempfile.NamedTemporaryFile(suffix=".pto", mode="w", delete=False) as f:
        f.write(mlir)
        pto_path = f.name
    out_path = pto_path.replace(".pto", ".cpp")
    try:
        result = subprocess.run(
            [ptoas_bin, pto_path, "--enable-insert-sync", "--pto-level=level3", "-o", out_path],
            capture_output=True, text=True, check=False, timeout=30,
        )
        assert result.returncode == 0, f"ptoas validation failed:\n{result.stderr.strip()}"
        with open(out_path) as cpp_f:
            print(f"\n=== ptoas output ({os.path.basename(pto_path)}) ===")
            print(cpp_f.read())
    except subprocess.TimeoutExpired:
        pytest.fail("ptoas timed out")
    finally:
        os.unlink(pto_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


# ===========================================================================
# Test 1: Basic @pl.func with scalar computation
# ===========================================================================


# @pl.func
def add_offset(base: pl.Scalar[pl.INDEX], offset: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
    """Simple scalar helper function."""
    return base + offset


@fe.kernel
def test_basic_func_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
    b: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    plm.load(tile_a, a, [0, 0])

    tile_type_b = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_b = plm.make_tile(tile_type_b, addr=0x4000, size=16384)
    plm.load(tile_b, b, [0, 0])

    offset = add_offset(0, 64)

    tile_type_c = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_c = plm.make_tile(tile_type_c, addr=0x8000, size=16384)
    plm.add(tile_c, tile_a, tile_b)
    return b


def test_basic_func():
    """Test basic @pl.func with scalar computation."""
    mlir = _compile_to_mlir(test_basic_func_kernel)
    print("\n=== test_basic_func MLIR ===")
    print(mlir)

    assert "func.func @add_offset" in mlir, "Expected func.func @add_offset"
    assert "func.call @add_offset" in mlir, "Expected func.call @add_offset"
    assert "%arg0: index" in mlir, "Expected %arg0: index"
    assert "%arg1: index" in mlir, "Expected %arg1: index"
    assert "-> index" in mlir, "Expected -> index return type"


# ===========================================================================
# Test 2: @pl.func with block operations
# ===========================================================================


# @pl.func
def add_constant(base: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
    """Helper function that adds constant."""
    return base + 10


@fe.kernel
def test_block_op_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    plm.load(tile_a, a, [0, 0])

    # Call @pl.func helper with simple operation
    result = add_constant(0)

    return a


def test_block_op_func():
    """Test @pl.func with simple operations."""
    mlir = _compile_to_mlir(test_block_op_kernel)
    print("\n=== test_block_op_func MLIR ===")
    print(mlir)

    # Verify func.func definition
    assert "func.func @add_constant" in mlir, "Expected func.func @add_constant"

    # Verify func.call
    assert "func.call @add_constant" in mlir, "Expected func.call @add_constant"

    # Verify constant arithmetic is inlined
    assert "arith.addi" in mlir or "arith.add" in mlir, "Expected inline arithmetic"


# ===========================================================================
# Test 3: Multiple @pl.func calls (code reuse)
# ===========================================================================


# @pl.func
def compute_index(base: pl.Scalar[pl.INDEX], stride: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
    """Reusable index computation."""
    return base + stride * 2


@fe.kernel
def test_multiple_calls_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    plm.load(tile_a, a, [0, 0])

    idx1 = compute_index(0, 1)
    idx2 = compute_index(0, 2)
    idx3 = compute_index(0, 3)

    return a


def test_multiple_func_calls():
    """Test multiple calls to same @pl.func (code reuse)."""
    mlir = _compile_to_mlir(test_multiple_calls_kernel)
    print("\n=== test_multiple_func_calls MLIR ===")
    print(mlir)

    assert mlir.count("func.func @compute_index") == 1, "Expected single func.func definition"
    assert mlir.count("func.call @compute_index") == 3, "Expected 3 func.call instances"
    assert "%arg1: index" in mlir, "Expected index parameter type"


# ===========================================================================
# Test 4: @pl.func with complex scalar expressions
# ===========================================================================


# @pl.func
def complex_calc(
    x: pl.Scalar[pl.INDEX], y: pl.Scalar[pl.INDEX], z: pl.Scalar[pl.INDEX]
) -> pl.Scalar[pl.INDEX]:
    """Complex scalar computation with multiple operations."""
    temp1 = x + y
    temp2 = temp1 * z
    return temp2 / 2


@fe.kernel
def test_complex_expr_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    plm.load(tile_a, a, [0, 0])

    result = complex_calc(10, 20, 5)

    return a


def test_complex_func():
    """Test @pl.func with complex scalar expressions."""
    mlir = _compile_to_mlir(test_complex_expr_kernel)
    print("\n=== test_complex_func MLIR ===")
    print(mlir)

    assert "func.func @complex_calc" in mlir, "Expected func.func @complex_calc"
    assert "func.call @complex_calc" in mlir, "Expected func.call @complex_calc"
    assert "%arg0: index" in mlir
    assert "%arg1: index" in mlir
    assert "%arg2: index" in mlir


# ===========================================================================
# Test 5: Unannotated plain function is auto-inlined (no decorator needed)
# ===========================================================================


def unannotated_helper(base, stride):
    """Plain Python function with no DSL type annotations."""
    return base + stride


@fe.kernel
def test_auto_inline_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    plm.load(tile_a, a, [0, 0])
    result: pl.Scalar[pl.INDEX] = unannotated_helper(0, 64)
    return a


def test_auto_inline_no_annotations():
    """Unannotated plain function is auto-inlined: arith.addi appears, no func.func."""
    mlir = _compile_to_mlir(test_auto_inline_kernel)
    print("\n=== test_auto_inline_no_annotations MLIR ===")
    print(mlir)

    assert "arith.addi" in mlir or "arith.add" in mlir, "Expected inline arithmetic"
    assert "func.func @unannotated_helper" not in mlir, "Should not emit func.func for auto-inlined fn"


# ===========================================================================
# Test 5c: Auto-inline with nested call raises UnsupportedFeatureError
# ===========================================================================


def nested_inner(x):
    return x + 1


def nested_outer(base, stride):
    """Unannotated function that calls another plain function — forbidden."""
    return nested_inner(base) + stride


def test_auto_inline_nested_call_raises():
    """Auto-inlined function with a nested bare-name call raises UnsupportedFeatureError."""
    @fe.kernel
    def kernel_with_nested(
        a: pl.Tensor[[64, 128], pl.FP16],
    ) -> pl.Tensor[[64, 128], pl.FP16]:
        tile_type_a = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
        tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
        plm.load(tile_a, a, [0, 0])
        result: pl.Scalar[pl.INDEX] = nested_outer(0, 64)
        return a

    with pytest.raises(UnsupportedFeatureError, match="cannot call other functions"):
        _compile_to_mlir(kernel_with_nested)


# ===========================================================================
# Test 5b: Implicit @pl.func — annotated plain function compiles to func.call
# ===========================================================================


def implicit_add(x: pl.Scalar[pl.INDEX], y: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
    """Annotated plain function — no @pl.func decorator needed."""
    return x + y


@fe.kernel
def test_implicit_func_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    plm.load(tile_a, a, [0, 0])

    result = implicit_add(10, 20)

    return a


def test_implicit_func():
    """Annotated plain function without @pl.func generates func.func + func.call."""
    mlir = _compile_to_mlir(test_implicit_func_kernel)
    print("\n=== test_implicit_func MLIR ===")
    print(mlir)

    assert "func.func @implicit_add" in mlir, "Expected func.func @implicit_add"
    assert "func.call @implicit_add" in mlir, "Expected func.call @implicit_add"
    assert "%arg0: index" in mlir, "Expected index parameter"


# ===========================================================================
# Test 6: Mixed @pl.func and @pl.inline usage
# ===========================================================================


# @pl.func
def func_helper(x: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
    """@pl.func helper."""
    return x + 1


@pl.inline
def inline_helper(x: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
    """@pl.inline helper."""
    return x + 2


@fe.kernel
def test_mixed_helpers_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    plm.load(tile_a, a, [0, 0])

    ten: pl.Scalar[pl.INDEX] = 10
    twenty: pl.Scalar[pl.INDEX] = 20
    result1 = func_helper(ten)
    result2 = inline_helper(twenty)

    return a


def test_mixed_helpers():
    """Test mixed usage of @pl.func and @pl.inline."""
    mlir = _compile_to_mlir(test_mixed_helpers_kernel)
    print("\n=== test_mixed_helpers MLIR ===")
    print(mlir)

    assert "func.call @func_helper" in mlir, "Expected func.call for @pl.func"
    assert "func.call @inline_helper" not in mlir, "Should not have func.call for @pl.inline"
    assert "arith.addi" in mlir or "arith.add" in mlir, "Expected inline arithmetic"


# ===========================================================================
# Test 7: @pl.func with different scalar types
# ===========================================================================


# @pl.func
def float_helper(x: pl.Scalar[pl.FP32], y: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
    """Float scalar helper."""
    return x + y


@fe.kernel
def test_float_types_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    plm.load(tile_a, a, [0, 0])

    x_val: pl.Scalar[pl.FP32] = 1.5
    y_val: pl.Scalar[pl.FP32] = 2.5
    result = float_helper(x_val, y_val)

    return a


def test_float_types():
    """Test @pl.func with float scalar types."""
    mlir = _compile_to_mlir(test_float_types_kernel)
    print("\n=== test_float_types MLIR ===")
    print(mlir)

    assert "func.func @float_helper" in mlir, "Expected func.func @float_helper"
    assert "f32" in mlir, "Expected f32 type annotations"


# ===========================================================================
# Test 8: @pl.func with block.get_block_num
# ===========================================================================


# @pl.func
def get_block_count() -> pl.Scalar[pl.INDEX]:
    """Helper to get block count."""
    bn = pl.block.get_block_num()
    bni = pl.block.index_cast(bn)
    return bni


@fe.kernel
def test_block_num_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    plm.load(tile_a, a, [0, 0])

    count = get_block_count()

    return a


def test_block_num():
    """Test @pl.func with block.get_block_num."""
    mlir = _compile_to_mlir(test_block_num_kernel)
    print("\n=== test_block_num MLIR ===")
    print(mlir)

    assert "func.func @get_block_count" in mlir, "Expected func.func @get_block_count"
    assert "func.call @get_block_count" in mlir, "Expected func.call @get_block_count"
    assert "pto.get_block_num" in mlir, "Expected pto.get_block_num"


# ===========================================================================
# Test 9: @pl.func with nested calls
# ===========================================================================


# @pl.func
def inner_add(x: pl.Scalar[pl.INDEX], y: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
    """Inner helper."""
    return x + y


# @pl.func
def outer_calc(x: pl.Scalar[pl.INDEX]) -> pl.Scalar[pl.INDEX]:
    """Outer helper calling inner helper."""
    return inner_add(x, 10)


@fe.kernel
def test_nested_calls_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    plm.load(tile_a, a, [0, 0])

    x_val: pl.Scalar[pl.INDEX] = 5
    result = outer_calc(x_val)

    return a


def test_nested_calls():
    """Test nested @pl.func calls."""
    mlir = _compile_to_mlir(test_nested_calls_kernel)
    print("\n=== test_nested_calls MLIR ===")
    print(mlir)

    assert "func.func @inner_add" in mlir, "Expected func.func @inner_add"
    assert "func.func @outer_calc" in mlir, "Expected func.func @outer_calc"
    assert mlir.count("func.call") >= 2, "Expected multiple func.call instances"


# ===========================================================================
# Test 10: @pl.func with INT64 type
# ===========================================================================


# @pl.func
def int64_helper(x: pl.Scalar[pl.INT64], y: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
    """INT64 scalar helper."""
    return x + y


@fe.kernel
def test_int64_types_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    plm.load(tile_a, a, [0, 0])

    x_val: pl.Scalar[pl.INT64] = 1000
    y_val: pl.Scalar[pl.INT64] = 2000
    result = int64_helper(x_val, y_val)

    return a


def test_int64_types():
    """Test @pl.func with INT64 scalar types."""
    mlir = _compile_to_mlir(test_int64_types_kernel)
    print("\n=== test_int64_types MLIR ===")
    print(mlir)

    assert "func.func @int64_helper" in mlir, "Expected func.func @int64_helper"
    assert "i64" in mlir, "Expected i64 type annotations"


# ===========================================================================
# Test 11: @pl.func with Tile params (helper modifying tiles)
# ===========================================================================


# @pl.func
def tile_neg_helper(
    src: pl.Tile[[64, 128], pl.FP16],
    dst: pl.Tile[[64, 128], pl.FP16],
) -> pl.Scalar[pl.INDEX]:
    """Tile helper that applies unary negation."""
    plm.neg(dst, src)
    return 0


@fe.kernel
def test_tile_neg_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)
    plm.load(tile_a, a, [0, 0])

    tile_type_b = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_b = plm.make_tile(tile_type_b, addr=0x4000, size=16384)
    result = tile_neg_helper(tile_a, tile_b)

    return a


def test_tile_neg():
    """Test @pl.func with tile params using unary op."""
    mlir = _compile_to_mlir(test_tile_neg_kernel)
    print("\n=== test_tile_neg MLIR ===")
    print(mlir)

    assert "func.func @tile_neg_helper" in mlir, "Expected func.func @tile_neg_helper"
    assert "!pto.tile_buf" in mlir, "Expected !pto.tile_buf in helper signature"
    assert "func.call @tile_neg_helper" in mlir, "Expected func.call @tile_neg_helper"
    assert "pto.tneg" in mlir, "Expected pto.tneg in helper body"
    _validate_mlir_with_ptoas(mlir)


# ===========================================================================
# Test 12: @pl.func with mixed Tile + scalar params
# ===========================================================================


# @pl.func
def scaled_load_helper(
    src: pl.Tile[[32, 64], pl.FP16],
    dst: pl.Tile[[32, 64], pl.FP16],
    scale: pl.Scalar[pl.INDEX],
):
    """Helper that takes both tile and scalar params."""
    plm.add(dst, src, src)
    # return scale + 1


@fe.kernel
def test_mixed_tile_scalar_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type_a = plm.TileType(shape=[32, 64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=8192)
    plm.load(tile_a, a, [0, 0])

    tile_type_b = plm.TileType(shape=[32, 64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_b = plm.make_tile(tile_type_b, addr=0x2000, size=8192)
    scaled_load_helper(tile_a, tile_b, 4)

    return a


def test_mixed_tile_scalar():
    """Test @pl.func with mixed Tile and scalar params."""
    mlir = _compile_to_mlir(test_mixed_tile_scalar_kernel)
    print("\n=== test_mixed_tile_scalar MLIR ===")
    print(mlir)

    assert "func.func @scaled_load_helper" in mlir, "Expected func.func @scaled_load_helper"
    assert "!pto.tile_buf" in mlir, "Expected !pto.tile_buf params"
    assert "index" in mlir, "Expected index scalar param"
    assert "func.call @scaled_load_helper" in mlir, "Expected func.call @scaled_load_helper"
    assert "pto.tadd" in mlir, "Expected pto.tadd in helper body"
    assert "-> ()" in mlir, "Expected void func.call (-> ())"
    assert "return" in mlir, "Expected return in void helper body"
    _validate_mlir_with_ptoas(mlir)


# ===========================================================================
# Test 13: @pl.func with small tile shape (32x64)
# ===========================================================================


# @pl.func
def tile_mul_32x64(
    src_a: pl.Tile[[32, 64], pl.FP16],
    src_b: pl.Tile[[32, 64], pl.FP16],
    dst: pl.Tile[[32, 64], pl.FP16],
):
    """Helper for 32x64 tile multiplication."""
    plm.mul(dst, src_a, src_b)
    # return


@fe.kernel
def test_small_tile_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
    b: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_type = plm.TileType(shape=[32, 64], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type, addr=0x0000, size=8192)
    tile_b = plm.make_tile(tile_type, addr=0x2000, size=8192)
    tile_c = plm.make_tile(tile_type, addr=0x4000, size=8192)
    plm.load(tile_a, a, [0, 0])
    plm.load(tile_b, b, [0, 0])
    tile_mul_32x64(tile_a, tile_b, tile_c)

    return a


def test_small_tile_shape():
    """Test @pl.func with 32x64 tile shape."""
    mlir = _compile_to_mlir(test_small_tile_kernel)
    print("\n=== test_small_tile_shape MLIR ===")
    print(mlir)

    assert "func.func @tile_mul_32x64" in mlir, "Expected func.func @tile_mul_32x64"
    assert "rows=32, cols=64" in mlir, "Expected rows=32, cols=64 in tile type"
    assert "pto.tmul" in mlir, "Expected pto.tmul in helper body"
    assert "func.call @tile_mul_32x64" in mlir, "Expected func.call @tile_mul_32x64"
    assert "-> ()" in mlir, "Expected void func.call (-> ())"
    _validate_mlir_with_ptoas(mlir)


# ===========================================================================
# Test Runner
# ===========================================================================

if __name__ == "__main__":
    print("Running function call tests...")

    test_basic_func()
    test_block_op_func()
    test_multiple_func_calls()
    test_complex_func()
    test_auto_inline_no_annotations()
    test_auto_inline_nested_call_raises()
    test_implicit_func()
    test_mixed_helpers()
    test_float_types()
    test_block_num()
    test_nested_calls()
    test_int64_types()
    test_tile_neg()
    test_mixed_tile_scalar()
    test_small_tile_shape()

    print("\n✅ All function call tests passed!")
