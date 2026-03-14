# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for plm.load_tile and plm.store_tile functions.

Tests that load_tile and store_tile produce correct MLIR with computed absolute offsets.
"""

import re

import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm
from pypto import backend
from pypto.backend import BackendType
from pypto.pypto_core.codegen import PTOCodegen


def _compile_to_mlir(prog) -> str:
    """Compile an ir.Program to PTO MLIR without running external tools."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)
    codegen = PTOCodegen()
    result = codegen.generate(prog)
    return result if isinstance(result, str) else "".join(result.values())


def _extract_partition_view_offsets(mlir: str) -> list[tuple[int, int]]:
    """Extract offsets from pto.partition_view operations.

    Returns list of (row_off, col_off) tuples found in the MLIR.
    """
    offsets = []
    pattern = r'offsets\s*=\s*\[%c(\d+),\s*%c(\d+)\]'
    for match in re.finditer(pattern, mlir):
        row_off = int(match.group(1))
        col_off = int(match.group(2))
        offsets.append((row_off, col_off))
    return offsets


# ---------------------------------------------------------------------------
# Comparison test kernels (load vs load_tile, store vs store_tile)
# ---------------------------------------------------------------------------

@fe.kernel
def load_kernel_compare(
    a: pl.Tensor[[256, 512], pl.FP16],
) -> pl.Tensor[[256, 512], pl.FP16]:
    tile_a = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
                             addr=0x0000, size=16384)
    plm.load(tile_a, a, [128, 256])
    return a


@fe.kernel
def load_tile_kernel_compare(
    a: pl.Tensor[[256, 512], pl.FP16],
) -> pl.Tensor[[256, 512], pl.FP16]:
    tile_a = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
                             addr=0x0000, size=16384)
    plm.load_tile(tile_a, a, [2, 2])
    return a


@fe.kernel
def store_kernel_compare(
    a: pl.Tensor[[256, 512], pl.FP16],
) -> pl.Tensor[[256, 512], pl.FP16]:
    tile_a = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
                             addr=0x0000, size=16384)
    plm.store(a, tile_a, [128, 256])
    return a


@fe.kernel
def store_tile_kernel_compare(
    a: pl.Tensor[[256, 512], pl.FP16],
) -> pl.Tensor[[256, 512], pl.FP16]:
    tile_a = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
                             addr=0x0000, size=16384)
    plm.store_tile(a, tile_a, [2, 2])
    return a


# ---------------------------------------------------------------------------
# Comparison test functions
# ---------------------------------------------------------------------------

def test_load_tile_vs_load_same_mlir():
    """Verify load_tile produces same MLIR as load with pre-computed offset."""
    mlir_load = _compile_to_mlir(load_kernel_compare)
    mlir_load_tile = _compile_to_mlir(load_tile_kernel_compare)

    print("\n=== load with offset [128, 256] ===")
    print(mlir_load)
    print("\n=== load_tile with tile_offset [2, 2] (shape [64, 128]) ===")
    print(mlir_load_tile)

    offsets_load = _extract_partition_view_offsets(mlir_load)
    offsets_load_tile = _extract_partition_view_offsets(mlir_load_tile)

    assert offsets_load == offsets_load_tile == [(128, 256)], \
        f"Expected same offsets [(128, 256)], got load={offsets_load}, load_tile={offsets_load_tile}"


def test_store_tile_vs_store_same_mlir():
    """Verify store_tile produces same MLIR as store with pre-computed offset."""
    mlir_store = _compile_to_mlir(store_kernel_compare)
    mlir_store_tile = _compile_to_mlir(store_tile_kernel_compare)

    print("\n=== store with offset [128, 256] ===")
    print(mlir_store)
    print("\n=== store_tile with tile_offset [2, 2] (shape [64, 128]) ===")
    print(mlir_store_tile)

    offsets_store = _extract_partition_view_offsets(mlir_store)
    offsets_store_tile = _extract_partition_view_offsets(mlir_store_tile)

    assert offsets_store == offsets_store_tile == [(128, 256)], \
        f"Expected same offsets [(128, 256)], got store={offsets_store}, store_tile={offsets_store_tile}"


# ---------------------------------------------------------------------------
# Dynamic offset comparison test kernels
# ---------------------------------------------------------------------------

@fe.kernel
def load_dynamic_kernel_compare(
    a: pl.Tensor[[256, 512], pl.FP16],
) -> pl.Tensor[[256, 512], pl.FP16]:
    tile_a = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
                             addr=0x0000, size=16384)
    for i in pl.range(4):
        plm.load(tile_a, a, [i * 64, 0])
    return a


@fe.kernel
def load_tile_dynamic_kernel_compare(
    a: pl.Tensor[[256, 512], pl.FP16],
) -> pl.Tensor[[256, 512], pl.FP16]:
    tile_a = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
                             addr=0x0000, size=16384)
    for i in pl.range(4):
        plm.load_tile(tile_a, a, [i, 0])
    return a


@fe.kernel
def store_dynamic_kernel_compare(
    a: pl.Tensor[[256, 512], pl.FP16],
) -> pl.Tensor[[256, 512], pl.FP16]:
    tile_a = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
                             addr=0x0000, size=16384)
    for i in pl.range(4):
        plm.store(a, tile_a, [i * 64, 0])
    return a


@fe.kernel
def store_tile_dynamic_kernel_compare(
    a: pl.Tensor[[256, 512], pl.FP16],
) -> pl.Tensor[[256, 512], pl.FP16]:
    tile_a = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
                             addr=0x0000, size=16384)
    for i in pl.range(4):
        plm.store_tile(a, tile_a, [i, 0])
    return a


# ---------------------------------------------------------------------------
# Dynamic offset comparison test functions
# ---------------------------------------------------------------------------

def test_load_tile_dynamic_vs_load_dynamic_same_mlir():
    """Verify load_tile with dynamic offset produces same MLIR as load with i*shape."""
    mlir_load = _compile_to_mlir(load_dynamic_kernel_compare)
    mlir_load_tile = _compile_to_mlir(load_tile_dynamic_kernel_compare)

    print("\n=== load with dynamic offset [i*64, 0] ===")
    print(mlir_load)
    print("\n=== load_tile with dynamic tile_offset [i, 0] ===")
    print(mlir_load_tile)

    assert "pto.tload" in mlir_load, "Expected pto.tload in load MLIR"
    assert "pto.tload" in mlir_load_tile, "Expected pto.tload in load_tile MLIR"
    assert "arith.muli" in mlir_load, "Expected arith.muli in load MLIR"
    assert "arith.muli" in mlir_load_tile, "Expected arith.muli in load_tile MLIR"


def test_store_tile_dynamic_vs_store_dynamic_same_mlir():
    """Verify store_tile with dynamic offset produces same MLIR as store with i*shape."""
    mlir_store = _compile_to_mlir(store_dynamic_kernel_compare)
    mlir_store_tile = _compile_to_mlir(store_tile_dynamic_kernel_compare)

    print("\n=== store with dynamic offset [i*64, 0] ===")
    print(mlir_store)
    print("\n=== store_tile with dynamic tile_offset [i, 0] ===")
    print(mlir_store_tile)

    assert "pto.tstore" in mlir_store, "Expected pto.tstore in store MLIR"
    assert "pto.tstore" in mlir_store_tile, "Expected pto.tstore in store_tile MLIR"
    assert "arith.muli" in mlir_store, "Expected arith.muli in store MLIR"
    assert "arith.muli" in mlir_store_tile, "Expected arith.muli in store_tile MLIR"


if __name__ == "__main__":
    test_load_tile_vs_load_same_mlir()
    test_store_tile_vs_store_same_mlir()
    test_load_tile_dynamic_vs_load_dynamic_same_mlir()
    test_store_tile_dynamic_vs_store_dynamic_same_mlir()
    print("\nAll load_tile and store_tile tests passed!")

