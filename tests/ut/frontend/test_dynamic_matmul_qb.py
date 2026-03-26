# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dynamic matmul with quad buffer (4-way rotation) optimization.

Quad buffer extends double buffer by using 4 separate buffer slots for the
load (MAT) and compute (LEFT/RIGHT) stages.  This allows deeper pipelining:
up to 3 loads can be in flight while one slot is being computed.

Tile sizes are 64x64 to fit 4 slots within the memory budget:
  - MAT:   512 KB → 4 × A(64×64×2B=8KB) + 4 × B(64×64×2B=8KB) = 64 KB ✓
  - LEFT:  64 KB  → 4 × 8 KB = 32 KB ✓
  - RIGHT: 64 KB  → 4 × 8 KB = 32 KB ✓
  - ACC:   128 KB → 1 × (64×64×4B=16KB) = 16 KB ✓

Usage:
    python3 tests/ut/frontend/test_dynamic_matmul_qb.py
"""

import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm


M = pl.DynVar('M')
K = pl.DynVar('K')
N = pl.DynVar('N')

TILE = 64
N_SLOTS = 4
TILE_BYTES_F16 = TILE * TILE * 2   # 8192 = 8 KB


@fe.kernel(auto_sync=True)
def dynamic_matmul_qb_kernel(
    a: pl.Tensor[[M, K], pl.FP16],
    b: pl.Tensor[[K, N], pl.FP16],
    c: pl.Tensor[[M, N], pl.FP32],
) -> pl.Tensor[[M, N], pl.FP32]:
    # ========== Load buffers (Mat space) — 4 slots each ==========
    mat_type_a = plm.TileType(
        shape=[TILE, TILE], dtype=pl.FP16,
        target_memory=pl.MemorySpace.Mat, blayout=2, slayout=1,
    )
    a_mat_0 = plm.make_tile(mat_type_a, addr=0x00000, size=TILE_BYTES_F16)
    a_mat_1 = plm.make_tile(mat_type_a, addr=0x02000, size=TILE_BYTES_F16)
    a_mat_2 = plm.make_tile(mat_type_a, addr=0x04000, size=TILE_BYTES_F16)
    a_mat_3 = plm.make_tile(mat_type_a, addr=0x06000, size=TILE_BYTES_F16)

    mat_type_b = plm.TileType(
        shape=[TILE, TILE], dtype=pl.FP16,
        target_memory=pl.MemorySpace.Mat, blayout=2, slayout=1,
    )
    b_mat_0 = plm.make_tile(mat_type_b, addr=0x08000, size=TILE_BYTES_F16)
    b_mat_1 = plm.make_tile(mat_type_b, addr=0x0A000, size=TILE_BYTES_F16)
    b_mat_2 = plm.make_tile(mat_type_b, addr=0x0C000, size=TILE_BYTES_F16)
    b_mat_3 = plm.make_tile(mat_type_b, addr=0x0E000, size=TILE_BYTES_F16)

    # ========== Compute buffers (Left / Right space) — 4 slots each ==========
    left_type = plm.TileType(
        shape=[TILE, TILE], dtype=pl.FP16,
        target_memory=pl.MemorySpace.Left, blayout=1, slayout=1,
    )
    a_left_0 = plm.make_tile(left_type, addr=0x00000, size=TILE_BYTES_F16)
    a_left_1 = plm.make_tile(left_type, addr=0x02000, size=TILE_BYTES_F16)
    a_left_2 = plm.make_tile(left_type, addr=0x04000, size=TILE_BYTES_F16)
    a_left_3 = plm.make_tile(left_type, addr=0x06000, size=TILE_BYTES_F16)

    right_type = plm.TileType(
        shape=[TILE, TILE], dtype=pl.FP16,
        target_memory=pl.MemorySpace.Right, blayout=1, slayout=2,
    )
    b_right_0 = plm.make_tile(right_type, addr=0x00000, size=TILE_BYTES_F16)
    b_right_1 = plm.make_tile(right_type, addr=0x02000, size=TILE_BYTES_F16)
    b_right_2 = plm.make_tile(right_type, addr=0x04000, size=TILE_BYTES_F16)
    b_right_3 = plm.make_tile(right_type, addr=0x06000, size=TILE_BYTES_F16)

    # ========== Accumulator (shared, single tile) ==========
    acc_type = plm.TileType(
        shape=[TILE, TILE], dtype=pl.FP32,
        target_memory=pl.MemorySpace.Acc, blayout=2, slayout=1,
        fractal=1024, valid_shape=[-1, -1],
    )
    tile_c = plm.make_tile(acc_type, addr=0x00000, size=16384)

    # ========== 4-element tuples for variable-index quad-buffer dispatch ==========
    a_mat_buf = (a_mat_0, a_mat_1, a_mat_2, a_mat_3)
    b_mat_buf = (b_mat_0, b_mat_1, b_mat_2, b_mat_3)
    a_left_buf = (a_left_0, a_left_1, a_left_2, a_left_3)
    b_right_buf = (b_right_0, b_right_1, b_right_2, b_right_3)

    with pl.section_cube():
        M_dim = pl.tensor.dim(a, 0)
        K_dim = pl.tensor.dim(a, 1)
        N_dim = pl.tensor.dim(b, 1)

        for i in pl.range(0, M_dim, TILE):
            for j in pl.range(0, N_dim, TILE):
                for k in pl.range(0, K_dim, TILE):
                    buf_idx = (k // TILE) % N_SLOTS
                    plm.load(a_mat_buf[buf_idx], a, [i, k])
                    plm.load(b_mat_buf[buf_idx], b, [k, j])
                    plm.move(a_left_buf[buf_idx], a_mat_buf[buf_idx])
                    plm.move(b_right_buf[buf_idx], b_mat_buf[buf_idx])
                    if k == 0:
                        plm.matmul(tile_c, a_left_buf[buf_idx], b_right_buf[buf_idx])
                    else:
                        plm.matmul_acc(tile_c, tile_c, a_left_buf[buf_idx], b_right_buf[buf_idx])
                plm.l0c_store(tile_c, [i, j], [TILE, TILE], c)

    return c


@fe.jit()
def test_dynamic_matmul_qb():
    compiled_lib = fe.compile(dynamic_matmul_qb_kernel, arch="a3")
    print("compiled lib path:", compiled_lib.lib_path)

    device = "npu:5"
    torch.npu.set_device(device)

    shapes = [
        [64, 256, 64],
        [256, 512, 256],
        [128, 256, 128],
        [128, 512, 256],
        [256, 256, 256],
        [512, 256, 512],
    ]
    torch.manual_seed(0)

    for M_val, K_val, N_val in shapes:
        print(f"\nTesting shape: ({M_val}, {K_val}) x ({K_val}, {N_val}) = ({M_val}, {N_val})")

        a = torch.randn(M_val, K_val, dtype=torch.float16, device=device)
        b = torch.randn(K_val, N_val, dtype=torch.float16, device=device)
        c = torch.zeros(M_val, N_val, dtype=torch.float32, device=device)

        fe.launch(None, 1, compiled_lib, a, b, c)
        torch.npu.synchronize()

        c_ref = torch.matmul(a.float(), b.float())
        torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=1e-2)
        print(f"  PASS")


if __name__ == "__main__":
    test_dynamic_matmul_qb()
    print("\nAll tests passed!")
