# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Dynamic matmul tests with double buffer (ping-pong) optimization.

Double buffer allows overlapping compute and memory transfer:
- While computing on ping buffers, load data into pong buffers
- While computing on pong buffers, load data into ping buffers
This hides memory latency behind computation.

This version uses tuple variable-index access (tile_buf[buf_idx]) to unify ping/pong
dispatch, replacing the manual if/else branch.  Event-id assignment follows the
hardware rule: each unique (set_pipe, wait_pipe) combination has its own independent
event_id namespace (0–7), so ping uses event_id=0 and pong uses event_id=1 within
every such combination independently.
"""

import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm


M = pl.DynVar('M')
K = pl.DynVar('K')
N = pl.DynVar('N')

@fe.kernel
def dynamic_matmul_db_kernel(
    a: pl.Tensor[[M, K], pl.FP16],
    b: pl.Tensor[[K, N], pl.FP16],
    c: pl.Tensor[[M, N], pl.FP32]
) -> pl.Tensor[[M, N], pl.FP32]:
    # ========== Load buffers (Mat space) — shared TileType for ping and pong ==========
    tile_type_a_load = plm.TileType(
        shape=[128, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Mat,
        blayout=2,
        slayout=1,
    )
    tile_a_load_ping = plm.make_tile(tile_type_a_load, addr=0x00000, size=32768)
    tile_a_load_pong = plm.make_tile(tile_type_a_load, addr=0x10000, size=32768)

    tile_type_b_load = plm.TileType(
        shape=[128, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Mat,
        blayout=2,
        slayout=1,
    )
    tile_b_load_ping = plm.make_tile(tile_type_b_load, addr=0x08000, size=32768)
    tile_b_load_pong = plm.make_tile(tile_type_b_load, addr=0x18000, size=32768)

    # ========== Compute buffers (Left / Right space) ==========
    tile_type_a_compute = plm.TileType(
        shape=[128, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Left,
        blayout=1,
        slayout=1,
    )
    tile_a_ping = plm.make_tile(tile_type_a_compute, addr=0x00000, size=32768)
    tile_a_pong = plm.make_tile(tile_type_a_compute, addr=0x08000, size=32768)

    tile_type_b_compute = plm.TileType(
        shape=[128, 128],
        dtype=pl.FP16,
        target_memory=pl.MemorySpace.Right,
        blayout=1,
        slayout=2,
    )
    tile_b_ping = plm.make_tile(tile_type_b_compute, addr=0x00000, size=32768)
    tile_b_pong = plm.make_tile(tile_type_b_compute, addr=0x08000, size=32768)

    # ========== Accumulator (shared) ==========
    tile_type_c = plm.TileType(
        shape=[128, 128],
        dtype=pl.FP32,
        target_memory=pl.MemorySpace.Acc,
        blayout=2,
        slayout=1,
        fractal=1024,
        valid_shape=[-1, -1],
    )
    tile_c = plm.make_tile(tile_type_c, addr=0x00000, size=65536)

    # ========== Tuple pairs for variable-index double-buffer dispatch ==========
    # All elements in each tuple share the same TileType, satisfying the
    # homogeneous-element requirement of the variable-index feature.
    tile_a_load_buf = (tile_a_load_ping, tile_a_load_pong)
    tile_b_load_buf = (tile_b_load_ping, tile_b_load_pong)
    tile_a_buf = (tile_a_ping, tile_a_pong)
    tile_b_buf = (tile_b_ping, tile_b_pong)

    with pl.section_cube():
        M_dim = pl.tensor.dim(a, 0)
        K_dim = pl.tensor.dim(a, 1)
        N_dim = pl.tensor.dim(b, 1)

        # Event-id tuples: index 0 = ping, index 1 = pong.
        # Different (set_pipe, wait_pipe) combinations have independent event_id
        # namespaces (0–7 each), so the same values 0/1 are reused across combos.
        event_ids = (0, 1)

        for i in pl.range(0, M_dim, 128):
            for j in pl.range(0, N_dim, 128):
                for k in pl.range(0, K_dim, 128):
                    # buf_idx cycles 0 (ping) / 1 (pong) each k-iteration.
                    # Both tile access and event_id selection use the same index,
                    # lowered to an if-else chain by the variable-index feature.
                    buf_idx = (k // 128) % 2

                    plm.load(tile_a_load_buf[buf_idx], a, [i, k])
                    plm.load(tile_b_load_buf[buf_idx], b, [k, j])

                    # (MTE2, MTE1) — load done, move can start
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=event_ids[buf_idx])
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=event_ids[buf_idx])

                    plm.move(tile_a_buf[buf_idx], tile_a_load_buf[buf_idx], target_memory=pl.MemorySpace.Left)
                    plm.move(tile_b_buf[buf_idx], tile_b_load_buf[buf_idx], target_memory=pl.MemorySpace.Right)

                    # (MTE1, M) — move done, matmul can start
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=event_ids[buf_idx])
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=event_ids[buf_idx])

                    if k == 0:
                        plm.matmul(tile_c, tile_a_buf[buf_idx], tile_b_buf[buf_idx])
                    else:
                        plm.matmul_acc(tile_c, tile_c, tile_a_buf[buf_idx], tile_b_buf[buf_idx])

                    # (M, MTE2) — matmul done, next iteration's load can start
                    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE2, event_id=event_ids[buf_idx])
                    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE2, event_id=event_ids[buf_idx])

                # Store result after all k-tiles are accumulated
                pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
                plm.l0c_store(tile_c, [i, j], [128, 128], c)

                # Barrier to ensure l0c_store is complete before next (i, j) iteration
                pl.system.bar_all()

    return c


@fe.jit()
def test_dynamic_matmul_db():
    compiled_lib = fe.compile(dynamic_matmul_db_kernel, arch="dav-c220-cube")
    print("compiled lib path:", compiled_lib.lib_path)

    device = "npu:1"
    torch.npu.set_device(device)

    shapes = [
        [128, 512, 128],
        [128, 512, 256],
        [256, 512, 256],
        [256, 512, 512],
        [512, 512, 512],
        [2048, 128, 2048],
    ]
    torch.manual_seed(0)

    for M_val, K_val, N_val in shapes:
        print(f"\nTesting shape: ({M_val}, {K_val}) x ({K_val}, {N_val}) = ({M_val}, {N_val})")

        a = torch.randn(M_val, K_val, dtype=torch.float16, device=device)
        b = torch.randn(K_val, N_val, dtype=torch.float16, device=device)
        c = torch.zeros(M_val, N_val, dtype=torch.float32, device=device)

        fe.launch(None, 1, compiled_lib, a, b, c)
        torch.npu.synchronize()

        print("***********npu output***********")
        print(c.shape, c.dtype)
        # print(c)
        c_ref = torch.matmul(a.float(), b.float())
        print("***********golden output***********")
        print(c_ref.shape, c_ref.dtype)
        # print(c_ref)

        torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=1e-2)
        print("result equal!")


if __name__ == "__main__":
    test_dynamic_matmul_db()
    print("\nAll tests passed!")
