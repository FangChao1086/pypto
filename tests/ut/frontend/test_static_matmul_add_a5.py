# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Static fused matmul + add test with manual synchronization."""

import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm


@fe.kernel(auto_sync=False)
def static_matmul_add_kernel(
    a: pl.Tensor[[64, 64], pl.FP32],
    b: pl.Tensor[[64, 64], pl.FP32],
    addend: pl.Tensor[[64, 64], pl.FP32],
    # matmul_out: pl.Tensor[[64, 64], pl.FP32],
    out: pl.Tensor[[64, 64], pl.FP32],
) -> pl.Tensor[[64, 64], pl.FP32]:
    tile_type_p = plm.TileType(shape=[32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    tile_p = plm.make_tile(tile_type_p, addr=0x0000, size=8192)

    with pl.section_cube():
        tile_type_a_load = plm.TileType(
            shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Mat, blayout=2, slayout=1)
        tile_a_load = plm.make_tile(tile_type_a_load, addr=0x0000, size=16384)

        tile_type_b_load = plm.TileType(
            shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Mat, blayout=2, slayout=1)
        tile_b_load = plm.make_tile(tile_type_b_load, addr=0x4000, size=16384)

        tile_type_a = plm.TileType(
            shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Left, blayout=2, slayout=1)
        tile_a = plm.make_tile(tile_type_a, addr=0x0000, size=16384)

        tile_type_b = plm.TileType(
            shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Right, blayout=1, slayout=2)
        tile_b = plm.make_tile(tile_type_b, addr=0x0000, size=16384)

        tile_type_c = plm.TileType(
            shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc, blayout=2, slayout=1, fractal=1024)
        tile_c = plm.make_tile(tile_type_c, addr=0x0000, size=16384)

        plm.load(tile_a_load, a, [0, 0])
        plm.load(tile_b_load, b, [0, 0])

        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

        plm.move(tile_a, tile_a_load)
        plm.move(tile_b, tile_b_load)

        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)

        plm.matmul(tile_c, tile_a, tile_b)

        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        plm.move(tile_p, tile_c, acc_to_vec_mode="dual_split_m")  # ACC -> UB
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=1)
        # pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=17)

    with pl.section_vector():
        sub_core = pl.block.get_subblock_idx()
        sub_index = pl.block.index_cast(sub_core)
        tile_type_addend = plm.TileType(shape=[32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        tile_addend = plm.make_tile(tile_type_addend, addr=0x2000, size=8192)

        tile_type_out = plm.TileType(shape=[32, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        tile_out = plm.make_tile(tile_type_out, addr=0x4000, size=8192)

        off = sub_index * 32
        plm.load(tile_addend, addend, [off, 0])

        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=1)

        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=1)

        plm.add(tile_out, tile_p, tile_addend)

        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=2)
        plm.store(out, tile_out, [off, 0])

    return out


@fe.jit()
def test_static_matmul_add():
    compiled_lib = fe.compile(static_matmul_add_kernel, arch="a5", codegen_mode="cce")
    print("compiled lib path:", compiled_lib.lib_path)

    device = "npu:0"
    torch.npu.set_device(device)

    shape = [64, 64]
    torch.manual_seed(0)
    dtype = torch.float32

    a = torch.randn(shape, device=device, dtype=dtype)
    b = torch.randn(shape, device=device, dtype=dtype)
    addend = torch.randn(shape, device=device, dtype=dtype)
    out = torch.zeros(shape, device=device, dtype=dtype)

    fe.launch(None, 1, compiled_lib, a, b, addend, out)
    torch.npu.synchronize()

    matmul_ref = torch.matmul(a, b)
    print("***********golden matmul output***********")
    print(matmul_ref.shape, matmul_ref.dtype)
    print(matmul_ref)

    print("***********npu output***********")
    print(out.shape, out.dtype)
    print(out)
    out_ref = matmul_ref + addend
    print("***********golden output***********")
    print(out_ref.shape, out_ref.dtype)
    print(out_ref)

    # torch.testing.assert_close(matmul_out, matmul_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
    print("result equal!")


if __name__ == "__main__":
    test_static_matmul_add()
    print("\nAll tests passed!")
