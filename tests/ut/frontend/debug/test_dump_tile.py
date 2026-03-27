# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime validation example for plm.dump_tile."""

import torch
import torch_npu

import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm


@fe.kernel
def dump_tile_add_kernel(
    x: pl.Tensor[[64, 32], pl.INT32],
    y: pl.Tensor[[64, 32], pl.INT32],
    z: pl.Tensor[[64, 32], pl.INT32],
) -> pl.Tensor[[64, 32], pl.INT32]:
    tile_type = plm.TileType(
        shape=[16, 32],
        dtype=pl.INT32,
        target_memory=pl.MemorySpace.Vec,
    )
    tile_a = plm.make_tile(tile_type, addr=0x0000, size=2048)
    tile_b = plm.make_tile(tile_type, addr=0x0800, size=2048)
    tile_c = plm.make_tile(tile_type, addr=0x1000, size=2048)

    with pl.section_vector():
        b_idx = pl.block.get_block_idx()
        block_idx = pl.block.index_cast(b_idx)
        b_num = pl.block.get_block_num()
        block_num = pl.block.index_cast(b_num)
        num_tiles_s0 = 4

        for s0_tile in pl.range(block_idx, num_tiles_s0, block_num):
            offset = s0_tile * 16
            plm.load(tile_a, x, [offset, 0])
            plm.load(tile_b, y, [offset, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            plm.add(tile_c, tile_a, tile_b)
            plm.dump_tile(tile_c)
            plm.dump_tile(tile_c, offsets=[4, 0], shapes=[8, 8])
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            plm.store(z, tile_c, [offset, 0])
            pl.system.bar_all()

    return z


@fe.jit()
def test_dump_tile():
    compiled_lib = fe.compile(dump_tile_add_kernel, arch="a3")
    print("compiled lib path:", compiled_lib.lib_path)

    device = "npu:0"
    torch.npu.set_device(device)

    x = torch.arange(64 * 32, device=device, dtype=torch.int32).reshape(64, 32)
    y = x
    z = torch.empty_like(x)

    fe.launch(None, 4, compiled_lib, x, y, z)
    torch.npu.synchronize()

    print("***********npu output***********")
    print(z.shape, z.dtype)
    print(z)

    z_ref = x + y
    print("***********golden output***********")
    print(z_ref.shape, z_ref.dtype)
    print(z_ref)

    torch.testing.assert_close(z, z_ref)
    print("result equal!")


if __name__ == "__main__":
    test_dump_tile()
    print("\nAll tests passed!")
