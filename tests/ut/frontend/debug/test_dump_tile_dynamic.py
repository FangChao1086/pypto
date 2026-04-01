# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime validation example for dynamic-signature plm.dump_tile."""

import torch
import torch_npu

import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm


M = pl.DynVar("M")
N = pl.DynVar("N")


@fe.kernel
def dump_tile_dynamic_add_kernel(
    x: pl.Tensor[[M, N], pl.INT32],
    y: pl.Tensor[[M, N], pl.INT32],
    z: pl.Tensor[[M, N], pl.INT32],
) -> pl.Tensor[[M, N], pl.INT32]:
    tile_type = plm.TileType(
        shape=[64, 128],
        dtype=pl.INT32,
        target_memory=pl.MemorySpace.Vec,
        valid_shape=[-1, -1],
    )
    tile_a = plm.make_tile(tile_type, addr=0x0000, size=32768)
    tile_b = plm.make_tile(tile_type, addr=0x8000, size=32768)
    tile_c = plm.make_tile(tile_type, addr=0x10000, size=32768)

    with pl.section_vector():
        m_dim = pl.tensor.dim(x, 0)
        n_dim = pl.tensor.dim(x, 1)
        row_off = m_dim // 4

        plm.set_validshape(tile_a, m_dim, n_dim)
        plm.load(tile_a, x, [0, 0])
        plm.set_validshape(tile_b, m_dim, n_dim)
        plm.load(tile_b, y, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        plm.set_validshape(tile_c, m_dim, n_dim)
        plm.add(tile_c, tile_a, tile_b)
        plm.dump_tile(tile_c)
        plm.dump_tile(tile_c, offsets=[row_off, 0], shapes=[8, 8])
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        plm.store(z, tile_c, [0, 0])

    return z


@fe.jit()
def test_dump_tile_dynamic():
    compiled_lib = fe.compile(dump_tile_dynamic_add_kernel, arch="a3")
    print("compiled lib path:", compiled_lib.lib_path)

    device = "npu:0"
    torch.npu.set_device(device)

    shapes = [
        [64, 128],
    ]

    for m, n in shapes:
        x = torch.arange(m * n, device=device, dtype=torch.int32).reshape(m, n)
        y = x
        z = torch.empty_like(x)

        fe.launch(None, 1, compiled_lib, x, y, z)
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
    test_dump_tile_dynamic()
    print("\nAll tests passed!")
