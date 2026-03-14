# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tiling op tests: verifies end-to-end tiling arg expansion with add/sub/mul on NPU.

NOTE: This test requires physical NPU hardware (npu:1) and the Ascend toolkit.
Run via the `npu-test` skill (see .claude/skills/npu-test/SKILL.md) rather than
plain `pytest`, as it compiles and executes kernels on real hardware.
"""

from dataclasses import dataclass

import pytest
import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm


@dataclass
class OpTiling:
    placeholder_before_1: int   # padding field before opkind
    placeholder_before_2: int   # padding field before opkind
    opkind: int                 # operation: 0=add, 1=sub, 2=mul
    placeholder_after: int      # padding field after opkind


M = pl.DynVar('M')
N = pl.DynVar('N')


@fe.kernel
def tiling_op_kernel(
    x: pl.Tensor[[M, N], pl.FP16],
    y: pl.Tensor[[M, N], pl.FP16],
    z: pl.Tensor[[M, N], pl.FP16],
    tiling: OpTiling,
) -> pl.Tensor[[M, N], pl.FP16]:
    tile_type = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type, addr=0x0000, size=16384)
    tile_b = plm.make_tile(
        plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x4000,
        size=16384,
    )
    tile_c = plm.make_tile(
        plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
        addr=0x8000,
        size=16384,
    )

    with pl.section_vector():
        M_dim = pl.tensor.dim(x, 0)
        N_dim = pl.tensor.dim(x, 1)
        for i in pl.range(0, M_dim, 64):
            for j in pl.range(0, N_dim, 128):
                pl.system.bar_all()
                plm.load(tile_a, x, [i, j])
                plm.load(tile_b, y, [i, j])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

                if tiling.opkind == 0:
                    plm.add(tile_c, tile_a, tile_b)
                elif tiling.opkind == 1:
                    plm.sub(tile_c, tile_a, tile_b)
                else:
                    plm.mul(tile_c, tile_a, tile_b)

                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                plm.store(z, tile_c, [i, j])
    return z


@fe.jit()
def test_tiling_op():
    compiled_lib = fe.compile(tiling_op_kernel, arch="dav-c220-vec")
    device = "npu:1"
    torch.npu.set_device(device)
    shapes = [[64, 128], [128, 256]]
    torch.manual_seed(0)
    dtype = torch.float16

    op_cases = [
        (0, lambda a, b: a + b, "add"),
        (1, lambda a, b: a - b, "sub"),
        (2, lambda a, b: a * b, "mul"),
    ]

    for shape in shapes:
        x = torch.rand(shape, device=device, dtype=dtype)
        y = torch.rand(shape, device=device, dtype=dtype)

        for opkind, ref_fn, op_name in op_cases:
            z = torch.empty(shape, device=device, dtype=dtype)
            tiling = OpTiling(
                placeholder_before_1=0,
                placeholder_before_2=0,
                opkind=opkind,
                placeholder_after=0,
            )
            fe.launch(None, 1, compiled_lib, x, y, z, tiling)
            torch.npu.synchronize()
            z_ref = ref_fn(x.float(), y.float()).half()
            torch.testing.assert_close(z, z_ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
