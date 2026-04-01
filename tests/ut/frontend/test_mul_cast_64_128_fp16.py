# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under terms and conditions
# of CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend tests for the @fe.kernel decorator with manual (non-SSA) plm.* ops.

Each test kernel is compiled to PTO MLIR and the output is checked for
correct pto.alloc_tile / pto.tload / pto.tmul / pto.tcvt patterns.
"""

import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm


# ---------------------------------------------------------------------------
# Test kernels — defined at module level so @fe.kernel runs at import time
# ---------------------------------------------------------------------------

# Kernel: load two tiles, multiply them, and cast result
@fe.kernel
def mul_cast_kernel(
    x: pl.Tensor[[64, 128], pl.FP16],
    y: pl.Tensor[[64, 128], pl.FP16],
    z: pl.Tensor[[64, 128], pl.FP32]
) -> pl.Tensor[[64, 128], pl.FP32]:
    tile_a = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
                             addr=0x0000, size=16384)
    tile_b = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
                             addr=0x4000, size=16384)
    tile_c = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec),
                             addr=0x8000, size=32768)
    with pl.section_vector():
        plm.load(tile_a, x, [0, 0])
        plm.load(tile_b, y, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        plm.mul_cast(tile_c, tile_a, tile_b, target_type=pl.FP32, mode="round")
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        plm.store(z, tile_c, [0, 0])
    return z


# ---------------------------------------------------------------------------
# Test functions (run with: python test_kernel.py)
# ---------------------------------------------------------------------------

@fe.jit()
def test_mul_cast():
    device = "npu:1"
    torch.npu.set_device(device)

    shape = [64, 128]  # tensor shape hard-coded as the kernel
    torch.manual_seed(0)
    dtype_in = torch.float16
    dtype_out = torch.float32
    x = torch.rand(shape, device=device, dtype=dtype_in)
    y = torch.rand(shape, device=device, dtype=dtype_in)
    z = torch.empty(shape, device=device, dtype=dtype_out)

    compiled_lib = fe.compile(mul_cast_kernel, arch="a3")
    print("compiled lib path:", compiled_lib.lib_path)
    fe.launch(None, 1, compiled_lib, x, y, z)

    torch.npu.synchronize()

    print("***********npu output***********")
    print(z.shape, z.dtype)
    print(z)
    z_mul = x * y
    z_ref = z_mul.to(dtype_out)
    print("***********golden output***********")
    print(z_ref.shape, z_ref.dtype)
    print(z_ref)
    torch.testing.assert_close(z, z_ref)
    print("result equal!")


if __name__ == "__main__":
    test_mul_cast()
    print("\nAll tests passed!")
