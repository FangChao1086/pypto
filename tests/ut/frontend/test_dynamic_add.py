# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Dynamic add tests for the @fe.kernel decorator with manual (non-SSA) plm.* ops."""

import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm


M = pl.DynVar('M')
N = pl.DynVar('N')

@fe.kernel
def dynamic_add_kernel(
    x: pl.Tensor[[M, N], pl.FP16],
    y: pl.Tensor[[M, N], pl.FP16],
    z: pl.Tensor[[M, N], pl.FP16]
) -> pl.Tensor[[M, N], pl.FP16]:
    tile_a = plm.create_tile([64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec,
                             addr=0x0000, size=16384)
    tile_b = plm.create_tile([64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec,
                             addr=0x0000, size=16384)
    tile_c = plm.create_tile([64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec,
                             addr=0x0000, size=16384)
    with pl.section_vector():
        m_loops = (M + 63) // 64
        n_loops = (N + 127) // 128

        for i in pl.range(m_loops):
            for j in pl.range(n_loops):
                m_offset = i * 64
                n_offset = j * 128

                plm.load(x, [m_offset, n_offset], [64, 128], out=tile_a)
                plm.load(y, [m_offset, n_offset], [64, 128], out=tile_b)
                plm.add(tile_a, tile_b, out=tile_c)
                plm.store(tile_c, [m_offset, n_offset], [64, 128], z)

    return z


# ---------------------------------------------------------------------------
# Test functions (run with: python test_kernel.py)
# ---------------------------------------------------------------------------

@fe.jit()
def test_dynamic_add():
    compiled_lib = fe.compile(dynamic_add_kernel)
    print("compiled lib path:", compiled_lib.lib_path)

    device = "npu:1"
    torch.npu.set_device(device)

    shapes = [
        [64, 128],
        [128, 256],
    ]
    torch.manual_seed(0)
    dtype = torch.float16

    for shape in shapes:

        x = torch.rand(shape, device=device, dtype=dtype)
        y = torch.rand(shape, device=device, dtype=dtype)
        z = torch.empty(shape, device=device, dtype=dtype)

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
    test_dynamic_add()
    print("\nAll tests passed!")
