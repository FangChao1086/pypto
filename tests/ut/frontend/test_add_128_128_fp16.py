# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend tests for the @fe.kernel decorator with manual (non-SSA) plm.* ops.

Each test kernel is compiled to PTO MLIR and the output is checked for
correct pto.alloc_tile / pto.tload / pto.tadd / pto.tmul patterns.
"""

import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm


# ---------------------------------------------------------------------------
# Test kernels — defined at module level so @fe.kernel runs at import time
# ---------------------------------------------------------------------------

# Kernel: load two tiles and add them
@fe.kernel
def add_kernel_128(
    x: pl.Tensor[[128, 128], pl.FP16],
    y: pl.Tensor[[128, 128], pl.FP16],
    z: pl.Tensor[[128, 128], pl.FP16]
) -> pl.Tensor[[128, 128], pl.FP16]:
    tile_a = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
                             addr=0x0000, size=16384)
    tile_b = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
                             addr=0x0000, size=16384)
    tile_c = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
                             addr=0x0000, size=16384)
    with pl.section_vector():
        vidx = pl.block.get_block_idx()
        vidx_i = pl.block.index_cast(vidx)  # cast i64 to index
        offset = vidx_i * 64
        plm.load(tile_a, x, [offset, 0])

        plm.load(tile_b, y, [offset, 0])

        plm.add(tile_c, tile_a, tile_b)

        plm.store(z, tile_c, [offset, 0])
    return z


# ---------------------------------------------------------------------------
# Test functions (run with: python test_kernel.py)
# ---------------------------------------------------------------------------

@fe.jit()
def test_add():
    device = "npu"
    torch.npu.set_device(device)

    shape = [128, 128]  # tensor shape hard-coded as the kernel
    torch.manual_seed(0)
    dtype = torch.float16
    x = torch.rand(shape, device=device, dtype=dtype)
    y = torch.rand(shape, device=device, dtype=dtype)
    z = torch.empty(shape, device=device, dtype=dtype)

    compiled_lib = fe.compile(add_kernel_128)
    print("compiled lib path:", compiled_lib.lib_path)
    fe.launch(None, 2, compiled_lib, x, y, z)

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
    test_add()
    print("\nAll tests passed!")
