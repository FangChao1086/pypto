# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under terms and conditions
# of CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not use this file except in compliance with License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend tests for the @fe.kernel decorator with manual (non-SSA) plm.* ops.

Each test kernel is compiled to PTO MLIR and the output is checked for
correct pto.alloc_tile / pto.tload / pto.tgather patterns.
"""

import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm


# ---------------------------------------------------------------------------
# Test kernels — defined at module level so @fe.kernel runs at import time
# ---------------------------------------------------------------------------

# Kernel: load source, indices, and tmp tiles, gather elements (with tmp)
@fe.kernel
def gather_kernel_with_tmp(
    src: pl.Tensor[[64, 128], pl.FP16],
    indices: pl.Tensor[[64, 128], pl.INT32],
    dst: pl.Tensor[[64, 128], pl.FP16]
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_src = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
                                 addr=0x0000, size=16384)
    tile_idx = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec),
                                 addr=0x4000, size=32768)
    tile_tmp = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec),
                                 addr=0xC000, size=32768)
    tile_dst = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
                                 addr=0x14000, size=16384)
    with pl.section_vector():
        plm.load(tile_src, src, [0, 0])
        plm.load(tile_idx, indices, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        plm.gather(tile_dst, tile_src, tile_idx, tile_tmp)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        plm.store(dst, tile_dst, [0, 0])
    return dst

# ---------------------------------------------------------------------------
# Test functions (run with: python test_kernel.py)
# ---------------------------------------------------------------------------

@fe.jit()
def test_gather_with_tmp():
    device = "npu:1"
    torch.npu.set_device(device)
    shape = [64, 128]  # tensor shape hard-coded as the kernel
    torch.manual_seed(0)
    dtype = torch.float16
    idx_dtype = torch.int32

    # Create source data
    src = torch.rand(shape, device=device, dtype=dtype)

    # Create indices (clamped to valid range)
    indices = torch.randint(0, shape[0] * shape[1], shape, device=device, dtype=idx_dtype)
    indices = torch.clamp(indices, 0, shape[0] * shape[1] - 1)

    dst = torch.empty(shape, device=device, dtype=dtype)

    compiled_lib = fe.compile(gather_kernel_with_tmp, arch="a3")
    print("compiled lib path:", compiled_lib.lib_path)
    fe.launch(None, 1, compiled_lib, src, indices, dst)

    torch.npu.synchronize()

    print("***********npu output***********")
    print(dst.shape, dst.dtype)
    print(dst)
    
    # Compute golden output
    src_flat = src.flatten()
    indices_flat = indices.flatten()
    dst_flat = torch.empty_like(src_flat)
    for i in range(len(indices_flat)):
        idx = indices_flat[i].item()
        dst_flat[i] = src_flat[idx]
    dst_ref = dst_flat.reshape(shape)
    
    print("***********golden output***********")
    print(dst_ref.shape, dst_ref.dtype)
    print(dst_ref)
    torch.testing.assert_close(dst, dst_ref)
    print("result equal!")


if __name__ == "__main__":
    test_gather_with_tmp()
    print("\nAll tests passed!")
