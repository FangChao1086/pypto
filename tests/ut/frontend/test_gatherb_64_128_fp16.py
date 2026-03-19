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
correct pto.alloc_tile / pto.tload / pto.tgatherb patterns.
"""

import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm


# ---------------------------------------------------------------------------
# Test kernels — defined at module level so @fe.kernel runs at import time
# ---------------------------------------------------------------------------

# Kernel: load source and offset tiles, gather elements using byte offsets
@fe.kernel
def gatherb_kernel(
    src: pl.Tensor[[64, 128], pl.FP16],
    offsets: pl.Tensor[[64, 128], pl.UINT32],
    dst: pl.Tensor[[64, 128], pl.FP16]
) -> pl.Tensor[[64, 128], pl.FP16]:
    tile_src = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
                                 addr=0x0000, size=16384)
    tile_off = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.UINT32, target_memory=pl.MemorySpace.Vec),
                                 addr=0x4000, size=32768)
    tile_dst = plm.make_tile(plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec),
                                 addr=0xC000, size=16384)
    with pl.section_vector():
        plm.load(tile_src, src, [0, 0])
        plm.load(tile_off, offsets, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        plm.gatherb(tile_dst, tile_src, tile_off)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        plm.store(dst, tile_dst, [0, 0])
    return dst


# ---------------------------------------------------------------------------
# Test functions (run with: python test_kernel.py)
# ---------------------------------------------------------------------------

@fe.jit()
def test_gatherb():
    device = "npu:0"
    torch.npu.set_device(device)

    shape = [64, 128]  # tensor shape hard-coded as the kernel
    torch.manual_seed(0)
    dtype = torch.float16
    off_dtype = torch.uint32
    
    # Create source data
    src = torch.rand(shape, device=device, dtype=dtype)
    
    # Create byte offsets (clamped to valid range)
    # Each element is 2 bytes, so offset range is [0, 2*64*128-1]
    max_offset = shape[0] * shape[1] * 2  # FP16 is 2 bytes per element
    offsets = torch.randint(0, max_offset, shape, dtype=off_dtype).to(device)
    
    dst = torch.empty(shape, device=device, dtype=dtype)

    compiled_lib = fe.compile(gatherb_kernel, arch="a3")
    print("compiled lib path:", compiled_lib.lib_path)
    fe.launch(None, 1, compiled_lib, src, offsets, dst)

    torch.npu.synchronize()

    print("***********npu output***********")
    print(dst.shape, dst.dtype)
    print(dst)
    
    # Compute golden output
    # dst[i, j] = src[byte_offsets[i, j] / 2]
    # Since we're using byte offsets, we need to convert to element indices
    src_flat = src.flatten()
    offsets_flat = offsets.flatten()
    dst_flat = torch.empty_like(src_flat)
    for i in range(len(offsets_flat)):
        byte_offset = offsets_flat[i].to(torch.int32).item()
        element_idx = byte_offset // 2  # Convert byte offset to element index
        if element_idx < len(src_flat):
            dst_flat[i] = src_flat[element_idx]
    dst_ref = dst_flat.reshape(shape)
    
    print("***********golden output***********")
    print(dst_ref.shape, dst_ref.dtype)
    print(dst_ref)
    torch.testing.assert_close(dst, dst_ref)
    print("result equal!")


if __name__ == "__main__":
    test_gatherb()
    print("\nAll tests passed!")
