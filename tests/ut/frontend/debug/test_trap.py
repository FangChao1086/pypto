# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Manual frontend runtime demo for plm.trap.

Expected behavior:
- With plm.trap() enabled, output should stop after TRAP_TEST_BEFORE_LAUNCH.
- If plm.trap() is commented out, output should continue through
  TRAP_TEST_AFTER_LAUNCH, TRAP_TEST_BEFORE_CHECK, and finally print equal.
"""

import torch
import torch_npu

import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm


@fe.kernel
def trap_kernel(
    x: pl.Tensor[[128, 64], pl.INT32],
    y: pl.Tensor[[128, 64], pl.INT32],
    z: pl.Tensor[[128, 64], pl.INT32],
    flag: pl.Scalar[pl.BOOL],
) -> pl.Tensor[[128, 64], pl.INT32]:
    tile_type = plm.TileType(shape=[128, 64], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
    tile_x = plm.make_tile(tile_type, addr=0x0000, size=32768)
    tile_y = plm.make_tile(tile_type, addr=0x8000, size=32768)
    tile_z = plm.make_tile(tile_type, addr=0x10000, size=32768)

    with pl.section_vector():
        pl.system.bar_all()
        plm.load(tile_x, x, [0, 0])
        plm.load(tile_y, y, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        plm.sub(tile_z, tile_x, tile_y)
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
        if flag:
            plm.trap()
        plm.store(z, tile_z, [0, 0])
        pl.system.bar_all()

    return z

@fe.jit()
def test_trap() -> None:
    print("TRAP_TEST_BEFORE_COMPILE", flush=True)
    compiled_lib = fe.compile(trap_kernel, arch="a3", enable_print_debug = True)
    print("TRAP_TEST_AFTER_COMPILE", compiled_lib.lib_path, flush=True)

    device = "npu:0"
    torch.npu.set_device(device)

    base = torch.arange(128 * 64, device=device, dtype=torch.int32).reshape(128, 64)
    x = base + 1
    y = base
    z = torch.zeros_like(x)

    print("TRAP_TEST_BEFORE_LAUNCH", flush=True)
    fe.launch(None, 1, compiled_lib, x, y, z, False)
    torch.npu.synchronize()
    print("TRAP_TEST_AFTER_LAUNCH", flush=True)

    print("TRAP_TEST_BEFORE_CHECK", flush=True)
    torch.testing.assert_close(z, torch.ones_like(z))
    print("equal", flush=True)


if __name__ == "__main__":
    test_trap()
    print("\nAll tests passed!")
