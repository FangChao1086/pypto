# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Frontend runtime validation for plm.printf with %d/%i."""

import torch
import torch_npu

import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm


def _run_add_case(compiled_lib, *scalar_args):
    device = "npu:0"
    torch.npu.set_device(device)
    x = torch.arange(128 * 128, device=device, dtype=torch.int32).reshape([128, 128])
    y = x
    z = torch.empty_like(x)

    fe.launch(None, 1, compiled_lib, x, y, z, *scalar_args)
    torch.npu.synchronize()

    z_ref = x + y
    torch.testing.assert_close(z, z_ref)


@fe.kernel
def printf_kernel_di(
    x: pl.Tensor[[128, 128], pl.INT32],
    y: pl.Tensor[[128, 128], pl.INT32],
    z: pl.Tensor[[128, 128], pl.INT32],
    flag_in: pl.Scalar[pl.BOOL],
    value_i8: pl.Scalar[pl.INT8],
    value_i16: pl.Scalar[pl.INT16],
    value_i32: pl.Scalar[pl.INT32],
    value_i64: pl.Scalar[pl.INT64],
) -> pl.Tensor[[128, 128], pl.INT32]:
    tile_type = plm.TileType(shape=[64, 128], dtype=pl.INT32, target_memory=pl.MemorySpace.Vec)
    tile_a = plm.make_tile(tile_type, addr=0x0000, size=32768)
    tile_b = plm.make_tile(tile_type, addr=0x8000, size=32768)
    tile_c = plm.make_tile(tile_type, addr=0x10000, size=32768)

    with pl.section_vector():
        plm.printf("DI_D_PRE flag=%d i8=%d i16=%d i32=%d i64=%d\n", flag_in, value_i8, value_i16, value_i32, value_i64)
        plm.printf("DI_I_PRE flag=%i i8=%i i16=%i i32=%i i64=%i\n", flag_in, value_i8, value_i16, value_i32, value_i64)
        for offset in pl.range(0, 128, 64):
            pl.system.bar_all()
            flag = offset == 0
            plm.printf("DI_D_LOOP off=%d flag=%d\n", offset, flag)
            plm.printf("DI_I_LOOP off=%+08i flag=%i\n", offset, flag)
            plm.load(tile_a, x, [offset, 0])
            plm.load(tile_b, y, [offset, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            plm.add(tile_c, tile_a, tile_b)
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            plm.store(z, tile_c, [offset, 0])

    return z


@fe.jit()
def test_printf_di():
    compiled_lib = fe.compile(printf_kernel_di, arch="a3")
    if compiled_lib is None:
        raise RuntimeError("compile failed for printf_kernel_di")
    _run_add_case(compiled_lib, True, -3, -123, -12345, -123456789)


if __name__ == "__main__":
    test_printf_di()
    print("\nAll tests passed!")
