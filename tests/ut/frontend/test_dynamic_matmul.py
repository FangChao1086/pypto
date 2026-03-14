"""Dynamic matmul tests for the @fe.kernel decorator with manual (non-SSA) plm.* ops.

Reference: PTOAS/test/samples/MatMul/tmatmulk.pto
"""

import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm


M = pl.DynVar('M')
K = pl.DynVar('K')
N = pl.DynVar('N')

@fe.kernel
def dynamic_matmul_kernel(
    a: pl.Tensor[[M, K], pl.FP32],
    b: pl.Tensor[[K, N], pl.FP32],
    c: pl.Tensor[[M, N], pl.FP32]
) -> pl.Tensor[[M, N], pl.FP32]:
    tile_type_a_load = plm.TileType(
        shape=[32, 32],
        dtype=pl.FP32,
        target_memory=pl.MemorySpace.Mat,
        blayout=2,
        slayout=1,
    )
    tile_a_load = plm.make_tile(tile_type_a_load, addr=0x0000, size=4096)
    
    tile_type_b_load = plm.TileType(
        shape=[32, 32],
        dtype=pl.FP32,
        target_memory=pl.MemorySpace.Mat,
        blayout=2,
        slayout=1,
    )
    tile_b_load = plm.make_tile(tile_type_b_load, addr=0x1000, size=4096)
    
    tile_type_a = plm.TileType(
        shape=[32, 32],
        dtype=pl.FP32,
        target_memory=pl.MemorySpace.Left,
        blayout=1,
        slayout=1,
    )
    tile_a = plm.make_tile(tile_type_a, addr=0x2000, size=4096)
    
    tile_type_b = plm.TileType(
        shape=[32, 32],
        dtype=pl.FP32,
        target_memory=pl.MemorySpace.Right,
        blayout=1,
        slayout=2,
    )
    tile_b = plm.make_tile(tile_type_b, addr=0x3000, size=4096)
    
    tile_type_c = plm.TileType(
        shape=[32, 32],
        dtype=pl.FP32,
        target_memory=pl.MemorySpace.Acc,
        blayout=2,
        slayout=1,
        fractal=1024,
        valid_shape=[-1, -1],
    )
    tile_c = plm.make_tile(tile_type_c, addr=0x4000, size=4096)
    
    with pl.section_cube():
        M_dim = pl.tensor.dim(a, 0)
        K_dim = pl.tensor.dim(a, 1)
        N_dim = pl.tensor.dim(b, 1)
        
        for i in pl.range(0, M_dim, 32):
            for j in pl.range(0, N_dim, 32):
                for k in pl.range(0, K_dim, 32):
                    plm.load(tile_a_load, a, [i, k])
                    plm.load(tile_b_load, b, [k, j])
                    
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
                    
                    plm.move(tile_a, tile_a_load, target_memory=pl.MemorySpace.Left)
                    plm.move(tile_b, tile_b_load, target_memory=pl.MemorySpace.Right)
                    
                    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
                    
                    if k == 0:
                        plm.matmul(tile_c, tile_a, tile_b)
                    else:
                        plm.matmul_acc(tile_c, tile_c, tile_a, tile_b)
                    
                    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE2, event_id=0)
                    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE2, event_id=0)
                
                pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
                plm.l0c_store(tile_c, [i, j], [32, 32], c)

    return c


@fe.jit()
def test_dynamic_matmul():
    compiled_lib = fe.compile(dynamic_matmul_kernel, arch="dav-c220-cube")
    print("compiled lib path:", compiled_lib.lib_path)

    device = "npu:1"
    torch.npu.set_device(device)

    shapes = [
        [32, 32, 32],
        [64, 64, 64],
    ]
    torch.manual_seed(0)
    dtype = torch.float32

    for M_val, K_val, N_val in shapes:
        print(f"\nTesting shape: ({M_val}, {K_val}) x ({K_val}, {N_val}) = ({M_val}, {N_val})")
        
        a = torch.randn(M_val, K_val, dtype=dtype, device=device)
        b = torch.randn(K_val, N_val, dtype=dtype, device=device)
        c = torch.zeros(M_val, N_val, dtype=dtype, device=device)
        
        fe.launch(None, 1, compiled_lib, a, b, c)
        torch.npu.synchronize()
        
        print("***********npu output***********")
        print(c.shape, c.dtype)
        print(c)
        c_ref = torch.matmul(a, b)
        print("***********golden output***********")
        print(c_ref.shape, c_ref.dtype)
        print(c_ref)
        
        torch.testing.assert_close(c, c_ref, rtol=1e-2, atol=1e-2)
        print("result equal!")


if __name__ == "__main__":
    test_dynamic_matmul()
    print("\nAll tests passed!")
