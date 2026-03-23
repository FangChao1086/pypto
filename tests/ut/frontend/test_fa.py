"""FlashAttention kernels using PyPTO IR manual (non-SSA) mode.

Two versions:
  1. fa_kt_kernel:  Takes pre-transposed K^T[D, Skv]
  2. fa_k_kernel:   Takes K[Skv, D], transposing via move(transpose=True)

Usage:
    python3 tests/ut/frontend/test_fa.py
"""

import math
import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm

TILE = 64
HALF = TILE // 2
SCALE = 1.0 / math.sqrt(TILE)

# Pre-computed memory sizes and addresses
TT2 = TILE * TILE * 2    # 8192
TT4 = TILE * TILE * 4    # 16384
A0 = 0
A1 = 8192                # TT2
A2 = 16384               # TT4
A3 = 24576               # TT4 + TT2

VB4 = HALF * TILE * 4    # 8192  - f32 half-tile bytes
VB2 = HALF * TILE * 2    # 4096  - f16 half-tile bytes
V0 = 0
V1 = 8192
V2 = 16384
V3 = 20480               # V2 + VB2
V4 = 28672
V5 = 36864
V6 = 45056
V7 = 53248

# Cross-core sync event IDs
QK_READY = 0
P_READY = 1
PV_READY = 2

# ================================================================
#  Shared vector section logic (FlashSoftmax + GlobalUpdate)
#  Factored here as comments — must be inlined in each kernel
#  because @pl.inline is not suitable for section_vector bodies.
# ================================================================

# ================================================================
#  Version 1: K^T pre-transposed  (kt: [D, Skv])
# ================================================================

Sq1 = pl.DynVar('Sq')
Skv1 = pl.DynVar('Skv')
D1 = pl.DynVar('D')

@fe.kernel
def fa_kt_kernel(
    q: pl.Tensor[[Sq1, D1], pl.FP16],
    kt: pl.Tensor[[D1, Skv1], pl.FP16],
    v: pl.Tensor[[Skv1, D1], pl.FP16],
    o: pl.Tensor[[Sq1, D1], pl.FP16],
    qk_buf: pl.Tensor[[Sq1, Skv1], pl.FP32],
    p_buf: pl.Tensor[[Sq1, Skv1], pl.FP16],
    pv_buf: pl.Tensor[[Sq1, D1], pl.FP32],
) -> pl.Tensor[[Sq1, D1], pl.FP16]:
    with pl.section_cube():
        skv_dim = pl.tensor.dim(kt, 1)
        skv_tiles = (skv_dim + (TILE - 1)) // TILE
        mat_type = plm.TileType(shape=[TILE, TILE], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat, blayout=2, slayout=1)
        q_mat = plm.make_tile(mat_type, addr=A0, size=TT2)
        k_mat = plm.make_tile(mat_type, addr=A1, size=TT2)
        p_mat = plm.make_tile(mat_type, addr=A2, size=TT2)
        v_mat = plm.make_tile(mat_type, addr=A3, size=TT2)
        q_left = plm.make_tile(plm.TileType(shape=[TILE, TILE], dtype=pl.FP16, target_memory=pl.MemorySpace.Left, blayout=1, slayout=1), addr=A0, size=TT2)
        k_right = plm.make_tile(plm.TileType(shape=[TILE, TILE], dtype=pl.FP16, target_memory=pl.MemorySpace.Right, blayout=1, slayout=2), addr=A0, size=TT2)
        p_left = plm.make_tile(plm.TileType(shape=[TILE, TILE], dtype=pl.FP16, target_memory=pl.MemorySpace.Left, blayout=1, slayout=1), addr=A1, size=TT2)
        v_right = plm.make_tile(plm.TileType(shape=[TILE, TILE], dtype=pl.FP16, target_memory=pl.MemorySpace.Right, blayout=1, slayout=2), addr=A1, size=TT2)
        acc_type = plm.TileType(shape=[TILE, TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc, blayout=2, slayout=1, fractal=1024)
        qk_acc = plm.make_tile(acc_type, addr=A0, size=TT4)
        pv_acc = plm.make_tile(acc_type, addr=A2, size=TT4)

        core_id = pl.block.index_cast(pl.block.get_block_idx())
        sq_off = core_id * TILE

        # Load Q once
        plm.load(q_mat, q, [sq_off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        plm.move(q_left, q_mat, target_memory=pl.MemorySpace.Left)

        # First KV tile
        plm.load(k_mat, kt, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
        plm.move(k_right, k_mat, target_memory=pl.MemorySpace.Right)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        plm.matmul(qk_acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        plm.l0c_store(qk_acc, [sq_off, 0], [TILE, TILE], qk_buf)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY)
        pl.system.wait_cross_core(pipe=pl.PipeType.M, event_id=P_READY)

        # PV_0
        plm.load(p_mat, p_buf, [sq_off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        plm.load(v_mat, v, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
        plm.move(p_left, p_mat, target_memory=pl.MemorySpace.Left)
        plm.move(v_right, v_mat, target_memory=pl.MemorySpace.Right)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        plm.matmul(pv_acc, p_left, v_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        plm.l0c_store(pv_acc, [sq_off, 0], [TILE, TILE], pv_buf)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY)

        # Remaining KV tiles
        for j in pl.range(1, skv_tiles):
            skv_off = j * TILE
            plm.load(k_mat, kt, [0, skv_off])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
            plm.move(k_right, k_mat, target_memory=pl.MemorySpace.Right)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
            plm.matmul(qk_acc, q_left, k_right)
            pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
            plm.l0c_store(qk_acc, [sq_off, skv_off], [TILE, TILE], qk_buf)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY)
            pl.system.wait_cross_core(pipe=pl.PipeType.M, event_id=P_READY)

            plm.load(p_mat, p_buf, [sq_off, skv_off])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
            plm.load(v_mat, v, [skv_off, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
            plm.move(p_left, p_mat, target_memory=pl.MemorySpace.Left)
            plm.move(v_right, v_mat, target_memory=pl.MemorySpace.Right)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
            plm.matmul(pv_acc, p_left, v_right)
            pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
            plm.l0c_store(pv_acc, [sq_off, 0], [TILE, TILE], pv_buf)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY)

    # =================== VECTOR SECTION ===================
    with pl.section_vector():
        skv_dim = pl.tensor.dim(kt, 1)
        skv_tiles = (skv_dim + (TILE - 1)) // TILE
        qk_vec = plm.make_tile(plm.TileType(shape=[HALF, TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=V0, size=VB4)
        tmp_vec = plm.make_tile(plm.TileType(shape=[HALF, TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=V1, size=VB4)
        p_f16 = plm.make_tile(plm.TileType(shape=[HALF, TILE], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec), addr=V2, size=VB2)
        reduce_dst = plm.make_tile(plm.TileType(shape=[HALF, TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[HALF, 1]), addr=V3, size=VB4)
        global_max = plm.make_tile(plm.TileType(shape=[HALF, TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=V4, size=VB4)
        global_sum = plm.make_tile(plm.TileType(shape=[HALF, TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=V5, size=VB4)
        running_o = plm.make_tile(plm.TileType(shape=[HALF, TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=V6, size=VB4)
        exp_corr = plm.make_tile(plm.TileType(shape=[HALF, TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=V7, size=VB4)

        core_id = pl.block.index_cast(pl.block.get_block_idx())
        sq_off = core_id * TILE
        sub_id = pl.block.index_cast(pl.block.get_subblock_idx())
        row_off = sub_id * HALF

        # First KV: FlashSoftmax INIT
        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY)
        plm.load(qk_vec, qk_buf, [sq_off + row_off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        plm.row_max(reduce_dst, qk_vec, tmp_vec)
        plm.row_expand(global_max, reduce_dst)
        plm.sub(tmp_vec, qk_vec, global_max)
        plm.muls(tmp_vec, tmp_vec, SCALE)
        plm.exp(qk_vec, tmp_vec)
        plm.row_sum(reduce_dst, qk_vec, tmp_vec)
        plm.row_expand(global_sum, reduce_dst)
        plm.cast(p_f16, qk_vec, target_type=pl.FP16, mode="round")
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        plm.store(p_buf, p_f16, [sq_off + row_off, 0])
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY)

        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY)
        plm.load(running_o, pv_buf, [sq_off + row_off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

        # Remaining KV: FlashSoftmax UPDATE + GlobalUpdate
        for j in pl.range(1, skv_tiles):
            skv_off = j * TILE
            pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY)
            plm.load(qk_vec, qk_buf, [sq_off + row_off, skv_off])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

            plm.row_max(reduce_dst, qk_vec, tmp_vec)
            plm.row_expand(tmp_vec, reduce_dst)
            plm.maximum(tmp_vec, tmp_vec, global_max)
            plm.sub(exp_corr, global_max, tmp_vec)
            plm.muls(exp_corr, exp_corr, SCALE)
            plm.exp(exp_corr, exp_corr)
            plm.mul(global_sum, global_sum, exp_corr)
            plm.mul(running_o, running_o, exp_corr)
            plm.muls(global_max, tmp_vec, 1.0)
            plm.sub(tmp_vec, qk_vec, global_max)
            plm.muls(tmp_vec, tmp_vec, SCALE)
            plm.exp(qk_vec, tmp_vec)
            plm.row_sum(reduce_dst, qk_vec, tmp_vec)
            plm.row_expand(tmp_vec, reduce_dst)
            plm.add(global_sum, global_sum, tmp_vec)
            plm.cast(p_f16, qk_vec, target_type=pl.FP16, mode="round")
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
            plm.store(p_buf, p_f16, [sq_off + row_off, skv_off])
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY)

            pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY)
            plm.load(qk_vec, pv_buf, [sq_off + row_off, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            plm.add(running_o, running_o, qk_vec)

        plm.div(running_o, running_o, global_sum)
        plm.cast(p_f16, running_o, target_type=pl.FP16, mode="round")
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        plm.store(o, p_f16, [sq_off + row_off, 0])

    return o


# ================================================================
#  Version 2: K[Skv, D] — transpose via move(transpose=True)
# ================================================================

Sq2 = pl.DynVar('Sq')
Skv2 = pl.DynVar('Skv')
D2 = pl.DynVar('D')

@fe.kernel
def fa_k_kernel(
    q: pl.Tensor[[Sq2, D2], pl.FP16],
    k: pl.Tensor[[Skv2, D2], pl.FP16],
    v: pl.Tensor[[Skv2, D2], pl.FP16],
    o: pl.Tensor[[Sq2, D2], pl.FP16],
    qk_buf: pl.Tensor[[Sq2, Skv2], pl.FP32],
    p_buf: pl.Tensor[[Sq2, Skv2], pl.FP16],
    pv_buf: pl.Tensor[[Sq2, D2], pl.FP32],
) -> pl.Tensor[[Sq2, D2], pl.FP16]:
    with pl.section_cube():
        skv_dim = pl.tensor.dim(k, 0)
        skv_tiles = (skv_dim + (TILE - 1)) // TILE
        mat_type = plm.TileType(shape=[TILE, TILE], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat, blayout=2, slayout=1)
        q_mat = plm.make_tile(mat_type, addr=A0, size=TT2)
        k_mat = plm.make_tile(mat_type, addr=A1, size=TT2)
        p_mat = plm.make_tile(mat_type, addr=A2, size=TT2)
        v_mat = plm.make_tile(mat_type, addr=A3, size=TT2)
        q_left = plm.make_tile(plm.TileType(shape=[TILE, TILE], dtype=pl.FP16, target_memory=pl.MemorySpace.Left, blayout=1, slayout=1), addr=A0, size=TT2)
        k_right = plm.make_tile(plm.TileType(shape=[TILE, TILE], dtype=pl.FP16, target_memory=pl.MemorySpace.Right, blayout=1, slayout=2), addr=A0, size=TT2)
        p_left = plm.make_tile(plm.TileType(shape=[TILE, TILE], dtype=pl.FP16, target_memory=pl.MemorySpace.Left, blayout=1, slayout=1), addr=A1, size=TT2)
        v_right = plm.make_tile(plm.TileType(shape=[TILE, TILE], dtype=pl.FP16, target_memory=pl.MemorySpace.Right, blayout=1, slayout=2), addr=A1, size=TT2)
        acc_type = plm.TileType(shape=[TILE, TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc, blayout=2, slayout=1, fractal=1024)
        qk_acc = plm.make_tile(acc_type, addr=A0, size=TT4)
        pv_acc = plm.make_tile(acc_type, addr=A2, size=TT4)

        core_id = pl.block.index_cast(pl.block.get_block_idx())
        sq_off = core_id * TILE

        plm.load(q_mat, q, [sq_off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        plm.move(q_left, q_mat, target_memory=pl.MemorySpace.Left)

        # First KV: load K[0:T,0:T], transpose during move to RIGHT
        plm.load(k_mat, k, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
        plm.move(k_right, k_mat, target_memory=pl.MemorySpace.Right, transpose=True)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        plm.matmul(qk_acc, q_left, k_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        plm.l0c_store(qk_acc, [sq_off, 0], [TILE, TILE], qk_buf)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY)
        pl.system.wait_cross_core(pipe=pl.PipeType.M, event_id=P_READY)

        plm.load(p_mat, p_buf, [sq_off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
        plm.load(v_mat, v, [0, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
        plm.move(p_left, p_mat, target_memory=pl.MemorySpace.Left)
        plm.move(v_right, v_mat, target_memory=pl.MemorySpace.Right)
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
        plm.matmul(pv_acc, p_left, v_right)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
        plm.l0c_store(pv_acc, [sq_off, 0], [TILE, TILE], pv_buf)
        pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY)

        for j in pl.range(1, skv_tiles):
            skv_off = j * TILE
            plm.load(k_mat, k, [skv_off, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
            plm.move(k_right, k_mat, target_memory=pl.MemorySpace.Right, transpose=True)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
            plm.matmul(qk_acc, q_left, k_right)
            pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
            plm.l0c_store(qk_acc, [sq_off, skv_off], [TILE, TILE], qk_buf)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY)
            pl.system.wait_cross_core(pipe=pl.PipeType.M, event_id=P_READY)

            plm.load(p_mat, p_buf, [sq_off, skv_off])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
            plm.load(v_mat, v, [skv_off, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=1)
            plm.move(p_left, p_mat, target_memory=pl.MemorySpace.Left)
            plm.move(v_right, v_mat, target_memory=pl.MemorySpace.Right)
            pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
            plm.matmul(pv_acc, p_left, v_right)
            pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
            plm.l0c_store(pv_acc, [sq_off, 0], [TILE, TILE], pv_buf)
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY)

    # Vector section (identical to v1)
    with pl.section_vector():
        skv_dim = pl.tensor.dim(k, 0)
        skv_tiles = (skv_dim + (TILE - 1)) // TILE
        qk_vec = plm.make_tile(plm.TileType(shape=[HALF, TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=V0, size=VB4)
        tmp_vec = plm.make_tile(plm.TileType(shape=[HALF, TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=V1, size=VB4)
        p_f16 = plm.make_tile(plm.TileType(shape=[HALF, TILE], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec), addr=V2, size=VB2)
        reduce_dst = plm.make_tile(plm.TileType(shape=[HALF, TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, valid_shape=[HALF, 1]), addr=V3, size=VB4)
        global_max = plm.make_tile(plm.TileType(shape=[HALF, TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=V4, size=VB4)
        global_sum = plm.make_tile(plm.TileType(shape=[HALF, TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=V5, size=VB4)
        running_o = plm.make_tile(plm.TileType(shape=[HALF, TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=V6, size=VB4)
        exp_corr = plm.make_tile(plm.TileType(shape=[HALF, TILE], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=V7, size=VB4)

        core_id = pl.block.index_cast(pl.block.get_block_idx())
        sq_off = core_id * TILE
        sub_id = pl.block.index_cast(pl.block.get_subblock_idx())
        row_off = sub_id * HALF

        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY)
        plm.load(qk_vec, qk_buf, [sq_off + row_off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        plm.row_max(reduce_dst, qk_vec, tmp_vec)
        plm.row_expand(global_max, reduce_dst)
        plm.sub(tmp_vec, qk_vec, global_max)
        plm.muls(tmp_vec, tmp_vec, SCALE)
        plm.exp(qk_vec, tmp_vec)
        plm.row_sum(reduce_dst, qk_vec, tmp_vec)
        plm.row_expand(global_sum, reduce_dst)
        plm.cast(p_f16, qk_vec, target_type=pl.FP16, mode="round")
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        plm.store(p_buf, p_f16, [sq_off + row_off, 0])
        pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY)

        pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY)
        plm.load(running_o, pv_buf, [sq_off + row_off, 0])
        pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

        for j in pl.range(1, skv_tiles):
            skv_off = j * TILE
            pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY)
            plm.load(qk_vec, qk_buf, [sq_off + row_off, skv_off])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            plm.row_max(reduce_dst, qk_vec, tmp_vec)
            plm.row_expand(tmp_vec, reduce_dst)
            plm.maximum(tmp_vec, tmp_vec, global_max)
            plm.sub(exp_corr, global_max, tmp_vec)
            plm.muls(exp_corr, exp_corr, SCALE)
            plm.exp(exp_corr, exp_corr)
            plm.mul(global_sum, global_sum, exp_corr)
            plm.mul(running_o, running_o, exp_corr)
            plm.muls(global_max, tmp_vec, 1.0)
            plm.sub(tmp_vec, qk_vec, global_max)
            plm.muls(tmp_vec, tmp_vec, SCALE)
            plm.exp(qk_vec, tmp_vec)
            plm.row_sum(reduce_dst, qk_vec, tmp_vec)
            plm.row_expand(tmp_vec, reduce_dst)
            plm.add(global_sum, global_sum, tmp_vec)
            plm.cast(p_f16, qk_vec, target_type=pl.FP16, mode="round")
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
            plm.store(p_buf, p_f16, [sq_off + row_off, skv_off])
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY)

            pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY)
            plm.load(qk_vec, pv_buf, [sq_off + row_off, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            plm.add(running_o, running_o, qk_vec)

        plm.div(running_o, running_o, global_sum)
        plm.cast(p_f16, running_o, target_type=pl.FP16, mode="round")
        pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
        plm.store(o, p_f16, [sq_off + row_off, 0])

    return o


# ================================================================
#  Reference + Tests
# ================================================================

def flash_attention_ref(q, k, v, d):
    scale_val = 1.0 / math.sqrt(d)
    qk = torch.matmul(q.float(), k.float().T) * scale_val
    attn = torch.softmax(qk, dim=-1)
    return torch.matmul(attn, v.float()).half()


def test_fa_kt():
    compiled = fe.compile(fa_kt_kernel, arch="a3")
    print("compiled:", compiled.lib_path)
    device = "npu:1"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    for sq, skv, d in [(64, 64, 64), (64, 128, 64)]:
        print(f"\nFA-KT ({sq},{skv},{d})")
        q = torch.rand((sq, d), device=device, dtype=torch.float16)
        k = torch.rand((skv, d), device=device, dtype=torch.float16)
        kt = k.T.contiguous()
        v = torch.rand((skv, d), device=device, dtype=torch.float16)
        o = torch.empty((sq, d), device=device, dtype=torch.float16)
        qk_buf = torch.zeros((sq, skv), device=device, dtype=torch.float32)
        p_buf = torch.zeros((sq, skv), device=device, dtype=torch.float16)
        pv_buf = torch.zeros((sq, d), device=device, dtype=torch.float32)
        fe.launch(None, 1, compiled, q, kt, v, o, qk_buf, p_buf, pv_buf)
        torch.npu.synchronize()
        o_ref = flash_attention_ref(q, k, v, d)

        # Debug: check intermediate buffers
        qk_ref = torch.matmul(q.float(), k.float().T)
        print(f"  qk_buf[0,:4]:  {qk_buf[0,:4].tolist()}")
        print(f"  qk_ref[0,:4]:  {qk_ref[0,:4].tolist()}")
        qk_diff = (qk_buf - qk_ref).abs().max().item()
        print(f"  QK max|diff|:  {qk_diff:.4f}")
        scale_val = 1.0 / math.sqrt(d)
        p_ref = torch.softmax(qk_ref * scale_val, dim=-1).half()
        print(f"  p_buf[0,:4]:   {p_buf[0,:4].tolist()}")
        print(f"  p_ref[0,:4]:   {p_ref[0,:4].tolist()}")
        print(f"  pv_buf[0,:4]:  {pv_buf[0,:4].tolist()}")
        pv_ref = torch.matmul(p_ref.float(), v.float())
        print(f"  pv_ref[0,:4]:  {pv_ref[0,:4].tolist()}")
        print(f"  o[0,:4]:       {o[0,:4].tolist()}")
        print(f"  o_ref[0,:4]:   {o_ref[0,:4].tolist()}")

        diff = (o - o_ref).abs().max().item()
        print(f"  max|diff|={diff:.4f}")
        torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)
        print("  PASS")


def test_fa_k():
    compiled = fe.compile(fa_k_kernel, arch="a3")
    print("compiled:", compiled.lib_path)
    device = "npu:1"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    for sq, skv, d in [(64, 128, 64), (64, 256, 64), (64, 512, 64)]:
        print(f"\nFA-K ({sq},{skv},{d})")
        q = torch.rand((sq, d), device=device, dtype=torch.float16)
        k = torch.rand((skv, d), device=device, dtype=torch.float16)
        v = torch.rand((skv, d), device=device, dtype=torch.float16)
        o = torch.empty((sq, d), device=device, dtype=torch.float16)
        qk_buf = torch.empty((sq, skv), device=device, dtype=torch.float32)
        p_buf = torch.empty((sq, skv), device=device, dtype=torch.float16)
        pv_buf = torch.empty((sq, d), device=device, dtype=torch.float32)
        fe.launch(None, 1, compiled, q, k, v, o, qk_buf, p_buf, pv_buf)
        torch.npu.synchronize()
        o_ref = flash_attention_ref(q, k, v, d)
        diff = (o - o_ref).abs().max().item()
        print(f"  max|diff|={diff:.4f}")
        torch.testing.assert_close(o, o_ref, rtol=5e-2, atol=5e-2)
        print("  PASS")


if __name__ == "__main__":
    print("=" * 60)
    print("Version 1: K^T pre-transposed")
    print("=" * 60)
    test_fa_kt()
    print("\n" + "=" * 60)
    print("Version 2: K not transposed (transpose during move)")
    print("=" * 60)
    test_fa_k()
    print("\nAll FlashAttention tests passed!")
