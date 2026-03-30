"""FlashAttention kernel using PyPTO IR manual (non-SSA) mode.

Multi-core with Q tiling loop + double buffer for K loads in Cube section.
K loaded with DN layout (TLOAD transposes on-chip), TILE_SQ=128, TILE_SKV=128, TILE_D=128.

Features:
  1. Multi-core: each core processes multiple Q tiles via strided loop
  2. Double buffer: k_mat ping/pong in MAT space for overlapping K loads with computation
  3. Backward sync: bar_all() at Q iteration boundary to ensure buffer reuse safety

Usage:
    python3 tests/ut/frontend/test_fa.py
"""

import math
import torch
import torch_npu
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm

# Tile dimensions
TS = 128          # Sq tile size
TKV = 128         # Skv tile size (KV dimension tiling)
TD = 128          # D / head dimension
TS_HALF = TS // 2 # 64 — each vector sub-block processes half of TS rows
SCALE = 1.0 / math.sqrt(TD)

# Cube tile byte sizes
Q_F16   = TS * TD * 2       # 32768  — [128, 128] FP16
KT_F16  = TD * TKV * 2      # 32768  — [128, 128] FP16 (K^T tile)
QK_F16  = TS * TKV * 2      # 32768  — [128, 128] FP16 (QK result per KV tile)
V_F16   = TKV * TD * 2      # 32768  — [128, 128] FP16
P_F16   = TS * TKV * 2      # 32768  — [128, 128] FP16 (P tile per KV iter)
QK_F32  = TS * TKV * 4      # 65536  — [128, 128] FP32 (ACC)
PV_F32  = TS * TD * 4       # 65536  — [128, 128] FP32 (ACC)

# MAT addresses (512KB budget) — with double buffer for K
MA0 = 0                          # q_mat         [128, 128] → 32 KB
MA0_PONG = MA0 + Q_F16              # q_mat         [128, 128] → 32 KB

MA1 = Q_F16 * 2                  # k_mat_ping    [128, 128] → 32 KB
MA1_PONG = MA1 + KT_F16             # k_mat_pong    [128, 128] → 32 KB

MA2 = MA1 + KT_F16 * 2           # p_mat         [128, 128] → 32 KB
MA2_PONG = MA2 + P_F16           # p_mat         [128, 128] → 32 KB

MA3 = MA2 + P_F16  * 2           # v_mat         [128, 128] → 32 KB
MA3_PONG = MA3 + V_F16           # v_mat         [128, 128] → 32 KB
# Total: 160 KB

# LEFT addresses (64KB budget)
LA0 = 0                      # q_left  [128, 128] → 32 KB
LA1 = Q_F16                  # p_left  [128, 128] → 32 KB
# Total: 64 KB

# RIGHT addresses (64KB budget)
RA0 = 0                      # k_right [128, 128] → 32 KB
RA1 = KT_F16                 # v_right [128, 128] → 32 KB
# Total: 64 KB

# ACC addresses (128KB budget on a3)
CA0 = 0                      # qk_acc  [128, 128] FP32 → 64 KB
CA1 = QK_F32                 # pv_acc  [128, 128] FP32 → 64 KB
# Total: 128 KB

# VEC addresses — sub-block processes [TS_HALF, TKV] for QK, [TS_HALF, TD] for O
# Reduce tiles use ColMajor [64,1] (256B) for reduce/expand ops,
# with RowMajor [1,64] (256B) aliases at same addr for element-wise ops (via TRESHAPE).
VB4_KV = TS_HALF * TKV * 4   # 32768  — [64, 128] FP32
VB2_KV = TS_HALF * TKV * 2   # 16384  — [64, 128] FP16
VB4    = TS_HALF * TD * 4    # 16384  — [64, 128] FP32
VB2    = TS_HALF * TD * 2    # 8192   — [64, 128] FP16
VB_RED = TS_HALF * 1 * 4     # 256    — [64, 1] FP32 ColMajor (reduce tile)
VA0 = 0                       # qk_vec    [64, 128] FP32
VA1 = VA0 + VB4_KV            # tmp_vec   [64, 128] FP32
VA2 = VA1 + VB4_KV            # p_f16     [64, 128] FP16
VA3 = VA2 + VB2_KV            # reduce_dst  [64, 1] FP32 ColMajor
VA4 = VA3 + VB_RED            # global_max  [64, 1] FP32 ColMajor
VA5 = VA4 + VB_RED            # global_sum  [64, 1] FP32 ColMajor
VA6 = VA5 + VB_RED            # exp_corr    [64, 1] FP32 ColMajor
VA7 = VA6 + VB_RED            # running_o   [64, 128] FP32
VA8 = VA7 + VB4               # pv_vec      [64, 128] FP32
VA9 = VA8 + VB4               # o_f16       [64, 128] FP16
# Total: VA9 + VB2 = 122880 bytes = 120 KB < 192 KB ✓
event_ids_01 = (0, 1)
event_ids_23= (2, 3)

# Cross-core sync event IDs
QK_READY = 0
P_READY = 1
PV_READY = 2
PV_CORE_STRIDE = 2 * TS  # each core needs 2 Q-tile slots in pv_buf (double buffer)

Sq2 = pl.DynVar('Sq')
Skv2 = pl.DynVar('Skv')
D2 = pl.DynVar('D')


"""
1. 右矩阵转置的写法
  对于输入 tensor K [N, K]，matmul 需要 K^T 即 [K, N] 的 RIGHT tile:
       ┬───────────────────────────────────────────────────┐
       │              写法 2: DN + RIGHT直载                │
       ┼───────────────────────────────────────────────────┤
       │ shape=[K,N], stride=[-1,1] → 但被 DN 解释为 [1,K]  │
       ┼───────────────────────────────────────────────────┤
       │ DN (列主序)                                        │
       ┼───────────────────────────────────────────────────┤
       │ [K_tile, N_tile] (如 [64,128])                    │
       ┼───────────────────────────────────────────────────┤
       │ RowMajor (1)                                      │
       ┼───────────────────────────────────────────────────┤
       │ ColMajor (2)                                      │
       ┼───────────────────────────────────────────────────┤
       │ 列主序读（硬件DMA按列取数据）实现转置                │
       ┴───────────────────────────────────────────────────┘
  - GlobalTensor 使用 Layout::DN 列主序，stride=[1, K]
  - TLOAD 按列主序读取数据，相当于硬件层面做了转置
  - 直接写入 RIGHT tile（blayout=row_major, slayout=col_major)
"""
def alloc_cube_buffer():
    q_mat_type = plm.TileType(shape=[TS, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)
    q_mat_0 = plm.make_tile(q_mat_type, addr=MA0, size=Q_F16)
    q_mat_1 = plm.make_tile(q_mat_type, addr=MA0_PONG, size=Q_F16)
    q_mat_buf = (q_mat_0, q_mat_1)

    k_mat_type = plm.TileType(shape=[TD, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat, blayout=1, slayout=2)
    k_mat_0 = plm.make_tile(k_mat_type, addr=MA1, size=KT_F16)
    k_mat_1 = plm.make_tile(k_mat_type, addr=MA1_PONG, size=KT_F16)
    k_mat_buf = (k_mat_0, k_mat_1)

    p_mat_0 = plm.make_tile(plm.TileType(shape=[TS, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat), addr=MA2, size=P_F16)
    p_mat_1 = plm.make_tile(plm.TileType(shape=[TS, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat), addr=MA2_PONG, size=P_F16)
    p_mat_buf = (p_mat_0, p_mat_1)

    v_mat_0 = plm.make_tile(plm.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat), addr=MA3, size=V_F16)
    v_mat_1 = plm.make_tile(plm.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat), addr=MA3_PONG, size=V_F16)
    v_mat_buf = (v_mat_0, v_mat_1)
    # LEFT / RIGHT
    left_0 = plm.make_tile(plm.TileType(shape=[TS, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Left), addr=LA0, size=Q_F16)
    left_1 = plm.make_tile(plm.TileType(shape=[TS, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Left), addr=LA1, size=Q_F16)
    left_buf = (left_0, left_1)

    right_0 = plm.make_tile(plm.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Right), addr=RA0, size=KT_F16)
    right_1 = plm.make_tile(plm.TileType(shape=[TKV, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Right), addr=RA1, size=KT_F16)
    right_buf = (right_0, right_1)
    # ACC
    acc_0 = plm.make_tile(plm.TileType(shape=[TS, TKV], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc), addr=CA0, size=QK_F32)
    acc_1 = plm.make_tile(plm.TileType(shape=[TS, TKV], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc), addr=CA1, size=PV_F32)
    acc_buf = (acc_0, acc_1)
    return q_mat_buf, k_mat_buf, p_mat_buf, v_mat_buf, left_buf, right_buf, acc_buf


def compute_qk(ctx):
    q_mat_idx = ctx.q_count % 2
    skv_off = ctx.skvi * TKV

    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_01[ctx.buf_idx])
    if ctx.skvi == 0:
        plm.load(q_mat_buf[q_mat_idx], q, [ctx.sq_off, 0])

    # QK phase — double-buffered K load
    plm.load(k_mat_buf[ctx.buf_idx], k, [skv_off, 0], layout="dn")
    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[ctx.l0ab_idx])
    plm.move(left_buf[ctx.l0ab_idx], q_mat_buf[q_mat_idx])
    plm.move(right_buf[ctx.l0ab_idx], k_mat_buf[ctx.buf_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_01[ctx.buf_idx])

    pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[ctx.l0c_idx])
    plm.matmul(acc_buf[ctx.l0c_idx], left_buf[ctx.l0ab_idx], right_buf[ctx.l0ab_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[ctx.l0ab_idx])
    plm.l0c_store(acc_buf[ctx.l0c_idx], [ctx.sq_off, skv_off], [TS, TKV], qk_buf)
    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[ctx.l0c_idx])
    ctx.l0ab_idx = 1 - ctx.l0ab_idx
    ctx.l0c_idx = 1 - ctx.l0c_idx
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY)
    return


def compute_pv(ctx):
    q_mat_idx = ctx.q_count % 2
    sv_off = ctx.skvi * TKV
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_23[ctx.buf_idx])
    plm.load(v_mat_buf[ctx.buf_idx], v, [sv_off, 0])
    pl.system.wait_cross_core(pipe=pl.PipeType.M, event_id=P_READY)
    plm.load(p_mat_buf[ctx.buf_idx], p_buf, [ctx.sq_off, sv_off])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[ctx.l0ab_idx])
    plm.move(left_buf[ctx.l0ab_idx], p_mat_buf[ctx.buf_idx])
    plm.move(right_buf[ctx.l0ab_idx], v_mat_buf[ctx.buf_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=event_ids_23[ctx.buf_idx])

    pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[ctx.l0c_idx])
    plm.matmul(acc_buf[ctx.l0c_idx], left_buf[ctx.l0ab_idx], right_buf[ctx.l0ab_idx])
    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
    pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)

    pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=event_ids_01[ctx.l0ab_idx])
    plm.l0c_store(acc_buf[ctx.l0c_idx], [ctx.core_id * PV_CORE_STRIDE + q_mat_idx * TS, 0], [TS, TD], pv_buf)
    pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=event_ids_01[ctx.l0c_idx])
    ctx.l0ab_idx = 1 - ctx.l0ab_idx
    ctx.l0c_idx = 1 - ctx.l0c_idx
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY)
    return


@fe.kernel
def fa_k_kernel(
    q: pl.Tensor[[Sq2, D2], pl.FP16],
    k: pl.Tensor[[Skv2, D2], pl.FP16],
    v: pl.Tensor[[Skv2, D2], pl.FP16],
    o: pl.Tensor[[Sq2, D2], pl.FP16],
    qk_buf: pl.Tensor[[Sq2, Skv2], pl.FP32],
    p_buf: pl.Tensor[[Sq2, Skv2], pl.FP16],
    pv_buf: pl.Tensor[[48 * TS, D2], pl.FP32],
) -> pl.Tensor[[Sq2, D2], pl.FP16]:
    # =================== CUBE SECTION ===================
    sq_dim = Sq2
    skv_dim = Skv2
    sq_tiles = (sq_dim + (TS - 1)) // TS
    skv_tiles = (skv_dim + (TKV - 1)) // TKV
    num_cores = pl.block.index_cast(pl.block.get_block_num())
    core_id = pl.block.index_cast(pl.block.get_block_idx())
    with pl.section_cube():
        # MAT tiles — double buffer for K (ping/pong)
        q_mat_buf, k_mat_buf, p_mat_buf, v_mat_buf, left_buf, right_buf, acc_buf = alloc_cube_buffer()
        # Double buffer event IDs: ping=0, pong=1

        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=0) # qk0
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=1) # qk1
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=2) # v0
        pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=3) # v1

        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
        
        pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_src(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=1)
        # Q tile loop: each core processes Q tiles in strided fashion
        ctx = pl.struct(sq_off=0, skvi=0, q_count=0, buf_idx=0, l0ab_idx=0, l0c_idx=0, core_id=core_id)
        for qi in pl.range(core_id, sq_tiles, num_cores):
            # ---- Remaining KV tiles with double buffer for k_mat ----
            for skvi in pl.range(0, skv_tiles):
                ctx.sq_off = qi * TS
                ctx.skvi = skvi
                ctx.buf_idx = (ctx.q_count * skv_tiles + skvi) % 2
                compute_qk(ctx)

                # PV phase
                compute_pv(ctx)

            ctx.q_count = ctx.q_count + 1
        
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=0) # qk0
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=1) # qk1
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=2) # v0
        pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.MTE2, event_id=3) # v1
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE1, event_id=1)
        pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=0)
        pl.system.sync_dst(set_pipe=pl.PipeType.FIX, wait_pipe=pl.PipeType.M, event_id=1)
    # =================== VECTOR SECTION ===================
    with pl.section_vector():
        sub_id = pl.block.index_cast(pl.block.get_subblock_idx())
        row_off = sub_id * TS_HALF

        # VEC tiles: sub-block processes [TS_HALF, TKV] for QK, [TS_HALF, TD] for O
        qk_vec = plm.make_tile(plm.TileType(shape=[TS_HALF, TKV], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA0, size=VB4_KV)
        tmp_vec = plm.make_tile(plm.TileType(shape=[TS_HALF, TKV], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA1, size=VB4_KV)
        p_f16 = plm.make_tile(plm.TileType(shape=[TS_HALF, TKV], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec), addr=VA2, size=VB2_KV)

        # Reduce tiles: ColMajor [64,1] for TROWMAX/TROWSUM/TROWEXPAND* ops
        reduce_dst = plm.make_tile(plm.TileType(shape=[TS_HALF, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2), addr=VA3, size=VB_RED)
        global_max = plm.make_tile(plm.TileType(shape=[TS_HALF, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2), addr=VA4, size=VB_RED)
        global_sum = plm.make_tile(plm.TileType(shape=[TS_HALF, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2), addr=VA5, size=VB_RED)
        exp_corr = plm.make_tile(plm.TileType(shape=[TS_HALF, 1], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec, blayout=2), addr=VA6, size=VB_RED)

        # RowMajor [1,64] aliases at same addresses — for element-wise ops (via TRESHAPE)
        reduce_dst_rm = plm.make_tile(plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA3, size=VB_RED)
        global_max_rm = plm.make_tile(plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA4, size=VB_RED)
        global_sum_rm = plm.make_tile(plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA5, size=VB_RED)
        exp_corr_rm = plm.make_tile(plm.TileType(shape=[1, TS_HALF], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA6, size=VB_RED)

        running_o = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA7, size=VB4)
        pv_vec = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec), addr=VA8, size=VB4)
        o_f16 = plm.make_tile(plm.TileType(shape=[TS_HALF, TD], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec), addr=VA9, size=VB2)
        q_count = 0
        # Q tile loop: must match cube's loop exactly for cross-core sync
        for qi in pl.range(core_id, sq_tiles, num_cores):
            sq_off = qi * TS
            q_mat_idx = q_count % 2
            # ---- First KV (ki=0): FlashSoftmax INIT ----
            pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY)
            plm.load(qk_vec, qk_buf, [sq_off + row_off, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            plm.row_max(reduce_dst, qk_vec, tmp_vec)
            pl.system.bar_v()  # TROWMAX → TROWEXPANDSUB
            plm.row_expand_sub(tmp_vec, qk_vec, reduce_dst)
            plm.muls(global_max_rm, reduce_dst_rm, 1.0)  # save max (RowMajor alias for CCE TMULS compat)
            plm.muls(tmp_vec, tmp_vec, SCALE)
            plm.exp(qk_vec, tmp_vec)
            pl.system.bar_v()  # TEXP → TROWSUM
            plm.row_sum(reduce_dst, qk_vec, tmp_vec)
            pl.system.bar_v()  # TROWSUM → TMULS(copy)
            plm.muls(global_sum_rm, reduce_dst_rm, 1.0)
            plm.cast(p_f16, qk_vec, target_type=pl.FP16, mode="round")
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
            plm.store(p_buf, p_f16, [sq_off + row_off, 0])
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY)

            pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY)
            plm.load(running_o, pv_buf, [core_id * PV_CORE_STRIDE + q_mat_idx * TS + row_off, 0])
            pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

            # ---- Remaining KV (ki=1..skv_tiles-1): FlashSoftmax UPDATE ----
            for ki in pl.range(1, skv_tiles):
                skv_off = ki * TKV
                pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY)
                plm.load(qk_vec, qk_buf, [sq_off + row_off, skv_off])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

                # --- softmax UPDATE (ref: softmax_opt_fa_not_init_impl) ---
                # pipe_barrier(PIPE_V) between dependent TVEC ops on a2/a3
                plm.row_max(reduce_dst, qk_vec, tmp_vec)
                pl.system.bar_v()  # TROWMAX → TMAX
                plm.maximum(reduce_dst_rm, reduce_dst_rm, global_max_rm)
                pl.system.bar_v()  # TMAX → TSUB
                plm.sub(exp_corr_rm, global_max_rm, reduce_dst_rm)
                pl.system.bar_v()  # TSUB → TMULS(copy)
                plm.muls(global_max_rm, reduce_dst_rm, 1.0)  # global_max = new_max
                pl.system.bar_v()  # TMULS(copy) → TROWEXPANDSUB
                # Interleave [1,64] correction and [64,128] softmax (independent data)
                plm.row_expand_sub(tmp_vec, qk_vec, reduce_dst)
                plm.muls(exp_corr_rm, exp_corr_rm, SCALE)
                plm.muls(tmp_vec, tmp_vec, SCALE)
                plm.exp(exp_corr_rm, exp_corr_rm)
                plm.exp(qk_vec, tmp_vec)
                plm.cast(p_f16, qk_vec, target_type=pl.FP16, mode="round")
                pl.system.bar_v()  # TEXP/TCVT → TMUL, TROWSUM
                plm.mul(global_sum_rm, global_sum_rm, exp_corr_rm)
                plm.row_sum(reduce_dst, qk_vec, tmp_vec)
                pl.system.bar_v()  # TROWSUM → TADD
                plm.add(global_sum_rm, global_sum_rm, reduce_dst_rm)

                pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
                plm.store(p_buf, p_f16, [sq_off + row_off, skv_off])
                pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY)

                pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY)
                plm.load(pv_vec, pv_buf, [core_id * PV_CORE_STRIDE + q_mat_idx * TS + row_off, 0])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                # --- GU: running update (ref: pto_macro_fa_gu) ---
                plm.row_expand_mul(running_o, running_o, exp_corr)
                plm.add(running_o, running_o, pv_vec)
            q_count = q_count + 1

            # Final: normalize and store output for this Q tile
            plm.row_expand_div(running_o, running_o, global_sum)
            plm.cast(o_f16, running_o, target_type=pl.FP16, mode="round")
            pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
            pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=0)
            plm.store(o, o_f16, [sq_off + row_off, 0])
    return o

# ================================================================
#  Reference + Tests
# ================================================================

def flash_attention_ref(q, k, v, d):
    scale_val = 1.0 / math.sqrt(d)
    qk = torch.matmul(q.float(), k.float().T) * scale_val
    attn = torch.softmax(qk, dim=-1)
    return torch.matmul(attn, v.float()).half()


def test_fa_k():
    compiled = fe.compile(fa_k_kernel, arch="a3", codegen_mode="cce")
    print("compiled:", compiled.lib_path)
    device = "npu:5"
    torch.npu.set_device(device)
    torch.manual_seed(42)
    # Test shapes: (Sq, Skv, D) — multi-core with Q tiling
    # Note: cross-core sync only verified for skv_tiles=1 (skv<=TKV).
    # Multi-KV-tile cross-core handshake is a known limitation of the
    # original fa_k_kernel's pto.sync.set/wait pattern.
    for sq, skv, d, num_cores in [ # num_cores is the number of cores of Cube
        (8192, 8192, TD, 24),    # 4 cores, 1 Q tile each
        # (128, 128, TD, 1),    # 4 cores, 1 Q tile each
    ]:
        print(f"\nFA-K ({sq},{skv},{d}) cores={num_cores}")
        q = torch.rand((sq, d), device=device, dtype=torch.float16)
        k = torch.rand((skv, d), device=device, dtype=torch.float16)
        v = torch.rand((skv, d), device=device, dtype=torch.float16)
        o = torch.zeros((sq, d), device=device, dtype=torch.float16)
        qk_buf = torch.zeros((sq, skv), device=device, dtype=torch.float32)
        p_buf = torch.zeros((sq, skv), device=device, dtype=torch.float16)
        pv_buf = torch.zeros((48 * TS, d), device=device, dtype=torch.float32)
        fe.launch(None, num_cores, compiled, q, k, v, o, qk_buf, p_buf, pv_buf)
        torch.npu.synchronize()
        o_ref = flash_attention_ref(q, k, v, d)
        diff = (o - o_ref).abs().max().item()
        print(f"  max|diff|={diff:.4f}")
        torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
        print("  PASS")


if __name__ == "__main__":
    print("FA DN layout for K, multi-core + Q tiling + double buffer")
    print("=" * 60)
    test_fa_k()
    print("\nAll FlashAttention tests passed!")
