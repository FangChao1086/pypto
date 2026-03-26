"""Flash Attention frontend UT.

Current staged structure:
1. Cube1   : QK -> qk_buf
2. Vector1 : online softmax + final normalized P
3. Cube2   : P @ V -> pv_buf
4. Vector2 : cast/store final O
"""

import torch
import torch_npu
import torch.nn.functional as F

import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm


S0 = pl.DynVar("S0")
S1 = pl.DynVar("S1")

TILE_S0 = 128
TILE_S1 = 128
VEC_HALF_S0 = TILE_S0 // 2
K_TILE = 64
HEAD_SIZE = 128
SOFTMAX_SCALE = 0.08838834764831845  # 1.0 / sqrt(128)
TEST_S0 = 2048
TEST_S1 = 1024
TEST_BLOCK_DIM = 8
FA_QK_READY = 7
FA_P_READY = 8
FA_PV_READY = 9

@fe.kernel(auto_sync=True)
def flash_attention_kernel(
    q: pl.Tensor[[S0, HEAD_SIZE], pl.FP16],
    k_t: pl.Tensor[[HEAD_SIZE, S1], pl.FP16],
    v: pl.Tensor[[S1, HEAD_SIZE], pl.FP16],
    qk_buf: pl.Tensor[[S0, S1], pl.FP32],
    p_buf: pl.Tensor[[S0, S1], pl.FP16],
    p_norm_buf: pl.Tensor[[S0, S1], pl.FP16],
    pv_buf: pl.Tensor[[S0, HEAD_SIZE], pl.FP32],
    mi_out: pl.Tensor[[S0, 1], pl.FP32],
    li_out: pl.Tensor[[S0, 1], pl.FP32],
    o: pl.Tensor[[S0, HEAD_SIZE], pl.FP16],
) -> pl.Tensor[[S0, HEAD_SIZE], pl.FP16]:
    """QK + softmax + PV + output."""
    q_mat = plm.make_tile(
        plm.TileType(
            shape=[TILE_S0, K_TILE],
            dtype=pl.FP16,
            target_memory=pl.MemorySpace.Mat,
            blayout=2,
            slayout=1,
        ),
        addr=0x0000,
        size=16384,
    )
    k_mat = plm.make_tile(
        plm.TileType(
            shape=[K_TILE, TILE_S1],
            dtype=pl.FP16,
            target_memory=pl.MemorySpace.Mat,
            blayout=2,
            slayout=1,
        ),
        addr=0x04000,
        size=16384,
    )
    p_mat = plm.make_tile(
        plm.TileType(
            shape=[TILE_S0, TILE_S1],
            dtype=pl.FP16,
            target_memory=pl.MemorySpace.Mat,
            blayout=2,
            slayout=1,
        ),
        addr=0x08000,
        size=32768,
    )
    v_mat = plm.make_tile(
        plm.TileType(
            shape=[TILE_S1, HEAD_SIZE],
            dtype=pl.FP16,
            target_memory=pl.MemorySpace.Mat,
            blayout=2,
            slayout=1,
        ),
        addr=0x10000,
        size=32768,
    )
    q_left = plm.make_tile(
        plm.TileType(
            shape=[TILE_S0, K_TILE],
            dtype=pl.FP16,
            target_memory=pl.MemorySpace.Left,
            blayout=1,
            slayout=1,
        ),
        addr=0x00000,
        size=16384,
    )
    k_right = plm.make_tile(
        plm.TileType(
            shape=[K_TILE, TILE_S1],
            dtype=pl.FP16,
            target_memory=pl.MemorySpace.Right,
            blayout=1,
            slayout=2,
        ),
        addr=0x00000,
        size=16384,
    )
    p_left = plm.make_tile(
        plm.TileType(
            shape=[TILE_S0, TILE_S1],
            dtype=pl.FP16,
            target_memory=pl.MemorySpace.Left,
            blayout=1,
            slayout=1,
        ),
        addr=0x08000,
        size=32768,
    )
    v_right = plm.make_tile(
        plm.TileType(
            shape=[TILE_S1, HEAD_SIZE],
            dtype=pl.FP16,
            target_memory=pl.MemorySpace.Right,
            blayout=1,
            slayout=2,
        ),
        addr=0x08000,
        size=32768,
    )
    qk_acc = plm.make_tile(
        plm.TileType(
            shape=[TILE_S0, TILE_S1],
            dtype=pl.FP32,
            target_memory=pl.MemorySpace.Acc,
            blayout=2,
            slayout=1,
            fractal=1024,
            valid_shape=[-1, -1],
        ),
        addr=0x00000,
        size=65536,
    )
    pv_acc = plm.make_tile(
        plm.TileType(
            shape=[TILE_S0, HEAD_SIZE],
            dtype=pl.FP32,
            target_memory=pl.MemorySpace.Acc,
            blayout=2,
            slayout=1,
            fractal=1024,
            valid_shape=[-1, -1],
        ),
        addr=0x10000,
        size=65536,
    )

    qk_vec = plm.make_tile(
        plm.TileType(
            shape=[VEC_HALF_S0, TILE_S1],
            dtype=pl.FP32,
            target_memory=pl.MemorySpace.Vec,
        ),
        addr=0x00000,
        size=32768,
    )
    tmp_vec = plm.make_tile(
        plm.TileType(
            shape=[VEC_HALF_S0, TILE_S1],
            dtype=pl.FP32,
            target_memory=pl.MemorySpace.Vec,
        ),
        addr=0x08000,
        size=32768,
    )
    p_fp16 = plm.make_tile(
        plm.TileType(
            shape=[VEC_HALF_S0, TILE_S1],
            dtype=pl.FP16,
            target_memory=pl.MemorySpace.Vec,
        ),
        addr=0x10000,
        size=16384,
    )
    mi_local = plm.make_tile(
        plm.TileType(
            shape=[VEC_HALF_S0, 1],
            dtype=pl.FP32,
            target_memory=pl.MemorySpace.Vec,
            blayout=2,
        ),
        addr=0x14000,
        size=256,
    )
    mi_running = plm.make_tile(
        plm.TileType(
            shape=[VEC_HALF_S0, TILE_S1],
            dtype=pl.FP32,
            target_memory=pl.MemorySpace.Vec,
        ),
        addr=0x18000,
        size=32768,
    )
    li_running = plm.make_tile(
        plm.TileType(
            shape=[VEC_HALF_S0, TILE_S1],
            dtype=pl.FP32,
            target_memory=pl.MemorySpace.Vec,
        ),
        addr=0x20000,
        size=32768,
    )
    alpha_nd = plm.make_tile(
        plm.TileType(
            shape=[VEC_HALF_S0, TILE_S1],
            dtype=pl.FP32,
            target_memory=pl.MemorySpace.Vec,
        ),
        addr=0x28000,
        size=32768,
    )

    # ---------------- Stage 1: Cube1 computes QK ---------------- #
    with pl.section_cube():
        b_idx = pl.block.get_block_idx()
        block_idx = pl.block.index_cast(b_idx)
        b_num = pl.block.get_block_num()
        block_num = pl.block.index_cast(b_num)

        s0 = pl.tensor.dim(q, 0)
        s1 = pl.tensor.dim(k_t, 1)

        num_tiles_s0 = s0 // TILE_S0
        num_tiles_s1 = s1 // TILE_S1
        num_tiles_k = HEAD_SIZE // K_TILE

        for s0_tile in pl.range(block_idx, num_tiles_s0, block_num):
            for s1_tile in pl.range(num_tiles_s1):
                for k_tile in pl.range(num_tiles_k):
                    plm.load_tile(q_mat, q, [s0_tile, k_tile])
                    plm.load_tile(k_mat, k_t, [k_tile, s1_tile])

                    # pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
                    # pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

                    plm.move(q_left, q_mat)
                    plm.move(k_right, k_mat)

                    # pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
                    # pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)

                    if k_tile == 0:
                        plm.matmul(qk_acc, q_left, k_right)
                    else:
                        plm.matmul_acc(qk_acc, qk_acc, q_left, k_right)

                    # pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE2, event_id=2)
                    # pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE2, event_id=2)

                # pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
                # pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
                plm.store_tile(qk_buf, qk_acc, [s0_tile, s1_tile])
                pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=FA_QK_READY)

    # ---------------- Stage 2: Vector1 computes softmax ---------------- #
    # First pass over S1 computes online mi/li and writes stage p_buf.
    # Second pass in the same vector section rebuilds final p_norm_buf.
    with pl.section_vector():
        b_idx = pl.block.get_block_idx()
        block_idx = pl.block.index_cast(b_idx)
        b_num = pl.block.get_block_num()
        block_num = pl.block.index_cast(b_num)

        s0 = pl.tensor.dim(q, 0)
        s1 = pl.tensor.dim(k_t, 1)
        sub_idx = pl.block.get_subblock_idx()
        subblock_idx = pl.block.index_cast(sub_idx)

        num_tiles_s0 = s0 // TILE_S0
        num_tiles_s1 = s1 // TILE_S1

        for s0_tile in pl.range(block_idx, num_tiles_s0, block_num):
            pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=FA_QK_READY)
            s0_vec_tile = s0_tile * 2 + subblock_idx
            for s1_tile in pl.range(num_tiles_s1):
                # pl.system.bar_all()
                plm.load_tile(qk_vec, qk_buf, [s0_vec_tile, s1_tile])
                # pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                # pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

                if s1_tile == 0:
                    plm.row_max(mi_local, qk_vec, tmp_vec)
                    plm.row_expand(mi_running, mi_local)
                    plm.sub(tmp_vec, qk_vec, mi_running)
                    plm.muls(tmp_vec, tmp_vec, SOFTMAX_SCALE)
                    plm.exp(qk_vec, tmp_vec)
                    plm.row_sum(mi_local, qk_vec, tmp_vec)
                    plm.row_expand(li_running, mi_local)
                else:
                    plm.row_max(mi_local, qk_vec, tmp_vec)
                    plm.row_expand(tmp_vec, mi_local)
                    plm.maximum(tmp_vec, tmp_vec, mi_running)
                    plm.sub(alpha_nd, mi_running, tmp_vec)
                    plm.muls(alpha_nd, alpha_nd, SOFTMAX_SCALE)
                    plm.exp(alpha_nd, alpha_nd)
                    plm.mul(li_running, li_running, alpha_nd)
                    plm.ub_copy(mi_running, tmp_vec)
                    plm.sub(tmp_vec, qk_vec, mi_running)
                    plm.muls(tmp_vec, tmp_vec, SOFTMAX_SCALE)
                    plm.exp(qk_vec, tmp_vec)
                    plm.row_sum(mi_local, qk_vec, tmp_vec)
                    plm.row_expand(tmp_vec, mi_local)
                    plm.add(li_running, li_running, tmp_vec)

                plm.cast(p_fp16, qk_vec, target_type=pl.FP16, mode="round")
                # pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                # pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                plm.store_tile(p_buf, p_fp16, [s0_vec_tile, s1_tile])

            for s1_tile in pl.range(num_tiles_s1):
                # pl.system.bar_all()
                plm.load_tile(qk_vec, qk_buf, [s0_vec_tile, s1_tile])
                # pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
                # pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)

                plm.sub(tmp_vec, qk_vec, mi_running)
                plm.muls(tmp_vec, tmp_vec, SOFTMAX_SCALE)
                plm.exp(qk_vec, tmp_vec)
                plm.div(qk_vec, qk_vec, li_running)
                plm.cast(p_fp16, qk_vec, target_type=pl.FP16, mode="round")

                # pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                # pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
                plm.store_tile(p_norm_buf, p_fp16, [s0_vec_tile, s1_tile])

            # pl.system.bar_all()
            # pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            # pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            plm.row_max(mi_local, mi_running, tmp_vec)
            plm.store_tile(mi_out, mi_local, [s0_vec_tile, 0])
            plm.row_max(mi_local, li_running, tmp_vec)
            plm.store_tile(li_out, mi_local, [s0_vec_tile, 0])
            pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=FA_P_READY)

    # ---------------- Stage 3: Cube2 computes PV ---------------- #
    with pl.section_cube():
        b_idx = pl.block.get_block_idx()
        block_idx = pl.block.index_cast(b_idx)
        b_num = pl.block.get_block_num()
        block_num = pl.block.index_cast(b_num)

        s0 = pl.tensor.dim(q, 0)
        s1 = pl.tensor.dim(k_t, 1)

        num_tiles_s0 = s0 // TILE_S0
        num_tiles_s1 = s1 // TILE_S1

        for s0_tile in pl.range(block_idx, num_tiles_s0, block_num):
            pl.system.wait_cross_core(pipe=pl.PipeType.MTE3, event_id=FA_P_READY)
            for s1_tile in pl.range(num_tiles_s1):
                plm.load_tile(p_mat, p_norm_buf, [s0_tile, s1_tile])
                plm.load_tile(v_mat, v, [s1_tile, 0])

                # pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
                # pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)

                plm.move(p_left, p_mat)
                plm.move(v_right, v_mat)

                # pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)
                # pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=1)

                if s1_tile == 0:
                    plm.matmul(pv_acc, p_left, v_right)
                else:
                    plm.matmul_acc(pv_acc, pv_acc, p_left, v_right)

                # pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE2, event_id=2)
                # pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.MTE2, event_id=2)

            # pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
            # pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
            plm.store_tile(pv_buf, pv_acc, [s0_tile, 0])
            pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=FA_PV_READY)

    # ---------------- Stage 4: Vector2 writes final O ---------------- #
    with pl.section_vector():
        b_idx = pl.block.get_block_idx()
        block_idx = pl.block.index_cast(b_idx)
        b_num = pl.block.get_block_num()
        block_num = pl.block.index_cast(b_num)
        sub_idx = pl.block.get_subblock_idx()
        subblock_idx = pl.block.index_cast(sub_idx)

        s0 = pl.tensor.dim(q, 0)
        num_tiles_s0 = s0 // TILE_S0

        for s0_tile in pl.range(block_idx, num_tiles_s0, block_num):
            pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=FA_PV_READY)
            s0_vec_tile = s0_tile * 2 + subblock_idx
            # pl.system.bar_all()
            plm.load_tile(qk_vec, pv_buf, [s0_vec_tile, 0])
            # pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            # pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.V, event_id=0)
            plm.cast(p_fp16, qk_vec, target_type=pl.FP16, mode="round")
            # pl.system.sync_src(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            # pl.system.sync_dst(set_pipe=pl.PipeType.V, wait_pipe=pl.PipeType.MTE3, event_id=1)
            plm.store_tile(o, p_fp16, [s0_vec_tile, 0])

    return o


@fe.jit()
def test_flash_attention_multicore():
    """Run QK + softmax + PV + O and compare against CPU."""
    def log(msg: str):
        print(f"[FA-UT] {msg}", flush=True)

    def print_diff(name: str, actual: torch.Tensor, expected: torch.Tensor):
        diff = (actual.float() - expected.float()).abs()
        print(
            f"[FA-UT] {name}: max_abs={diff.max().item():.6e}, "
            f"mean_abs={diff.mean().item():.6e}",
            flush=True,
        )

    log("stage 0: compile")
    compiled_lib = fe.compile(flash_attention_kernel, arch="a3")
    print("compiled lib path:", compiled_lib.lib_path, flush=True)

    device = "npu:7"
    torch.npu.set_device(device)

    s0 = TEST_S0
    s1 = TEST_S1
    block_dim = TEST_BLOCK_DIM

    torch.manual_seed(0)
    log("stage 1: allocate tensors")
    q = torch.randn(s0, HEAD_SIZE, dtype=torch.float16, device=device)
    k_ref = torch.randn(s1, HEAD_SIZE, dtype=torch.float16, device=device)
    k_t = k_ref.transpose(0, 1).contiguous()
    v = torch.randn(s1, HEAD_SIZE, dtype=torch.float16, device=device)

    qk_buf = torch.zeros(s0, s1, dtype=torch.float32, device=device)
    p_buf = torch.zeros(s0, s1, dtype=torch.float16, device=device)
    p_norm_buf = torch.zeros(s0, s1, dtype=torch.float16, device=device)
    pv_buf = torch.zeros(s0, HEAD_SIZE, dtype=torch.float32, device=device)
    mi_out = torch.zeros(s0, 1, dtype=torch.float32, device=device)
    li_out = torch.zeros(s0, 1, dtype=torch.float32, device=device)
    o = torch.zeros(s0, HEAD_SIZE, dtype=torch.float16, device=device)

    log("stage 2: launch kernel")
    fe.launch(
        None,
        block_dim,
        compiled_lib,
        q,
        k_t,
        v,
        qk_buf,
        p_buf,
        p_norm_buf,
        pv_buf,
        mi_out,
        li_out,
        o,
    )
    log("stage 3: synchronize")
    torch.npu.synchronize() 

    log("stage 4: copy outputs to cpu")
    qk_cpu = qk_buf.cpu()
    p_cpu = p_buf.cpu().float()
    p_norm_cpu = p_norm_buf.cpu().float()
    pv_cpu = pv_buf.cpu()
    mi_cpu = mi_out.cpu()
    li_cpu = li_out.cpu()
    o_cpu = o.cpu()

    log("stage 5: build golden")
    qk_ref = torch.matmul(q.cpu().float(), k_ref.cpu().float().transpose(0, 1))
    v_ref_full = v.cpu().float()
    stage_p_ref = torch.zeros_like(qk_ref)
    mi_ref = torch.zeros(s0, dtype=torch.float32)
    li_ref = torch.zeros(s0, dtype=torch.float32)
    for s0_start in range(0, s0, TILE_S0):
        s0_end = s0_start + TILE_S0
        mi_running_ref = None
        li_running_ref = None
        for s1_start in range(0, s1, TILE_S1):
            s1_end = s1_start + TILE_S1
            qk_tile = qk_ref[s0_start:s0_end, s1_start:s1_end]
            mi_local_ref = torch.max(qk_tile, dim=1).values
            if s1_start == 0:
                mi_running_ref = mi_local_ref
                p_tile_ref = torch.exp((qk_tile - mi_running_ref[:, None]) * SOFTMAX_SCALE)
                li_running_ref = torch.sum(p_tile_ref, dim=1)
            else:
                mi_new_ref = torch.maximum(mi_running_ref, mi_local_ref)
                alpha_ref = torch.exp((mi_running_ref - mi_new_ref) * SOFTMAX_SCALE)
                p_tile_ref = torch.exp((qk_tile - mi_new_ref[:, None]) * SOFTMAX_SCALE)
                li_local_ref = torch.sum(p_tile_ref, dim=1)
                li_running_ref = alpha_ref * li_running_ref + li_local_ref
                mi_running_ref = mi_new_ref
            stage_p_ref[s0_start:s0_end, s1_start:s1_end] = p_tile_ref.half().float()

        mi_ref[s0_start:s0_end] = mi_running_ref
        li_ref[s0_start:s0_end] = li_running_ref

    qk_scaled_ref = qk_ref * SOFTMAX_SCALE
    p_norm_online_ref = torch.exp((qk_ref - mi_ref[:, None]) * SOFTMAX_SCALE) / li_ref[:, None]
    p_norm_online_ref_fp16 = p_norm_online_ref.half().float()
    attn_ref = torch.softmax(qk_scaled_ref, dim=-1)
    attn_ref_fp16 = attn_ref.half().float()
    pv_ref = torch.matmul(attn_ref_fp16, v_ref_full)
    o_ref = pv_ref.half()

    o_sdpa_ref = F.scaled_dot_product_attention(
        q.cpu().float().unsqueeze(0).unsqueeze(0),
        k_ref.cpu().float().unsqueeze(0).unsqueeze(0),
        v_ref_full.unsqueeze(0).unsqueeze(0),
        dropout_p=0.0,
        is_causal=False,
        scale=SOFTMAX_SCALE,
    )[0, 0].half()

    mi_cpu_col0 = mi_cpu[:, 0].contiguous()
    li_cpu_col0 = li_cpu[:, 0].contiguous()
    print_diff("qk", qk_cpu, qk_ref)
    print_diff("stage-p", p_cpu, stage_p_ref)
    print_diff("mi(col0)", mi_cpu_col0, mi_ref)
    print_diff("li(col0)", li_cpu_col0, li_ref)
    print_diff("normalize-p(vs online)", p_norm_cpu, p_norm_online_ref_fp16)
    print_diff("normalize-p(vs softmax)", p_norm_cpu, attn_ref_fp16)
    print_diff("online-p vs softmax-p", p_norm_online_ref_fp16, attn_ref_fp16)
    print_diff("pv", pv_cpu, pv_ref)
    print_diff("o", o_cpu, o_ref.float())

    print("***********multicore npu qk output***********")
    print(qk_cpu.shape, qk_cpu.dtype)
    print(qk_cpu)
    print("***********multicore golden qk output***********")
    print(qk_ref.shape, qk_ref.dtype)
    print(qk_ref)

    print("***********multicore npu p output***********")
    print(p_norm_cpu.shape, p_norm_cpu.dtype)
    print(p_norm_cpu)
    print("***********multicore golden p output***********")
    print(attn_ref_fp16.shape, attn_ref_fp16.dtype)
    print(attn_ref_fp16)

    print("***********multicore npu pv output***********")
    print(pv_cpu.shape, pv_cpu.dtype)
    print(pv_cpu)
    print("***********multicore golden pv output***********")
    print(pv_ref.shape, pv_ref.dtype)
    print(pv_ref)

    print("***********multicore npu final o output***********")
    print(o_cpu.shape, o_cpu.dtype)
    print(o_cpu)
    print("***********multicore golden final o output***********")
    print(o_ref.shape, o_ref.dtype)
    print(o_ref)
    print("***********multicore sdpa final o output***********")
    print(o_sdpa_ref.shape, o_sdpa_ref.dtype)
    print(o_sdpa_ref)

    log("stage 6: assert qk")
    torch.testing.assert_close(qk_cpu, qk_ref, rtol=1e-2, atol=1e-2)
    log("stage 7: assert p")
    torch.testing.assert_close(p_norm_cpu, attn_ref_fp16, rtol=1e-2, atol=1e-2)
    log("stage 8: assert pv")
    torch.testing.assert_close(pv_cpu, pv_ref, rtol=1e-2, atol=1e-2)
    log("stage 9: assert o")
    torch.testing.assert_close(o_cpu, o_ref, rtol=1e-2, atol=1e-2)
    log("stage 10: assert o vs sdpa")
    torch.testing.assert_close(o_cpu, o_sdpa_ref, rtol=1e-2, atol=1e-2)
    log("done")


if __name__ == "__main__":
    test_flash_attention_multicore()
    print("\nAll tests passed!")
