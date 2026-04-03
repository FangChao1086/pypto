# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under terms:and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not get use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""FlashAttention buffer and synchronization management tests.

This module tests the buffer management policies and synchronization mechanisms
adapted from ops-transformer for FlashAttention on A5 architecture.
"""

import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm
from pypto.language.manual.buffer_policy import SyncType, PipelineType
from pypto.pypto_core.ir import MemorySpace
from pypto import backend
from pypto.backend import BackendType
from pypto.pypto_core.codegen import PTOCodegen
from pypto.frontend.kernel import KernelDef


def _compile_to_mlir(prog) -> str:
    """Compile a KernelDef or ir.Program to PTO MLIR without running external tools."""
    from pypto.frontend.kernel import KernelDef
    if isinstance(prog, KernelDef):
        prog = prog.parse()
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.PTO)
    codegen = PTOCodegen()
    result = codegen.generate(prog)
    return result if isinstance(result, str) else "".join(result.values())


# ---------------------------------------------------------------------------
# Test Case 1: Double Buffer Policy (BuffersPolicyDB)
# ---------------------------------------------------------------------------

@fe.kernel
def test_double_buffer_kernel(
    a: pl.Tensor[[256, 128], pl.FP16],
    b: pl.Tensor[[256, 128], pl.FP16],
) -> pl.Tensor[[256, 128], pl.FP16]:
    """Test double buffer policy with buffer-level synchronization.
    
    This test validates the complete doublebuffer pattern using buffer lock/free:
    - Ping-pong buffer rotation across multiple iterations
    - Fine-grained pipeline synchronization (PIPE_MTE2 and PIPE_V)
    - Overlapping data transfer and computation
    - Proper MLIR generation for doublebuffer operations
    
    The kernel performs 4 iterations using true double buffering:
    1. Preload first buffer
    2. For each iteration (except last):
       - Get next buffer and preload data
       - Compute on current buffer
       - Store result from current buffer
       - Swap buffers
    3. Process last buffer
    
    This pattern hides memory latency by overlapping load and compute operations.
    """
    with pl.section_vector():
        tile_type = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Vec)
        policy = plm.BuffersPolicyDB(tile_type, pl.MemorySpace.Vec, SyncType.INNER_CORE_SYNC)
        
        # Preload first buffer
        buf_curr = policy.get()
        buf_curr.lock(PipelineType.PIPE_MTE2)
        plm.load(buf_curr, a, [0, 0])
        buf_curr.free(PipelineType.PIPE_MTE2)
        
        # Main loop: overlap load and compute
        for i in pl.range(3):
            # Get next buffer and preload data (overlap with current buffer's compute)
            buf_next = policy.get()
            buf_next.lock(PipelineType.PIPE_MTE2)
            plm.load(buf_next, a, [(i + 1) * 64, 0])
            buf_next.free(PipelineType.PIPE_MTE2)
            
            # Compute on current buffer (overlaps with next buffer's load)
            plm.add(buf_curr, buf_curr, buf_curr)
            buf_curr.lock(PipelineType.PIPE_V)
            plm.store(b, buf_curr, [i * 64, 0])
            buf_curr.free(PipelineType.PIPE_V)
            
            # Swap buffers
            buf_curr = buf_next
        
        # Process last buffer
        plm.add(buf_curr, buf_curr, buf_curr)
        buf_curr.lock(PipelineType.PIPE_V)
        plm.store(b, buf_curr, [3 * 64, 0])
        buf_curr.free(PipelineType.PIPE_V)
    
    return b


def test_double_buffer_policy():
    """Test BuffersPolicyDB with buffer-level synchronization.
    
    Validates:
    - Double buffer allocation (ping and pong)
    - Ping-pong rotation across multiple iterations
    - Fine-grained pipeline synchronization using lock/free
    - Proper MLIR generation for doublebuffer operations
    """
    from pypto.language.manual.buffer_policy import EventIdGenerator
    EventIdGenerator.reset()
    
    mlir = _compile_to_mlir(test_double_buffer_kernel)
    print("\n=== test_double_buffer_policy MLIR ===")
    print(mlir)
    
    # Verify basic structure
    assert "pto.section.vector {" in mlir, "Expected pto.section.vector for AIV"
    assert "pto.alloc_tile" in mlir, "Expected pto.alloc_tile"
    
    # Verify double buffer allocation
    assert mlir.count("pto.alloc_tile") == 2, \
        f"Expected 2 alloc_tile, got {mlir.count('pto.alloc_tile')}"
    
    # 验证存在 scf.for 循环（而不是直接展开）
    assert "scf.for" in mlir, "Expected scf.for loop structure"
    
    # 验证循环体内有正确的操作（每个操作出现1次，在循环体内）
    assert "pto.tload" in mlir, "Expected pto.tload in loop body"
    assert "pto.tadd" in mlir, "Expected pto.tadd in loop body"
    assert "pto.tstore" in mlir, "Expected pto.tstore in loop body"
    
    # 验证循环范围是 3 次迭代（主循环）
    assert "to %c3" in mlir, "Expected loop to iterate 3 times"
    
    # 验证 buffer lock/free 同步
    assert "manual.sync" in mlir, \
        "Expected manual.sync for buffer lock/free"
    assert '"direction": "lock"' in mlir, \
        "Expected lock direction in sync"
    assert '"direction": "free"' in mlir, \
        "Expected free direction in sync"
    assert "#pto<pipe PIPE_MTE2>" in mlir, \
        "Expected PIPE_MTE2 in sync"
    assert "#pto<pipe PIPE_V>" in mlir, \
        "Expected PIPE_V in sync"

# ---------------------------------------------------------------------------
# Test Case 2: Triple Buffer Policy (BuffersPolicy3buff)
# ---------------------------------------------------------------------------

@fe.kernel
def test_triple_buffer_kernel(
    p: pl.Tensor[[192, 128], pl.FP16],  # P 矩阵（来自 Vector 的 softmax）
    v: pl.Tensor[[128, 64], pl.FP16],  # V 矩阵（Value）
    output: pl.Tensor[[192, 64], pl.FP16],  # 输出 Tensor（存储 BMM2 结果）
) -> pl.Tensor[[192, 64], pl.FP16]:
    """Test triple buffer policy with cross-core synchronization (CROSS_CORE_SYNC_FORWARD).
    
    Simplified test focusing on:
    - L1 triple buffer (no UB doublebuffer)
    - Vector core (producer): writes P matrix to L1
    - Cube core (consumer): reads P from L1, (performs BMM2 (P × V)
    - Independent rotation (get_vec() and get_cube())
())
    - Multiple iterations to verify rotation
    
    This matches ops-transformer's flash_attention_score pattern:
    - Vector: ProcessVec1 writes P matrix (64x128) to L1
    - Cube: IterateBmm2 performs BMM2: P (64x128) × V (128x64) → Result (64x64)
    """
    
    # 创建共享的 L1 triple buffer policy（存储 P 矩阵）
    l1_tile_type = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)
    
    # Vector 和 Cube 使用相同的物理内存
    l1_policy_vec = plm.BuffersPolicy3buff(
        l1_tile_type, 
        MemorySpace.Mat, 
        SyncType.CROSS_CORE_SYNC_FORWARD,
        base_addr=0x0  # 相同的物理基地址
    )
    
    l1_policy_cube = plm.BuffersPolicy3buff(
        l1_tile_type, 
        MemorySpace.Mat, 
        SyncType.CROSS_CORE_SYNC_FORWARD,
        base_addr=0x0  # 相同的物理基地址！
    )
    
    # 输出 Tensor 由函数参数提供，无需创建 Tile 缓冲区
    
    # ========== Vector 核心操作（生产者）==========
    with pl.section_vector():
        for i in pl.range(6):
            # 使用 get_vec() 获取 L1 缓冲区（独立轮转）
            l1_buf = l1_policy_vec.get_vec()
            
            # 加载 P 矩阵到 L1 缓冲区（64x128）
            plm.load(l1_buf, p, [i * 64, 0])
            
            # 通知 Cube 数据已就绪
            l1_policy_vec.sync_forward(event_id=0)
    
    # ========== Cube 核心操作（消费者）==========
    with pl.section_cube():
        # L0C 缓冲区用于 BMM2 结果（64x64）
        l0c_tile_type = plm.TileType(shape=[64, 64], dtype=pl.FP16, target_memory=pl.MemorySpace.Acc)
        l0c_buffer = plm.make_tile(l0c_tile_type, addr=0x20000, size=8192)
        
        # V 矩阵缓冲区（128x64）
        v_tile_type = plm.TileType(shape=[128, 64], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)
        v_buffer = plm.make_tile(v_tile_type, addr=0x10000, size=16384)
        
        # 加载 V 矩阵到 L1
        plm.load(v_buffer, v, [0, 0])
        
        for i in pl.range(6):
            # 使用 get_cube() 获取 L1 缓冲区（独立轮转）
            l1_buf = l1_policy_cube.get_cube()
            
            # 等待 Vector 数据就绪
            l1_policy_cube.wait_data_ready(event_id=0)
            
            # 执行 BMM2 矩阵乘法：P (64x128) × V (128x64) → Result (64x64)
            plm.matmul(l1_buf, v_buffer, out=l0c_buffer)
            
            # 存储结果
            plm.store(output, l0c_buffer, [i * 64, 0])
    
    return output


def test_triple_buffer_policy():
    """Test BuffersPolicy3buff with cross-core synchronization (CROSS_CORE_SYNC_FORWARD).
    
    Validates:
    - Triple buffer allocation (a, b, c)
    - Cross-core synchronization between Vector and Cube cores
    - Independent buffer rotation (get_vec() and get_cube())
    - Cube core performs matrix multiplication (BMM2)
    - Multiple iterations to verify rotation correctness
    - Proper MLIR generation for triple buffer operations
    """
    mlir = _compile_to_mlir(test_triple_buffer_kernel)
    print("\n=== test_triple_buffer_policy MLIR ===")
    print(mlir)
    
    # 验证两个 section
    assert "pto.section.vector {" in mlir, "Expected pto.section.vector for AIV"
    assert "pto.section.cube {" in mlir, "Expected pto.section.cube for AIC"
    
    # 验证三缓冲区分配（每个 policy 3 个，共 6 个 L1 + 1 个 V + 1 个 L0C + 1 个输出）
    assert "pto.alloc_tile" in mlir, "Expected pto.alloc_tile"
    assert mlir.count("pto.alloc_tile") == 9, \
        f"Expected 9 alloc_tile (2 policies × 3 L1 + 1 V + 1 L0C + 1 output), got {mlir.count('pto.alloc_tile')}"
    
    # 验证 L1 缓冲区地址
    # 验证存在 scf.for 循环结构
    assert "scf.for" in mlir, "Expected scf.for loop structure"
    
    # 验证循环体内有正确的操作
    # Vector section: 1次 tload (在循环体内)
    # Cube section: 1次 tload (V load, 循环外) + 1次 tmatmul + 1次 tstore (在循环体内)
    assert "pto.tload" in mlir, "Expected pto.tload"
    assert "pto.tmatmul" in mlir, "Expected pto.tmatmul in loop body"
    assert "pto.tstore" in mlir, "Expected pto.tstore in loop body"
    
    # 验证循环范围是 6 次迭代
    assert "to %c6" in mlir, "Expected loop to iterate 6 times"
    assert "pto.tload" in mlir, "Expected pto.tload"
    assert "pto.tmatmul" in mlir, "Expected pto.tmatmul in loop body"
    assert "pto.tstore" in mlir, "Expected pto.tstore in loop body"
    
    # 验证循环范围是 6 次迭代
    assert "to %c6" in mlir, "Expected loop to iterate 6 times"
    
    # 验证跨核同步
    assert "pto.wait_event" in mlir, "Expected pto.wait_event for wait_data_ready"
    assert "pto.record_event" in mlir, "Expected pto.record_event for sync_forward"
    
    # 验证事件 ID
    assert "EVENT_ID0>" in mlir, "Expected EVENT_ID0"



# ---------------------------------------------------------------------------
# Test Case 3b: Triple Buffer KV Reuse Pattern
# ---------------------------------------------------------------------------

@fe.kernel
def test_triple_buffer_kv_reuse_pattern_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
    b: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    """Test triple buffer policy with KV reuse pattern.
    
    KV reuse maintains an independent rotation state,
    allowing efficient reuse of KV cache across attention heads.
    """
    with pl.section_cube():
        tile_type = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)
        policy = plm.BuffersPolicy3buff(tile_type, MemorySpace.Mat, SyncType.INNER_CORE_SYNC)
        
        # KV reuse: independent rotation
        buf_reused0 = policy.get_reused()
        buf_reused1 = policy.get_reused()
        buf_reused2 = policy.get_reused()
        
        # Load data
        plm.load(buf_reused0, a, [0, 0])
        plm.load(buf_reused1, b, [0, 0])
        
        # Compute
        plm.add(buf_reused0, buf_reused1, buf_reused2)
    
    return buf_reused2


def test_triple_buffer_kv_reuse_pattern():
    """Test BuffersPolicy3buff with KV reuse pattern."""
    mlir = _compile_to_mlir(test_triple_buffer_kv_reuse_pattern_kernel)
    print("\n=== test_triple_buffer_kv_reuse_pattern MLIR ===")
    print(mlir)
    
    assert "pto.section.cube {" in mlir
    assert mlir.count("pto.alloc_tile") == 3, f"Expected 3 alloc_tile, got {mlir.count('pto.alloc_tile')}"
    assert "pto.tadd" in mlir

# ---------------------------------------------------------------------------
# Test Case 4: Triple Buffer Q Reuse
# ---------------------------------------------------------------------------

@fe.kernel
def test_triple_buffer_q_reuse_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
    b: pl.Tensor[[64, 128], pl.FP16],
    c: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    """Test triple buffer policy with Q reuse pattern."""
    with pl.section_cube():
        tile_type = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)
        policy = plm.BuffersPolicy3buff(tile_type, MemorySpace.Mat, SyncType.INNER_CORE_SYNC)
        
        buf_a = policy.get()
        buf_b = policy.get()
        buf_c = policy.get()
        
        # Q reuse: get previous buffer
        buf_prev = policy.get_pre()
        
        plm.load(buf_a, a, [0, 0])
        plm.load(buf_b, b, [0, 0])
        plm.load(buf_c, c, [0, 0])
        plm.add(buf_a, buf_b, buf_prev)
    
    return buf_prev

def test_triple_buffer_q_reuse():
    """Test BuffersPolicy3buff with Q reuse."""
    mlir = _compile_to_mlir(test_triple_buffer_q_reuse_kernel)
    print("\n=== test_triple_buffer_q_reuse MLIR ===")
    print(mlir)
    
    assert "pto.section.cube {" in mlir
    assert mlir.count("pto.alloc_tile") == 3
    assert "pto.tadd" in mlir

# ---------------------------------------------------------------------------
# Test Case 5: Four Buffer Policy (BuffersPolicy4buff)
# ---------------------------------------------------------------------------

@fe.kernel
def test_four_buffer_kernel(
    a: pl.Tensor[[64, 64], pl.FP32],
    b: pl.Tensor[[64, 64], pl.FP32],
    c: pl.Tensor[[64, 64], pl.FP32],
    d: pl.Tensor[[64, 64], pl.FP32],
) -> pl.Tensor[[64, 64], pl.FP32]:
    """Test four buffer policy with AIC (Mat memory space)."""
    with pl.section_cube():
        tile_type = plm.TileType(shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Mat)
        policy = plm.BuffersPolicy4buff(tile_type, MemorySpace.Acc, SyncType.INNER_CORE_SYNC)
        
        buf0 = policy.get()
        buf1 = policy.get()
        buf2 = policy.get()
        buf3 = policy.get()
        
        plm.load(buf0, a, [0, 0])
        plm.load(buf1, b, [0, 0])
        plm.load(buf2, c, [0, 0])
        plm.load(buf3, d, [0, 0])
        
        # 新增：细粒度同步，同步MTE2流水线（load操作）
        policy.sync_inner(PipelineType.PIPE_MTE2)
    
    return buf3


def test_four_buffer_policy():
    """Test BuffersPolicy4buff with AIC memory space."""
    mlir = _compile_to_mlir(test_four_buffer_kernel)
    print("\n=== test_four_buffer_policy MLIR ===")
    print(mlir)
    
    assert "pto.section.cube {" in mlir, "Expected pto.section.cube for AIC"
    assert "pto.alloc_tile" in mlir, "Expected pto.alloc_tile"
    assert mlir.count("pto.alloc_tile") == 4, f"Expected 4 alloc_tile, got {mlir.count('pto.alloc_tile')}"
    assert "base_addr = 0" in mlir, "Expected base_addr = 0 for first buffer"
    assert "base_addr = 16384" in mlir, "Expected base_addr = 16384 for second buffer"
    assert "base_addr = 32768" in mlir, "Expected base_addr = 32768 for third buffer"
    assert "base_addr = 49152" in mlir, "Expected base_addr = 49152 for fourth buffer"
    
    # 新增：验证细粒度同步
    assert "pto.barrier_sync [#pto<pipe PIPE_MTE2>]" in mlir, \
        "Expected pto.barrier_sync with PIPE_MTE2"
 

# ---------------------------------------------------------------------------
# Test Case 6: Cross Core Both Sync
# ---------------------------------------------------------------------------

@fe.kernel
def test_cross_core_both_sync_kernel(
    a: pl.Tensor[[64, 128], pl.FP16],
    b: pl.Tensor[[64, 128], pl.FP16],
) -> pl.Tensor[[64, 128], pl.FP16]:
    """Test cross-core bidirectional synchronization using policy.sync() (AIC version)."""
    with pl.section_cube():
        tile_type = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)
        policy = plm.BuffersPolicyDB(tile_type, MemorySpace.Mat, SyncType.CROSS_CORE_SYNC_BOTH)
        
        tile_a = policy.get()
        tile_b = policy.get()
        tile_c = plm.make_tile(tile_type, addr=0x8000, size=16384)
        
        policy.sync_both(event_id=0, direction="allocate")
        
        plm.load(tile_a, a, [0, 0])
        plm.load(tile_b, b, [0, 0])
        plm.matmul(tile_a, tile_b, tile_c)
        
        policy.sync_both(event_id=0, direction="record")
    
    return tile_c


def test_cross_core_both_sync():
    """Test cross_core_sync_both generates correct MLIR (AIC version)."""
    mlir = _compile_to_mlir(test_cross_core_both_sync_kernel)
    print("\n=== test_cross_core_both_sync MLIR ===")
    print(mlir)
    
    assert "pto.section.cube {" in mlir, "Expected pto.section.cube for AIC"
    assert "pto.wait_event" in mlir, "Expected pto.wait_event for allocate"
    assert "pto.record_event" in mlir, "Expected pto.record_event for record"
    assert "#pto<event EVENT_ID0>" in mlir, "Expected EVENT_ID0"


# ---------------------------------------------------------------------------
# Test Case 7: FlashAttention Complete Flow
# ---------------------------------------------------------------------------

@fe.kernel
def test_flashattention_complete_kernel(
    q: pl.Tensor[[1, 1, 64, 128], pl.FP16],
    k: pl.Tensor[[1, 1, 64, 128], pl.FP16],
    v: pl.Tensor[[1, 1, 64, 128], pl.FP16],
    output: pl.Tensor[[1, 1, 64, 128], pl.FP16],
) -> pl.Tensor[[1, 1, 64, 128], pl.FP16]:
    """Test complete FlashAttention flow with fine-grained synchronization.
    
    This uses for loops to iterate through 2 rounds, with automatic ping-pong
    rotation via get(). Each core (AIC and AIV) has its own independent
    BuffersPolicyDB instance, but they share same physical memory.
    
    Synchronization pattern follows pto-isa's TSync_Custom semantics:
    - Producer: allocate_buffer(event_id_free) → record_data_ready(event_id_record)
    - Consumer: wait_data_ready(event_id_record) → free_buffer(event_id_free)
    
    Event ID pairing:
    - Round 0: AIC(allocate=0, record=1) → AIV(wait=1, free=0)
    - Round 1: AIC(allocate=2, record=3) → AIV(wait=3, free=2)
    """
    
    # 为 Cube 和 Vector 创建独立的 Policy 对象
    ub_tile_type = plm.TileType(shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
    
    # 关键：两个 Policy 使用相同的 base_addr，指向相同的物理内存
    ub_policy_cube = plm.BuffersPolicyDB(
        ub_tile_type, 
        MemorySpace.Vec, 
        SyncType.CROSS_CORE_SYNC_BOTH,
        base_addr=0x0  # 相同的物理基地址
    )
    
    ub_policy_vec = plm.BuffersPolicyDB(
        ub_tile_type, 
        MemorySpace.Vec, 
        SyncType.CROSS_CORE_SYNC_BOTH,
        base_addr=0x0  # 相同的物理基地址！
    )
    
    # AIC 核心操作 - compute_qk
    with pl.section_cube():
        # Q/K tiles (L1 buffers)
        q_tile_type = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)
        k_tile_type = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)
        
        q_buf0 = plm.make_tile(q_tile_type, addr=0x10000, size=16384)
        q_buf1 = plm.make_tile(q_tile_type, addr=0x14000, size=16384)
        k_buf0 = plm.make_tile(k_tile_type, addr=0x18000, size=16384)
        k_buf1 = plm.make_tile(k_tile_type, addr=0x1C000, size=16384)
        
        # L0C buffer for QK result
        l0c_tile_type = plm.TileType(shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
        l0c_qk0 = plm.make_tile(l0c_tile_type, addr=0x20000, size=16384)
        l0c_qk1 = plm.make_tile(l0c_tile_type, addr=0x24000, size=16384)
        
        # 循环两轮
        for round_idx in pl.range(2):
            # 动态选择 buffer
            if round_idx == 0:
                q_buf = q_buf0
                k_buf = k_buf0
                l0c_qk = l0c_qk0
                
                # 等待 AIV 释放 buffer (allocate)
                ub_policy_cube.allocate_buffer(0)
                
                # Load Q and K
                plm.load(q_buf, q, [0, 0, 0, 0])
                plm.load(k_buf, k, [0, 0, 0, 0])
                
                # BMM: QK = Q @ K^T (L1 -> L0C)
                plm.matmul(q_buf, k_buf, out=l0c_qk)
                
                # 从 policy 获取 UB buffer（自动 ping-pong 切换）
                ub_qk = ub_policy_cube.get()
                
                # Move L0C to UB
                plm.move(ub_qk, l0c_qk)
                
                # 通知 AIV 数据已就绪 (record)
                ub_policy_cube.record_data_ready(1)
            else:
                q_buf = q_buf1
                k_buf = k_buf1
                l0c_qk = l0c_qk1
                
                # 等待 AIV 释放 buffer (allocate)
                ub_policy_cube.allocate_buffer(2)
                
                # Load Q and K
                plm.load(q_buf, q, [0, 0, 0, 0])
                plm.load(k_buf, k, [0, 0, 0, 0])
                
                # BMM: QK = Q @ K^T (L1 -> L0C)
                plm.matmul(q_buf, k_buf, l0c_qk)
                
                # 从 policy 获取 UB buffer（自动 ping-pong 切换）
                ub_qk = ub_policy_cube.get()
                
                # Move L0C to UB
                plm.move(ub_qk, l0c_qk)

                # 通知 AIV 数据已就绪 (record)
                ub_policy_cube.record_data_ready(3)
    
    # AIV 核心操作 - wait for QK and compute softmax
    with pl.section_vector():
        # 使用 init_consumer() 进行正确的初始化
        ub_policy_vec.init_consumer()
        
        # 循环两轮
        for round_idx in pl.range(2):
            # 等待 AIC 数据就绪 (wait)
            if round_idx == 0:
                ub_policy_vec.wait_data_ready(1)
                
                # 从 policy 获取 UB buffer（自动 ping-pong 切换）
                ub_qk = ub_policy_vec.get()
                
                # 使用 UB buffer 中的数据
                plm.exp(ub_qk, ub_qk)
                
                # Store结果到输出tensor
                plm.store(output, ub_qk, [0, 0, 0, 0])
                
                # 释放 buffer (free)
                ub_policy_vec.free_buffer(0)
            else:
                ub_policy_vec.wait_data_ready(3)
                
                # 从 policy 获取 UB buffer（自动 ping-pong 切换）
                ub_qk = ub_policy_vec.get()
                
                # 使用 UB buffer 中的数据
                plm.exp(ub_qk, ub_qk)
                
                # Store结果到输出tensor
                plm.store(output, ub_qk, [0, 0, 0, 64])
                
                # 释放 buffer (free)
                ub_policy_vec.free_buffer(2)
    
    return output


def test_flashattention_complete_flow():
    """Test complete FlashAttention flow with independent policy objects."""
    mlir = _compile_to_mlir(test_flashattention_complete_kernel)
    print("\n=== test_flashattention_complete_flow MLIR ===")
    print(mlir)
    
    # 验证生成了不同的 section
    # 验证存在 scf.for 循环结构
    assert "scf.for" in mlir, "Expected scf.for loop structure"
    
    # 验证循环体内有正确的操作（每个操作在循环体内出现1次）
    assert "pto.tmatmul" in mlir, "Expected pto.tmatmul for QK"
    assert "pto.tmove" in mlir, "Expected pto.tmove for L0C to UB"
    assert "pto.texp" in mlir, "Expected pto.texp in vector section"
    
    # 验证循环范围是 2 次迭代
    assert "to %c2" in mlir, "Expected loop to iterate 2 times"
    # 验证循环范围是 2 次迭代
    assert "to %c2" in mlir, "Expected loop to iterate 2 times"
    
    # 验证 BuffersPolicyDB 生成了 4 个 UB buffer（2个policy × 2个buffer）
    assert "pto.alloc_tile" in mlir, "Expected pto.alloc_tile"
    
    # 只统计 base_addr = 0 的 alloc_tile（policy 创建的 buffer）
    import re
    policy_alloc_pattern = r'pto\.alloc_tile.*base_addr = 0'
    policy_alloc_count = len(re.findall(policy_alloc_pattern, mlir))
    assert policy_alloc_count == 4, f"Expected 4 policy alloc_tile (base_addr=0), got {policy_alloc_count}"
    
    # 验证 buffer 地址（每个 Policy 有独立的地址空间，动态分配）
    # 注意：实际地址可能不同，取决于 MLIR 生成逻辑
    # 我们只验证有 4 个不同的 alloc_tile 调用
    
    # 验证双向同步
    assert "pto.wait_event" in mlir, "Expected pto.wait_event"
    assert "pto.record_event" in mlir, "Expected pto.record_event"
    assert "#pto<event EVENT_ID0>" in mlir, "Expected EVENT_ID0"
    assert "#pto<event EVENT_ID1>" in mlir, "Expected EVENT_ID1"
    assert "#pto<event EVENT_ID2>" in mlir, "Expected EVENT_ID2"
    assert "#pto<event EVENT_ID3>" in mlir, "Expected EVENT_ID3"
    
    # 验证显式初始化：init_consumer() 应该生成 4 个 record_event
    # init_consumer() 调用 free_buffer(0), free_buffer(16), free_buffer(1), free_buffer(17)
    # 每个 free_buffer 生成一个 pto.record_event
    lines = mlir.split('\n')
    
    # 找到 section_vector 的位置
    vector_section_idx = next(i for i, line in enumerate(lines) if 'pto.section.vector' in line)
    
    # 在 section_vector 开始后的前 20 行中查找 init_consumer() 生成的 record_event
    vector_section_start = '\n'.join(lines[vector_section_idx:vector_section_idx+20])
    
    # 提取 init_consumer() 生成的 record_event 的事件 ID
    init_record_event_ids = []
    for line in vector_section_start.split('\n'):
        if 'pto.record_event' in line:
            # 提取 EVENT_ID<数字> 中的数字
            match = re.search(r'#pto<event EVENT_ID(\d+)>', line)
            if match:
                init_record_event_ids.append(int(match.group(1)))
    
    # 验证 init_consumer() 生成了 4 个 record_event
    assert len(init_record_event_ids) == 4, \
        f"Expected 4 record_event from init_consumer(), got {len(init_record_event_ids)}"
    

# ---------------------------------------------------------------------------
# Test Case 9: Multi-level Buffer Combination
# ---------------------------------------------------------------------------

@fe.kernel
def test_multilevel_buffer_kernel(
    q: pl.Tensor[[64, 128], pl.FP16],
    k: pl.Tensor[[64, 128], pl.FP16],
    o: pl.Tensor[[64, 64], pl.FP32],
) -> pl.Tensor[[64, 64], pl.FP32]:
    """Test multi-level buffer combination (L1, L0C, UB) with AIC."""
    with pl.section_cube():
        # L1 buffers for Q and K
        l1_tile_type = plm.TileType(shape=[64, 128], dtype=pl.FP16, target_memory=pl.MemorySpace.Mat)
        l1_q = plm.make_tile(l1_tile_type, addr=0x0000, size=16384)
        l1_k = plm.make_tile(l1_tile_type, addr=0x4000, size=16384)
        
        # L0C buffer for QK result
        l0c_tile_type = plm.TileType(shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Acc)
        l0c_qk = plm.make_tile(l0c_tile_type, addr=0x8000, size=16384)
        
        # UB buffer for QK result
        ub_tile_type = plm.TileType(shape=[64, 64], dtype=pl.FP32, target_memory=pl.MemorySpace.Vec)
        ub_qk = plm.make_tile(ub_tile_type, addr=0xC000, size=16384)
        
        # Load Q and K to L1
        plm.load(l1_q, q, [0, 0])
        plm.load(l1_k, k, [0, 0])
        
        # BMM1: QK = Q @ K^T (L1 -> L0C)
        plm.matmul(l1_q, l1_k, l0c_qk)
        
        # Move L0C to UB
        plm.move(ub_qk, l0c_qk)
        
        # Store UB result to output tensor
        plm.store(o, ub_qk, [0, 0])
    
    return o


def test_multilevel_buffer():
    """Test multi-level buffer combination with AIC memory space."""
    mlir = _compile_to_mlir(test_multilevel_buffer_kernel)
    print("\n=== test_multilevel_buffer MLIR ===")
    print(mlir)
    
    assert "pto.section.cube {" in mlir, "Expected pto.section.cube for AIC"
    assert "pto.alloc_tile" in mlir, "Expected pto.alloc_tile"
    assert mlir.count("pto.alloc_tile") == 4, f"Expected 4 alloc_tile, got {mlir.count('pto.alloc_tile')}"
    assert "pto.tmatmul" in mlir, "Expected pto.tmatmul"
    assert "pto.tmove" in mlir, "Expected pto.tmove"
    assert "pto.tstore" in mlir, "Expected pto.tstore for storing result to output"


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running FlashAttention buffer and sync management tests...")
    
    # Test 1: Double Buffer (AIV version)
    test_double_buffer_policy()
    print("✓ Test 1: Double Buffer (AIV) passed")
    
    # Test 2: Triple Buffer (AIC version)
    test_triple_buffer_policy()
    print("✓ Test 2: Triple Buffer (AIC) passed")
    
    # Test 3b: Triple Buffer KV Reuse Pattern
    test_triple_buffer_kv_reuse_pattern()
    print("✓ Test 3b: Triple Buffer KV Reuse Pattern passed")
    
    # Test 4: Triple Buffer Q Reuse
    test_triple_buffer_q_reuse()
    print("✓ Test 4: Triple Buffer Q Reuse passed")
    
    # Test 5: Four Buffer (AIC version)
    test_four_buffer_policy()
    print("✓ Test 5: Four Buffer (AIC) passed")
    
    # Test 6: Cross Core Both Sync (AIC version)
    test_cross_core_both_sync()
    print("✓ Test 6: Cross Core Both Sync (AIC) passed")
    
    # Test 7: FlashAttention Complete Flow (AIC/AIV mixed)
    test_flashattention_complete_flow()
    print("✓ Test 7: FlashAttention Complete Flow (AIC/AIV mixed) with explicit init passed")
    
    # Test 9: Multi-level Buffer Combination (AIC version)
    test_multilevel_buffer()
    print("✓ Test 9: Multi-level Buffer Combination (AIC) passed")
    
    print("\n✅ All tests passed!")
