# Ascend NPU Hardware Knowledge

## Core Architecture

### Core Types and Ratios
- **Cube cores**: Matrix computation (TMATMUL). Ratio Cube:Vector = 1:2.
- **Vector cores**: Vector/scalar ops (TVEC, element-wise). Each Cube core has 2 paired Vector sub-blocks.
- `get_block_idx()` → current core index (returns i64, needs `index_cast` for index arithmetic).
- `get_subblock_idx()` → Vector sub-block index: 0 or 1.
- `get_block_num()` → total core count.
- For Cube-only: `get_block_idx()` = Cube core index.
- For Mix (Cube+Vector): `get_block_idx() // 2` = corresponding Cube core index.

### Memory Hierarchy

| Memory | Space Enum | Size (A2/A3) | Size (A5) | Purpose |
|--------|-----------|-------------|-----------|---------|
| **VEC (UB)** | `Vec` | 192 KB | 248 KB | Vector compute buffer |
| **MAT (L1)** | `Mat` | 512 KB | 512 KB | Matrix load buffer (from GM) |
| **LEFT (L0A)** | `Left` | 64 KB | 64 KB | Left operand for Cube matmul |
| **RIGHT (L0B)** | `Right` | 64 KB | 64 KB | Right operand for Cube matmul |
| **ACC (L0C)** | `Acc` | 128 KB | 256 KB | Accumulator output from Cube matmul |

### Data Flow for Matmul
```
GM → TLOAD → MAT (L1) → TMOV → LEFT/RIGHT (L0A/L0B) → TMATMUL → ACC (L0C) → TSTORE → GM
```

For Vector ops:
```
GM → TLOAD → VEC (UB) → TVEC ops → VEC (UB) → TSTORE → GM
```

---

## Tile Layout (blayout / slayout) — CRITICAL for Correct Matmul

**Omitting these parameters causes completely wrong matmul results (not just precision loss).**

The Cube matmul hardware operates on data in **NZ (fractal) format**. The `blayout` (block layout) and `slayout` (scatter layout) parameters control how data is physically arranged in each memory buffer. These MUST match the hardware expectations.

### Required Layout Per Memory Space

| Memory | blayout | slayout | fractal | Code |
|--------|---------|---------|---------|------|
| **MAT (L1)** | 2 (col_major) | 1 (row_major) | 512 | `blayout=2, slayout=1` |
| **LEFT (L0A)** | 1 (row_major) | 1 (row_major) | 512 | `blayout=1, slayout=1` |
| **RIGHT (L0B)** | 1 (row_major) | 2 (col_major) | 512 | `blayout=1, slayout=2` |
| **ACC (L0C)** | 2 (col_major) | 1 (row_major) | **1024** (FP32) | `blayout=2, slayout=1, fractal=1024` |
| **VEC (UB)** | default | default | 512 | (no explicit layout needed) |

### Layout Parameter Values

- `blayout` (Block Layout): `0=none_box, 1=row_major, 2=col_major`
- `slayout` (Scatter Layout): `0=none_box, 1=row_major, 2=col_major`
- `fractal`: 512 for FP16 tiles, **1024 for FP32 ACC tiles**

### What blayout and slayout Mean

The Cube processes data in 16×16 fractal blocks. A tile (e.g. 64×64) is divided into a grid of fractal blocks. The two layout parameters control how these blocks are organized:

- **blayout (block layout)**: How 16×16 fractal blocks are arranged in the overall tile grid.
  - `row_major (1)`: Blocks stored row by row (block[0,0], block[0,1], ...)
  - `col_major (2)`: Blocks stored column by column (block[0,0], block[1,0], ...) — this is the NZ format

- **slayout (scatter layout)**: How elements are scattered within each 16×16 fractal block.
  - `none_box (0)`: No scatter, simple packed layout
  - `row_major (1)`: Elements within a fractal block stored row-major
  - `col_major (2)`: Elements within a fractal block stored column-major

### Why These Specific Layouts

- **MAT/ACC use NZ format** (blayout=2, slayout=1): TLOAD from GM stores data in NZ (column-major block order, row-major within each block). TMOV from MAT to L0A/L0B expects this format. ACC output is also in NZ format.
- **LEFT uses row-major** (blayout=1, slayout=1): The left matrix operand A in A×B is stored with rows accessible sequentially for the Cube's inner product computation.
- **RIGHT uses col-major scatter** (blayout=1, slayout=2): The right matrix operand B in A×B has columns scattered for the Cube's dot-product access pattern.
- **VEC uses simple layout**: Vector ops work on contiguous data; no fractal format needed.

### Complete Cube Matmul Tile Example (64×64)

```python
# MAT (load buffer)
mat_type = plm.TileType(shape=[64, 64], dtype=pl.FP16,
    target_memory=pl.MemorySpace.Mat, blayout=2, slayout=1)
a_mat = plm.make_tile(mat_type, addr=0, size=8192)
b_mat = plm.make_tile(mat_type, addr=8192, size=8192)

# LEFT / RIGHT (compute buffers)
a_left = plm.make_tile(plm.TileType(shape=[64, 64], dtype=pl.FP16,
    target_memory=pl.MemorySpace.Left, blayout=1, slayout=1), addr=0, size=8192)
b_right = plm.make_tile(plm.TileType(shape=[64, 64], dtype=pl.FP16,
    target_memory=pl.MemorySpace.Right, blayout=1, slayout=2), addr=0, size=8192)

# ACC (accumulator — note fractal=1024 for FP32)
acc_type = plm.TileType(shape=[64, 64], dtype=pl.FP32,
    target_memory=pl.MemorySpace.Acc, blayout=2, slayout=1, fractal=1024)
c_acc = plm.make_tile(acc_type, addr=0, size=16384)
```

---

## Pipeline Architecture

### Pipeline Types (PipeType)

| Pipeline | Enum | Operations |
|----------|------|-----------|
| **MTE2** | `PipeType.MTE2` | TLOAD (GM → L1/UB) |
| **MTE1** | `PipeType.MTE1` | TMOV_M2L (MAT→LEFT), TMOV_M2B (MAT→RIGHT) |
| **MTE3** | `PipeType.MTE3` | TSTORE (UB → GM) for Vector section |
| **M (PIPE_M)** | `PipeType.M` | TMATMUL (Cube matmul) |
| **V (PIPE_V)** | `PipeType.V` | TVEC (Vector compute), TVECWAIT_EVENT |
| **FIX (PIPE_FIX)** | `PipeType.FIX` | TSTORE_ACC (ACC → GM), TMOV_M2S, TMOV_V2M |
| **S (PIPE_S)** | `PipeType.S` | Scalar operations |
| **ALL** | `PipeType.ALL` | All pipelines (for barrier) |

### Key Pipeline Mapping
- `plm.load()` → TLOAD → **MTE2**
- `plm.move(src, target_memory=Left/Right)` → TMOV → **MTE1**
- `plm.matmul()` → TMATMUL → **M**
- `plm.l0c_store()` → TSTORE_ACC → **FIX**
- `plm.store()` from VEC → TSTORE → **MTE3**
- `plm.add/sub/mul/exp/...` on VEC tiles → TVEC → **V**

---

## Synchronization

### Intra-Pipeline Sync (set_flag / wait_flag)
Used to synchronize between pipelines on the SAME core.

```python
pl.system.sync_src(set_pipe=PipeType.X, wait_pipe=PipeType.Y, event_id=N)  # X signals Y
pl.system.sync_dst(set_pipe=PipeType.X, wait_pipe=PipeType.Y, event_id=N)  # Y waits for X
```

Always use as a pair (sync_src + sync_dst with same args).

### Cross-Core Sync (FFTS: set_cross_core / wait_cross_core)
Used for Cube ↔ Vector synchronization across paired cores.

```python
pl.system.set_cross_core(pipe=PipeType.FIX, event_id=0)   # Cube signals Vector
pl.system.wait_cross_core(pipe=PipeType.V, event_id=0)     # Vector waits for Cube
```

Requires `pto.set_ffts` in MLIR (auto-injected by `jit.py` when `has_cross_sync=True`).

### Barriers
```python
pl.system.bar_v()     # Vector cores synchronize
pl.system.bar_m()     # Cube cores synchronize
pl.system.bar_all()   # All cores synchronize (both Cube and Vector)
```

### Critical Sync Rules

#### Rule 1: MTE2 → MTE1 before TMOV
```python
plm.load(mat_tile, tensor, offsets)
pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
plm.move(left_tile, mat_tile, target_memory=pl.MemorySpace.Left)
```

#### Rule 2: MTE1 → M before TMATMUL
```python
plm.move(left_tile, mat_tile, target_memory=pl.MemorySpace.Left)
plm.move(right_tile, mat_tile, target_memory=pl.MemorySpace.Right)
pl.system.sync_src(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
pl.system.sync_dst(set_pipe=pl.PipeType.MTE1, wait_pipe=pl.PipeType.M, event_id=0)
plm.matmul(acc, left_tile, right_tile)
```

#### Rule 3: M → FIX before l0c_store (CRITICAL!)
**Without this, l0c_store reads ACC before matmul completes → garbage data.**
```python
plm.matmul(acc, left, right)
pl.system.sync_src(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
pl.system.sync_dst(set_pipe=pl.PipeType.M, wait_pipe=pl.PipeType.FIX, event_id=0)
plm.l0c_store(acc, offsets, shapes, output_tensor)
```

#### Rule 4: Backward sync in loops
When a buffer is reused across loop iterations, wait for previous iteration's pipelines.

### Event ID Usage
- Each `(set_pipe, wait_pipe)` combination has its own independent event_id namespace (0–7).
- For double buffer: ping uses event_id=0, pong uses event_id=1.
- Different pipe combos can safely reuse the same event_id values.

---

## FlashAttention Hardware Pattern

### Cube+Vector Cross-Core Pipeline
```
Cube Section:                          Vector Section:
  Q @ K^T → ACC                         (wait QK_READY)
  l0c_store ACC → qk_buf                load qk_buf → VEC
  set_cross_core(FIX, QK_READY) ──────► FlashSoftmax on VEC
  wait_cross_core(M, P_READY)           store P → p_buf
  ◄────────────────────────────────────── set_cross_core(MTE3, P_READY)
  load p_buf, load V
  P @ V → ACC
  l0c_store ACC → pv_buf
  set_cross_core(FIX, PV_READY) ──────► (wait PV_READY)
                                         load pv_buf → running_O
                                         GlobalUpdate: O += PV
```

**The `wait_cross_core(M, P_READY)` is mandatory.** Without it, Cube reads stale p_buf.

### Vector Section: Sub-Block Processing
Each Vector core has 2 sub-blocks (sub_id = 0 or 1). Each processes TILE/2 rows.

---

## Compilation Flow
```
Python kernel (@fe.kernel)
  → AST Parser → PyPTO IR (Program)
  → PTOCodegen → PTO MLIR (.pto file)
  → ptoas --pto-arch=a3 --pto-level=level3 --enable-insert-sync → C++ (.cpp)
  → bisheng --cce-aicore-arch=dav-c220-cube → Shared library (.so)
  → fe.launch() → NPU execution
```

### ptoas Flags
- `--pto-arch=a3|a5`: Target architecture
- `--pto-level=level3`: Enable manual address allocation (required for `addr=` in alloc_tile)
- `--enable-insert-sync`: Auto-insert missing pipeline sync (helps but doesn't cover all cases)
