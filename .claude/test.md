# FlashAttention Test Status

## Test File
`tests/ut/frontend/test_fa.py`

## Current Status: PASSING

Both `fa_kt_kernel` and `fa_k_kernel` pass precision checks.

| Kernel | Shapes | Max Diff | Tolerance |
|--------|--------|----------|-----------|
| fa_kt_kernel | (64,64,64), (64,128,64) | 0.0005 | atol=1e-2 |
| fa_k_kernel | (64,128,64), (64,256,64), (64,512,64) | 0.02 | atol=5e-2 |

fa_k_kernel has slightly worse precision due to the in-kernel transpose (`plm.move(..., transpose=True)`).

## Root Causes Fixed

### 1. Missing Tile Layout Parameters (CRITICAL)

Cube matmul tiles MUST specify `blayout` and `slayout` matching the hardware NZ format. Without these, data is loaded/stored in wrong format → completely wrong matmul results.

**Required layouts for Cube matmul:**

| Memory | blayout | slayout | fractal | Notes |
|--------|---------|---------|---------|-------|
| MAT (L1) | 2 (col_major) | 1 (row_major) | 512 | NZ format |
| LEFT (L0A) | 1 (row_major) | 1 (row_major) | 512 | Left operand |
| RIGHT (L0B) | 1 (row_major) | 2 (col_major) | 512 | Right operand |
| ACC (L0C) | 2 (col_major) | 1 (row_major) | **1024** | FP32 accumulator |
| VEC (UB) | default | default | 512 | No special layout |

**Reference**: `tests/ut/frontend/test_dynamic_matmul.py` uses correct layouts.

### 2. Missing wait_cross_core in fa_kt_kernel

`wait_cross_core(M, P_READY)` was commented out. Cube read uninitialized `p_buf` → `pv_buf = 0` → `o = 0`.

## PLM Parser Convention (CRITICAL)

The AST parser for `plm.*` manual ops: **first positional arg = output tile**. Parser moves arg[0] to last position in IR call.

```python
plm.sub(OUT, lhs, rhs)     # → manual.sub(lhs, rhs, OUT)
plm.muls(OUT, tile, scalar) # → manual.muls(tile, scalar, OUT)
plm.matmul(OUT, left, right) # → manual.matmul(left, right, OUT)
plm.row_max(OUT, tile, tmp)  # → manual.row_max(tile, tmp, OUT)
plm.row_expand(OUT, src)     # → manual.row_expand(src, OUT)
plm.cast(OUT, tile, target_type=pl.FP16, mode="round")
```

**Exceptions** (parsed as block ops, NO reordering):
- `plm.make_tile(tile_type, addr=X, size=Y)` → `block.make_tile(...)`
- `plm.l0c_store(tile, offsets, shapes, tensor)` → `block.l0c_store(tile, offsets, shapes, tensor)`

## Sync Rules Summary

### Intra-pipe (same core)
| Before | After | Sync |
|--------|-------|------|
| TLOAD (MTE2) | TMOV (MTE1) | `sync(MTE2, MTE1)` |
| TLOAD (MTE2) | TVEC (V) | `sync(MTE2, V)` |
| TMOV (MTE1) | TMATMUL (M) | `sync(MTE1, M)` |
| TMATMUL (M) | TSTORE_ACC (FIX) | `sync(M, FIX)` **CRITICAL** |
| TVEC (V) | TSTORE_VEC (MTE3) | `sync(V, MTE3)` |

### Cross-core (Cube ↔ Vector)
```python
# Cube signals Vector:
pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=N)
# Vector waits:
pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=N)

# Vector signals Cube:
pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=N)
# Cube waits:
pl.system.wait_cross_core(pipe=pl.PipeType.M, event_id=N)
```

## Kernel Architecture

### FA Algorithm (single-core, Cube+Vector streaming)
```
For each KV tile j:
  [Cube]   S_j = Q @ K_j^T              # QK matmul
  [Cube]   l0c_store S_j → qk_buf       # ACC → GM
  [Cube]   set_cross_core(FIX, QK_READY) # signal Vector
  [Cube]   wait_cross_core(M, P_READY)   # MUST wait for P
  [Vector] wait_cross_core(V, QK_READY)  # wait for QK
  [Vector] load qk_buf → qk_vec         # GM → VEC
  [Vector] FlashSoftmax(qk_vec) → p_f16  # online softmax
  [Vector] store p_f16 → p_buf           # VEC → GM
  [Vector] set_cross_core(MTE3, P_READY) # signal Cube
  [Cube]   P_j @ V_j → pv_acc           # PV matmul
  [Cube]   l0c_store pv_acc → pv_buf    # ACC → GM
  [Cube]   set_cross_core(FIX, PV_READY) # signal Vector
  [Vector] wait_cross_core(V, PV_READY)  # wait for PV
  [Vector] GlobalUpdate: O += PV_j       # running accumulation
[Vector] O /= global_sum                  # final normalization
```

### Memory Layout
- **Cube MAT**: 4 tiles × 8KB = 32KB (q_mat, k_mat, p_mat, v_mat)
- **Cube LEFT**: 2 tiles × 8KB = 16KB (q_left@0, p_left@8K)
- **Cube RIGHT**: 2 tiles × 8KB = 16KB (k_right@0, v_right@8K)
- **Cube ACC**: 2 tiles × 16KB = 32KB (qk_acc@0, pv_acc@16K)
- **Vector VEC**: 8 tiles, total 60KB (within 192KB limit)

### Two Versions
1. **fa_kt_kernel**: K^T pre-transposed `kt: [D, Skv]` — best precision (max diff 0.0005)
2. **fa_k_kernel**: K not transposed `k: [Skv, D]` — uses `plm.move(..., transpose=True)`, slightly worse precision (max diff ~0.02)

## Compilation Flow
```bash
source compile.sh  # sets up environment
python3 tests/ut/frontend/test_fa.py
```

`fe.compile(kernel, arch="a3")` → PTOCodegen → `.pto` MLIR → `ptoas --enable-insert-sync --pto-level=level3 --pto-arch=a3` → `.cpp` → `bisheng` → `.so`

## Known Issues
1. `plm.muls(tile, tile, scalar)` — parser reorders first arg as output. Must write `plm.muls(OUT, input_tile, scalar)`.
2. `math.sqrt()` not supported in kernel body — precompute as module-level constant.
3. `pl.tensor.dim()` inline in expressions produces missing SSA operand — always assign to variable first.
4. `plm.cast` needs explicit `mode="round"` kwarg; default empty string causes codegen error.
5. `get_block_idx()` returns i64; needs `pl.block.index_cast()` for index arithmetic.
