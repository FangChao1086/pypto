# Python DSL Syntax Specification

## Overview

PyPTO provides two programming models:

| Model | Import | Memory | Ops return | Use case |
|-------|--------|--------|------------|----------|
| **Manual (non-SSA)** | `import pypto.language.manual as plm` | User allocates tiles with explicit addr/size | `None` (write into pre-allocated `out`) | Full control over memory layout and reuse |
| **SSA** | `import pypto.language as pl` | Compiler-managed | New `Tensor`/`Tile`/`Scalar` | Higher-level, auto-optimized |

**Current focus is the manual model.** Both models share the same type system, decorators, control flow, and system operations from `pypto.language as pl`.

## Quick Start — Manual Matmul with Double Buffer

```python
import pypto.frontend as fe
import pypto.language as pl
import pypto.language.manual as plm

M = pl.DynVar('M')
K = pl.DynVar('K')
N = pl.DynVar('N')

@fe.kernel
def matmul_db(
    a: pl.Tensor[[M, K], pl.FP16],
    b: pl.Tensor[[K, N], pl.FP16],
    c: pl.Tensor[[M, N], pl.FP32],
) -> pl.Tensor[[M, N], pl.FP32]:
    # 1. Allocate tiles with explicit address and size
    tile_type_a = plm.TileType(shape=[128, 128], dtype=pl.FP16,
                               target_memory=pl.MemorySpace.Mat, blayout=2, slayout=1)
    tile_a_ping = plm.make_tile(tile_type_a, addr=0x00000, size=32768)
    tile_a_pong = plm.make_tile(tile_type_a, addr=0x10000, size=32768)

    tile_type_c = plm.TileType(shape=[128, 128], dtype=pl.FP32,
                               target_memory=pl.MemorySpace.Acc, blayout=2, slayout=1,
                               fractal=1024, valid_shape=[-1, -1])
    tile_c = plm.make_tile(tile_type_c, addr=0x00000, size=65536)

    tile_a_buf = (tile_a_ping, tile_a_pong)   # tuple for double-buffer dispatch

    with pl.section_cube():
        M_dim = pl.tensor.dim(a, 0)
        K_dim = pl.tensor.dim(a, 1)
        event_ids = (0, 1)

        for i in pl.range(0, M_dim, 128):
            for k in pl.range(0, K_dim, 128):
                buf_idx = (k // 128) % 2

                plm.load(tile_a_buf[buf_idx], a, [i, k])
                pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1,
                                   event_id=event_ids[buf_idx])
                pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1,
                                   event_id=event_ids[buf_idx])
                # ... move, matmul, sync, store ...
    return c
```

---

## Module Structure

```python
import pypto.language as pl           # Types, decorators, control flow, tensor/system ops
import pypto.language.manual as plm   # Manual (non-SSA) tile operations
import pypto.frontend as fe           # @fe.kernel, fe.compile(), fe.launch()
```

## Type System

### Scalar Types

```python
x: pl.INT64
y: pl.FP32
z: pl.BOOL
```

| Category | Types |
|----------|-------|
| **Integers** | `INT4`, `INT8`, `INT16`, `INT32`, `INT64` |
| **Unsigned** | `UINT4`, `UINT8`, `UINT16`, `UINT32`, `UINT64` |
| **Float** | `FP4`, `FP8E4M3FN`, `FP8E5M2`, `FP16`, `FP32` |
| **Brain/Hisi Float** | `BF16`, `HF4`, `HF8` |
| **Other** | `BOOL`, `INDEX` |

### Tensor Type

```python
a: pl.Tensor[[4, 8], pl.FP32]              # Fixed shape
b: pl.Tensor[[M, N], pl.FP16]              # Dynamic shape (DynVar)
c: pl.Tensor[[64, 128], pl.FP32, pl.NZ]    # With layout (ND/DN/NZ)
```

### Tile Type

```python
t: pl.Tile[[16, 16], pl.FP16]
```

### Scalar Wrapper

```python
s: pl.Scalar[pl.FP32]
```

### Pointer Type

```python
p: pl.Ptr[pl.FP32]    # Raw global-memory pointer → !pto.ptr<f32>
```

### Dynamic Shape Variables

```python
M = pl.DynVar('M')
N = pl.DynVar('N')

@pl.function
def kernel(a: pl.Tensor[[M, N], pl.FP16]) -> ...:
    M_dim = pl.tensor.dim(a, 0)   # runtime value
```

### Parameter Directions

```python
@pl.function
def kernel(
    qi: pl.Tensor[[16, 128], pl.BF16],                  # In (default)
    output: pl.InOut[pl.Tensor[[16, 128], pl.FP32]],     # InOut
    result: pl.Out[pl.Tensor[[16, 128], pl.FP32]],       # Out
) -> pl.Tensor[[16, 128], pl.FP32]:
    ...
```

---

## Decorators

### @fe.kernel / @pl.function / @pl.program

```python
@fe.kernel                    # Wrap into ir.Program, ready for fe.compile()
def my_kernel(...): ...

@pl.function                  # Parse into ir.Function
def my_func(...): ...

@pl.function(type=pl.FunctionType.Orchestration)
def orchestrator(...): ...

@pl.program                   # Parse class methods into ir.Program
class MyModel:
    @pl.function
    def main(self, ...): ...
```

### @pl.inline / @pl.func

```python
@pl.inline                    # Statement-level inlining at call site
def helper(x: pl.Tensor[...]) -> ...: ...

@pl.func                      # Scalar helper → func.func in MLIR
def compute_offset(base: pl.INDEX, stride: pl.INDEX) -> pl.INDEX:
    return base + stride
```

---

## Control Flow

### For Loops

```python
# Sequential
for i in pl.range(start, stop, step):
    ...

# Parallel
for i in pl.parallel(start, stop, step):
    ...

# Unroll (compile-time constants only)
for i in pl.unroll(0, 4, 1):
    ...

# Chunked (split into outer/inner loops)
for i in pl.range(0, 10, 1, chunk=5):
    ...
```

### Loop-Carried Values (SSA iter_args)

```python
sum_init: pl.INT64 = 0
for i, (sum,) in pl.range(0, n, 1, init_values=(sum_init,)):
    sum = pl.yield_(sum + i)
sum_final = sum
```

### While Loop

```python
for (x,) in pl.while_(init_values=(0,)):
    pl.cond(x < 10)
    x = x + 1
    x_out = pl.yield_(x)
```

### If Statement (SSA-style)

```python
if condition:
    y1 = pl.yield_(value1)
else:
    y1 = pl.yield_(value2)
```

### Sections

```python
with pl.section_cube():       # Cube core execution
    ...

with pl.section_vector():     # Vector core execution
    ...
```

---

## Tensor Operations (`pl.tensor.*`)

Used for high-level tensor manipulation before loading into tiles.

```python
t = pl.tensor.create_tensor([64, 128], dtype=pl.FP32)   # Allocate tensor
s = pl.tensor.dim(a, 0)                                  # Query shape dim
v = pl.tensor.view(a, shape=[32, 32], offset=[0, 0])     # Slice/view
r = pl.tensor.matmul(a, b, out_dtype=pl.FP32)            # Matmul
```

| Function | Signature | Description |
|----------|-----------|-------------|
| `create_tensor` | `(shape, dtype) → Tensor` | Allocate new tensor |
| `dim` | `(tensor, axis) → Scalar` | Query dimension size |
| `view` | `(tensor, shape, offset) → Tensor` | Create subview |
| `matmul` | `(lhs, rhs, out_dtype?, a_trans?, b_trans?) → Tensor` | Matrix multiply |
| `mul/add/sub/div` | `(lhs, rhs) → Tensor` | Elementwise arithmetic |
| `cast` | `(tensor, dtype, mode?) → Tensor` | Type cast |
| `exp` | `(tensor) → Tensor` | Exponential |
| `reshape` | `(tensor, shape) → Tensor` | Reshape |
| `transpose` | `(tensor, axis1, axis2) → Tensor` | Transpose |
| `row_max/row_sum` | `(tensor) → Tensor` | Row-wise reduction |
| `assemble` | `(target, source, offset) → Tensor` | Copy into subregion |

## Pointer Operations (`pl.make_tensor`, `pl.addptr`)

```python
@pl.function
def kernel(workspace: pl.Ptr[pl.FP32]):
    buf0: pl.Ptr[pl.FP32] = pl.addptr(workspace, 0)
    buf1: pl.Ptr[pl.FP32] = pl.addptr(workspace, 1024)
    view0: pl.Tensor[[32, 32], pl.FP32] = pl.make_tensor(buf0, [32, 32], [32, 1])
```

## System Operations (`pl.system.*`)

Hardware synchronization primitives:

```python
pl.system.sync_src(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
pl.system.sync_dst(set_pipe=pl.PipeType.MTE2, wait_pipe=pl.PipeType.MTE1, event_id=0)
pl.system.bar_v()       # Vector barrier
pl.system.bar_m()       # Matrix barrier
pl.system.bar_all()     # Global barrier
```

Pipeline types: `MTE1`, `MTE2`, `MTE3`, `M`, `V`, `S`, `FIX`, `ALL`

---

## Manual Operations (`plm.*`)

`import pypto.language.manual as plm`

All manual operations write into pre-allocated output tiles (non-SSA). The user explicitly manages memory addresses and buffer reuse.

### Tile Allocation

```python
@dataclass
class TileType:
    shape: Sequence[int]         # Tile dimensions
    dtype: DataType              # Element type
    target_memory: MemorySpace   # Vec (default), Mat, Left, Right, Acc
    valid_shape: list[int] | None = None   # Valid shape (optional)
    blayout: int | None = None   # Block layout: 0=none_box, 1=row_major, 2=col_major
    slayout: int | None = None   # Scatter layout: 0=none_box, 1=row_major, 2=col_major
    fractal: int | None = None   # Fractal size
    pad: int | None = None       # Pad mode: 0=null, 1=zero, 2=max, 3=min

tile = plm.make_tile(tile_type, addr=0x00000, size=32768)
```

**Double-buffer pattern** — allocate ping/pong at different addresses:

```python
tile_ping = plm.make_tile(tile_type, addr=0x00000, size=32768)
tile_pong = plm.make_tile(tile_type, addr=0x10000, size=32768)
tile_buf = (tile_ping, tile_pong)

buf_idx = (k // 128) % 2
plm.load(tile_buf[buf_idx], tensor, [i, k])   # variable-index dispatch
```

### Memory Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `load` | `(out, tensor, offsets, shapes?) → None` | GM → tile |
| `load_tile` | `(out, tensor, tile_offsets) → None` | GM → tile (tile-relative offsets) |
| `store` | `(tensor, tile, offsets, shapes?) → Tensor` | tile → GM |
| `store_tile` | `(tensor, tile, tile_offsets) → Tensor` | tile → GM (tile-relative) |
| `l0c_store` | `(tile, offsets, shapes, tensor) → Tensor` | Acc → GM |
| `move` | `(tile, target_memory, out, transpose?) → None` | Move between memory levels |
| `ub_copy` | `(tile, out) → None` | Copy within UB |
| `full` | `(value, out) → None` | Fill tile with scalar |
| `fillpad` | `(tile, out) → None` | Pad tile |

```python
plm.load(tile_a, tensor_a, [i, k])                       # Load from GM
plm.move(tile_a, pl.MemorySpace.Left, tile_a_compute)    # Mat → Left
plm.store(output_tensor, tile_c, [i, j])                 # Tile → GM
plm.l0c_store(tile_c, [i, j], [128, 128], output_tensor) # Acc → GM
```

### Elementwise Operations

**Tile × Tile** — all `(lhs, rhs, out) → None`:
`add`, `sub`, `mul`, `div`, `rem`, `maximum`, `minimum`, `and_`, `or_`, `shl`, `shr`

**Tile × Scalar** — all `(lhs, scalar, out) → None`:
`adds`, `subs`, `muls`, `divs`, `rems`, `ands`, `ors`, `shls`, `shrs`, `maxs`, `mins`, `lrelu`

**Unary** — all `(tile, out) → None`:
`neg`, `exp`, `sqrt`, `rsqrt`, `recip`, `log`, `abs`, `relu`, `not_`

Special: `cast(tile, target_type, out, mode="round") → None`

### Ternary / Multi-Input

`xor(lhs, rhs, tmp, out)`, `xors(lhs, scalar, tmp, out)`, `prelu(tile, slope, tmp, out)`, `addc(lhs, rhs, rhs2, out)`, `subc(lhs, rhs, rhs2, out)`, `addsc(lhs, scalar, rhs2, out)`, `subsc(lhs, scalar, rhs2, out)`, `sel(mask, lhs, rhs, out)`, `sels(lhs, rhs, mode, out)`

### Comparison

`cmp(lhs, rhs, out, cmp_type=0)`, `cmps(lhs, scalar, out, cmp_type=0)` — cmp_type: EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5

### Reduction (require `tmp` scratch buffer)

`row_max(tile, tmp, out)`, `row_sum(tile, tmp, out)`, `row_min(tile, tmp, out)`

### Broadcast / Expansion

`row_expand(src, out)`, `row_expand_add/sub/mul/div(tile, row_vec, out)`, `col_expand(col_vec, out)`, `col_expand_mul/div/sub(tile, col_vec, out)`, `expands(scalar, out)`

### Matrix Operations

| Function | Signature | Description |
|----------|-----------|-------------|
| `matmul` | `(lhs, rhs, out)` | `out = lhs @ rhs` |
| `matmul_acc` | `(acc, lhs, rhs, out)` | `out = acc + lhs @ rhs` |
| `matmul_bias` | `(lhs, rhs, bias, out)` | `out = lhs @ rhs + bias` |
| `gemv` | `(lhs, rhs, out)` | Matrix-vector multiply |
| `gemv_acc` | `(acc, lhs, rhs, out)` | GEMV with accumulation |
| `gemv_bias` | `(lhs, rhs, bias, out)` | GEMV with bias |

### Layout

```python
plm.reshape(tile, shape=[64, 64], out=out_tile)
plm.transpose(tile, axis1=0, axis2=1, out=out_tile)
```

### Query (same as `pl.*`)

```python
plm.get_block_idx()        # Current block index (Scalar)
plm.get_block_num()        # Total block count (Scalar)
plm.get_subblock_idx()     # Subblock index: 0 or 1 (Scalar)
```

---

## Expressions

### Variables and Constants

```python
x                          # Variable reference
42                         # Integer literal
3.14                       # Float literal
pl.const(0, pl.INT64)     # Typed constant
```

**Closure variables:** Names not in DSL scope resolve from enclosing Python scope (`int`, `float`, `bool`, `list`, `tuple`).

### Binary / Unary Operators

| Python | IR | Category |
|--------|----|----------|
| `+`, `-`, `*`, `//`, `%`, `/`, `**` | Add, Sub, Mul, FloorDiv, FloorMod, FloatDiv, Pow | Arithmetic |
| `==`, `!=`, `<`, `<=`, `>`, `>=` | Eq, Ne, Lt, Le, Gt, Ge | Comparison |
| `and`, `or`, `^` | And, Or, Xor | Logical |
| `&`, `\|`, `<<`, `>>` | BitAnd, BitOr, BitShiftLeft, BitShiftRight | Bitwise |
| `-x`, `~x`, `not x` | Neg, BitNot, Not | Unary |
| `abs(x)`, `min(a,b)`, `max(a,b)` | Abs, Min, Max | Built-in |

---

## Functions

```python
# Single return
def func(x: pl.INT64) -> pl.INT64:
    return x + 1

# Multiple return
def func(x: pl.INT64) -> tuple[pl.INT64, pl.INT64]:
    return x + 1, x * 2

# No return
def func(x: pl.INT64):
    pass
```

### Function Types

| Type | Description |
|------|-------------|
| `pl.FunctionType.Opaque` | Default, unspecified |
| `pl.FunctionType.Orchestration` | Host/AICPU control flow |
| `pl.FunctionType.InCore` | AICore sub-graph |

### Tiling Parameters

Group related scalars into a struct (last parameter, at most one):

```python
from pypto.language.typing.tiling import Array

class Tiling:
    m: int                  # → Scalar[INT32]
    n: int                  # → Scalar[INT32]
    offsets: Array[int, 3]  # → 3 × Scalar[INT32]

@pl.function
def kernel(x: pl.Tensor[[64], pl.FP32], tiling: Tiling) -> pl.Scalar[pl.INT32]:
    n = tiling.n                  # resolves to tiling_n
    off = tiling.offsets[1]       # resolves to tiling_offsets_1
    return off
```

## Cross-Module Function Reuse

```python
@pl.function
def softmax(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]: ...

@pl.program
class MyModel:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y = softmax(x)   # auto-added to Program as GlobalVar call
        return y
```

`@pl.inline` expands the body at each call site instead of generating a separate function.

---

## Compilation and Execution

```python
import pypto.frontend as fe

@fe.kernel
def my_kernel(...): ...

compiled = fe.compile(my_kernel, arch="dav-c220-cube")
fe.launch(None, num_cores, compiled, *args)
```

## References

- [IR Overview](../ir/00-overview.md) — Core IR structures
- [IR Parser](../ir/07-parser.md) — Parsing Python syntax back to IR
- [Operator Registration](../ir/05-operators.md) — Op system and type inference
- [Architecture](../ARCHITECTURE.md) — System architecture overview
