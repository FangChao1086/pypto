# PyPTO Architecture

PyPTO is a Python frontend and compiler for the PTOAS MLIR, targeting Ascend NPU accelerators. It transforms high-level Python kernel code into hardware-optimized PTO-ISA MLIR or C++ code.

## System Overview

```
User Python Code (@kernel / @function)
        │
        ▼
┌─────────────────┐
│  Language Layer  │  Python DSL: decorators, type annotations, ops
│  (python/pypto/ │  AST parsing: Python AST → PyPTO IR
│   language/)    │
└────────┬────────┘
         ▼
┌─────────────────┐
│    IR Layer      │  Immutable IR nodes: Program, Function, Stmt, Expr, Type
│  (src/ir/,      │  Operator registry, type inference, reflection
│   include/pypto/ │
│   ir/)          │
└────────┬────────┘
         ▼
┌─────────────────┐
│  Pass Pipeline   │  Transformation passes: sync insertion, verification
│  (src/ir/       │  Property-based verification, PassContext instrumentation
│   transforms/)  │
└────────┬────────┘
         ▼
┌─────────────────┐
│   Code Generation│  PTOCodegen → PTO MLIR string
│  (src/codegen/) │  CCECodegen → C++ code
└────────┬────────┘  OrchestrationCodegen → control flow MLIR
         ▼
┌─────────────────┐
│    Backend       │  Hardware abstraction: SoC, memory hierarchy
│  (src/backend/) │  Per-backend op codegen registration
└────────┬────────┘
         ▼
    ptoas / bisheng → .so (executable kernel)
```

---

## Directory Structure

### `include/pypto/` — C++ Public Headers

| Directory | Purpose |
|-----------|---------|
| `core/` | Fundamental types: `DataType`, `Error` hierarchy, `CHECK`/`INTERNAL_CHECK`, logging |
| `ir/` | IR node hierarchy: `Expr`, `Stmt`, `Type`, `Function`, `Program`, `MemRef` |
| `ir/reflection/` | Field descriptors for generic traversal, serialization, comparison |
| `ir/transforms/` | Pass infrastructure: `Pass`, `PassContext`, `IRVisitor`, `IRMutator` |
| `ir/transforms/base/` | Visitor/Mutator base classes, `ExprFunctor`/`StmtFunctor` dispatch |
| `ir/serialization/` | MessagePack serialization/deserialization |
| `ir/verifier/` | Property verification framework |
| `ir/reporter/` | Memory and analysis report generation |
| `codegen/` | Code generation base class and PTO/CCE/Orchestration generators |
| `backend/common/` | `Backend` interface, `SoC` hardware model, `BackendRegistry` |
| `backend/910B_PTO/` | 910B PTO backend (assembly-level codegen) |
| `backend/910B_CCE/` | 910B CCE backend (C++ codegen) |

### `src/` — C++ Implementation

| Directory | Purpose |
|-----------|---------|
| `core/` | Error handling, backtrace capture |
| `ir/` | IR node implementations (`expr.cpp`, `stmt.cpp`, `type.cpp`, `builder.cpp`) |
| `ir/op/` | Operator definitions: `tensor_ops/`, `block_ops/`, `manual_ops/`, `sync_ops/` |
| `ir/transforms/` | Pass infrastructure, visitor/mutator, structural comparison, `insert_sync_pass` |
| `ir/serialization/` | Serializer/deserializer (MessagePack format) |
| `ir/verifier/` | Type check, SSA verification, no-nested-call verification |
| `ir/reporter/` | Memory usage report generation |
| `codegen/pto/` | `PTOCodegen`: IR → PTO MLIR string (1936 lines, largest file) |
| `codegen/cce/` | `CCECodegen`: IR → C++ code, with code context/emitter/type converter |
| `codegen/orchestration/` | Orchestration function codegen |
| `backend/common/` | Backend base, registry, SoC config |
| `backend/910B_PTO/` | PTO backend ops + manual ops registration |
| `backend/910B_CCE/` | CCE backend ops registration |

### `python/` — Python Layer

| Directory | Purpose |
|-----------|---------|
| `bindings/` | nanobind C++→Python bindings (`ir.cpp`, `passes.cpp`, `codegen.cpp`, etc.) |
| `pypto/pypto_core/` | Type stubs (`.pyi`) for IDE autocomplete |
| `pypto/ir/` | Python IR utilities: `IRBuilder`, `PassManager`, `compile()`, op wrappers |
| `pypto/ir/op/` | Low-level IR ops (return raw `Expr`): `tensor_ops`, `block_ops`, `ptr_ops`, `system_ops` |
| `pypto/language/` | User-facing DSL: decorators, type annotations, control flow |
| `pypto/language/parser/` | `ASTParser`: Python AST → PyPTO IR conversion |
| `pypto/language/typing/` | DSL type wrappers: `Tensor`, `Tile`, `Scalar`, `Ptr` |
| `pypto/language/op/` | High-level ops (return `Tensor`/`Tile`/`Scalar`): type-safe user-facing API |
| `pypto/language/manual/op/` | Manual (non-SSA) operation definitions |
| `pypto/frontend/` | `@kernel`/`@jit` decorators, `compile()`, `launch()` |
| `pypto/backend/` | Python re-exports of backend bindings |
| `pypto/runtime/` | Kernel execution: `run()`, `RunConfig`, `TensorSpec` |

---

## Core Data Structures

### Type System

```
Type (abstract)
├── UnknownType
├── ScalarType          — dtype: DataType (FP16, FP32, INT32, ...)
├── TensorType          — shape: Expr[], dtype, layout (ND/DN/NZ), stride
├── TileType            — shape: int[], dtype, block_layout, scalar_layout
├── MemRefType          — shaped_type, memory_space (DDR/Vec/Mat/Left/Right/Acc)
├── PtrType             — base_type, base_ptr, offset (for workspace indirect-select)
└── TupleType           — element_types: Type[]
```

### Expression Hierarchy

```
Expr (abstract, immutable)
├── Var                 — named variable
├── MemRef              — memory allocation (auto-named mem_<id>)
├── GlobalVar           — function reference
├── ConstInt/Float/Bool — literal constants
├── Call                — op invocation: op + args + kwargs
├── MakeTuple           — tuple construction
├── TupleGetItemExpr    — tuple element access
├── Binary ops          — Add, Sub, Mul, FloorDiv, Min, Max, Eq, Lt, And, Or, ...
└── Unary ops           — Abs, Neg, Not, Cast, BitNot
```

### Statement Hierarchy

```
Stmt (abstract, immutable)
├── AssignStmt          — target = value
├── ForStmt             — var, start, end, body, kind (Sequential/Parallel/Unroll)
├── WhileStmt           — condition, body
├── IfStmt              — condition, then_body, else_body
├── ScopeStmt           — body, kind (InCore)
├── SectionStmt         — body, kind (Vector=0, Cube=1)
├── SeqStmts            — ordered statement list
├── OpStmts             — flattened op invocations
├── EvalStmt            — evaluate expr for side effects
├── ReturnStmt          — return value
├── YieldStmt           — yield from loop
├── BreakStmt           — loop break
└── ContinueStmt        — loop continue
```

### Program Structure

```
Program
├── functions: Map<GlobalVar, Function>
│
Function
├── name, params: Expr[], body: Stmt
├── return_type: Type
├── function_type: Opaque | Orchestration | InCore | Helper
└── param_directions: In | Out | InOut
```

---

## Key Data Flows

### 1. User Code → IR (Parsing)

```
@pl.function                        Python decorator
    ↓
ASTParser.parse()                   language/parser/ast_parser.py
    ├── TypeResolver                Resolve type annotations → IR types
    ├── ExprEvaluator               Evaluate compile-time expressions
    ├── ScopeManager                Track variable scopes
    └── SpanTracker                 Capture source locations
    ↓
ir.Program                          Immutable IR AST
```

### 2. IR → Optimized IR (Pass Pipeline)

Default pass execution order (configured in `python/pypto/ir/pass_manager.py`):

```
RunVerifier           → Type check + structural validation
InsertSync            → Add pipeline synchronization ops (implemented in insert_sync_pass.cpp)
```

> **Note**: Pass factory functions are declared in `include/pypto/ir/transforms/passes.h`.
> Currently only `InsertSync` has a C++ implementation in `src/ir/transforms/`.
> Other passes are registered via the header but their implementations are
> pending or handled elsewhere in the compilation pipeline.

### 3. Optimized IR → Target Code (Codegen)

```
PTOCodegen.Generate(program)
    ├── Visit Program → functions
    ├── Visit Function → MLIR func.func
    ├── Visit Stmts → MLIR control flow (scf.for, scf.if)
    ├── Visit Call → lookup backend op codegen → emit MLIR op
    │   └── Backend910B_PTO registered ops (REGISTER_BACKEND_OP)
    ├── Visit MemRef → alloc_tile, make_tensor_view
    └── Assemble → complete MLIR module string
         ↓
    ptoas tool → C++ source
         ↓
    bisheng compiler → .so shared library
```

### 4. Runtime Execution

```
compile(program)  →  CompiledKernel(lib_path, param_specs)
launch(kernel, *args)
    ├── Validate args against param_specs
    ├── Convert to ctypes
    └── dlopen(.so) → call_kernel()  →  NPU execution
```

---

## Operator System

Three operator categories with increasing hardware specificity:

| Category | Directory | Purpose | Example |
|----------|-----------|---------|---------|
| **Tensor ops** | `ir/op/tensor_ops/` | High-level N-D tensor operations | `tensor.add`, `tensor.matmul` |
| **Block ops** | `ir/op/block_ops/` | Hardware tile operations with memory | `block.load`, `block.add`, `block.matmul` |
| **Manual ops** | `ir/op/manual_ops/` | Non-SSA operations for manual control | `manual.elementwise`, `manual.matmul` |

**Registration**: Each op is registered via `REGISTER_OP("name")` macro with:
- Argument schema and validation
- Type deduction function (`f_deduce_type`)
- Pipeline type (`PipeType`: MTE1, MTE2, M, V, FIX)
- Backend codegen function (`REGISTER_BACKEND_OP`)

**Two-layer Python API**:
- `pypto.ir.op.*` — low-level, returns raw `Expr`
- `pypto.language.op.*` — high-level, returns `Tensor`/`Tile`/`Scalar` wrappers

---

## Key Design Patterns

### Immutability + Copy-on-Write

All IR nodes are immutable. Transformations use `IRMutator` which returns the original node if children are unchanged, or creates a new node with modified children.

### Visitor/Mutator Pattern

- `IRVisitor`: Read-only traversal, recursive default implementations
- `IRMutator`: Copy-on-write transformations via `VisitExpr_`/`VisitStmt_` overrides
- Both use `ExprFunctor<R>`/`StmtFunctor<R>` CRTP double dispatch

### Reflection System

Every IR node provides `GetFieldDescriptors()` returning typed field descriptors:
- `IgnoreField()` — skip in traversal/serialization
- `UsualField()` — include in traversal
- `DefField()` — definition (special handling)

Enables generic: serialization, structural comparison, hashing, pretty printing.

### Property-Based Pass Verification

- `IRProperty` enum: SSAForm, TypeChecked, NoNestedCalls, MemoryAllocated, ...
- `IRPropertySet`: Bitset for O(1) property tracking
- Each pass declares `required/produced/invalidated` properties
- `PassContext` + `VerificationInstrument` auto-verify properties

### Registry Pattern

- `OpRegistry`: Global operator registration with fluent API
- `BackendRegistry`: Backend singleton management
- `PropertyVerifierRegistry`: Verifier registration
- All use static initialization via registration macros

---

## Cross-Layer Synchronization

Three layers must stay synchronized for any API change:

| Layer | Location | Example |
|-------|----------|---------|
| C++ header + impl | `include/pypto/`, `src/` | `bool IsScalar() const;` |
| Python binding | `python/bindings/modules/` | `.def("is_scalar", &TensorExpr::IsScalar)` |
| Type stub | `python/pypto/pypto_core/*.pyi` | `def is_scalar(self) -> bool: ...` |

Naming convention: C++ `GetValue()` → Python `get_value()` (snake_case).

---

## Hardware Model

```
SoC
└── Die
    └── Cluster
        ├── Cube Core (matrix computation, PIPE_M)
        └── Vector Core ×2 (vector/scalar ops, PIPE_V)
            ├── Vec memory (192KB a2/a3, 248KB a5)
            ├── Mat memory (512KB)
            ├── Left/Right memory (64KB each)
            └── Acc memory (128KB a2/a3, 256KB a5)
```

Memory spaces: DDR → Vec/Mat/Left/Right/Acc (on-chip)

Pipeline types for synchronization:
- `MTE2` (TLOAD), `MTE1` (TMOV_M2L, TMOV_M2B), `PIPE_FIX` (TSTORE_ACC, TMOV_M2S, TMOV_V2M)
- `PIPE_M` (TMATMUL), `PIPE_V` (TVEC, TVECWAIT_EVENT)

---

## Build & Test

```bash
# Full build
export HOME=/data/g00895580
source compile.sh

# Run basic test
python3 tests/ut/frontend/test_dynamic_matmul_db.py
```

Test structure: `tests/ut/{core,ir,frontend,pass}/`
