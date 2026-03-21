# AI Assistant Rules for PyPTO Project

Please follow the following rules when working on the PyPTO project:
## Architecture
- Look at `docs/en/dev/ARCHITECTURE.md` to get a global picture of the project. For details look at docs
  in `docs/en/dev/ir`, `docs/en/dev/codegen`, `docs/en/dev/language`.
- This project is about to build a Python frontend for the MLIR of PTOAS. 
  The PTOAS can refer to `ptoas_ir.md` in the root folder. For details of 
  the MLIR of PTOAS, you can search in `/data/g00895580/test/PTOAS/include/PTO/IR`.
- We design a new pypto-ir above the MLIR of PTOAS. The project transforms 
  ops defined in `python/pypto/language/` to the MLIR of PTOAS via the frontend.
- We majorly use manual operations which are not SSA and all operands are in 
  the arguments. Manual ops should be put in `python/pypto/language/manual/op/`.
- `src/ir` - Source code of pypto-ir. Its Python bindings are located in 
  `python/bindings` and `python/pypto/ir`.
- `python/pypto/frontend` - The frontend kernel and JIT decorators, responsible 
  for triggering the transformation from pypto-ir to PTOAS MLIR.
- `src/codegen/pto/pto_codegen.cpp` and `src/backend/910B_PTO/` are designed 
  for lowering pypto-ir to PTOAS MLIR (emitted as string).
- The PTOAS MLIR string can be compiled by the `ptoas` tool, which will be 
  available in `PATH` after executing `source compile.sh`.

## Project Rules
## Build Commands
### Full configure, compile，build, install
```bash
export HOME=/data/g00895580
source compile.sh
```

## Test Commands
### Run a frontend testcase
- The following case is a basic testcase, you can test the basic function by this case. Precision of this case need to be checked Ok.
```bash
python3 tests/ut/frontend/test_dynamic_matmul_db.py
```
### mlir python path
/data/g00895580/mlir/llvm-project/build-shared/tools/mlir/python_packages/mlir_core/mlir

## Pipeline and sync
- TLOAD is PIPE_MTE2, TSTORE_ACC is PIPE_FIX, TMOV_M2L and TMOV_M2B are MTE1, TMOV_M2S and TMOV_V2M are PIPE_FIX, TMOV_M2V is PIPV, TMATMUL is PIPE_M, TVEC and TVECWAIT_EVENT is PIPE_V
- When a buffer is used within a loop, backward synchronization is needed: at the start of each iteration, execution must wait for all associated pipelines from the previous iteration to have completed before the buffer can be reused.
## Ascend Npu Harward Core information
- For AscendNPU, there are multiple cores, each processing a chunk of data. The MatMul operation is computed on Cube cores, while most other operations are computed on Vector cores. The ratio of Cube cores to Vector cores is 1:2.
- For Cube-Only or Vector-Only operation, use pto.get_block_idx() to get current Cube or vector core index, use pto.get_block_num() to get total living Cube number. Use pto.get_subblock_idx() to get current Vector sub block idx, which is 0 or 1. 
- For Mix(which contais both Cube and Vector) operation, use pto.get_block_idx() // 2 to get the corresponding Cube core index.

## Rules
- matmul kernel code need to begin with pto.section_cube():, and other vector kernel need to begin with pto.section_vector():.
- The frontend representation must offer strong ergonomics and high expressiveness.
- Temp files should be put in /data/g00895580/tmp
- Refer to "Pipeline and sync" in this doc to get information about how to add sync op.
- pto.VEC is 192KB on a2 and a3, and 248KB on a5; pto.MAT is 512KB on a2/a3/a5; pto.LEFT and pto.RIGHT are 64KB; pto.ACC is 128KB on a2 and a3 and 256KB on a5.