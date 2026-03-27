/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * @file backend_910b_cce_ops.cpp
 * @brief Backend op registration for Backend910B_CCE
 *
 * This file registers all block operations for the CCE backend.
 * Each registration specifies the pipe type and CCE codegen function.
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>

#include "pypto/backend/910B_CCE/backend_910b_cce.h"
#include "pypto/backend/common/backend.h"
#include "pypto/codegen/cce/cce_codegen.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

// ============================================================================
// Helper Functions for CCE Code Generation
// ============================================================================

/**
 * @brief Compute stride-based offset for multi-dimensional tensor access
 * @param codegen The CCE codegen instance
 * @param tensor_var_name The tensor variable name (e.g., "inputGlobal")
 * @param offset_exprs Vector of offset expressions for each dimension
 * @param tensor_type The tensor type for shape information
 * @return C++ expression string for total offset computation
 */
static std::string ComputeStrideBasedOffset(codegen::CCECodegen& codegen, const std::string& tensor_var_name,
                                            const ir::MakeTuplePtr& offsets,
                                            const ir::TensorTypePtr& tensor_type) {
  // In single-file mode: compute strides from IR tensor shape (no Tensor struct)
  if (codegen.IsSingleFileMode()) {
    return codegen.ComputeIRBasedOffset(tensor_type, offsets);
  }

  // Get Tensor struct pointer for stride access
  std::string tensor_struct = codegen.GetTensorStruct(tensor_var_name);

  // Build offset computation: offset[0] * stride[0] + offset[1] * stride[1] + ...
  std::ostringstream offset_computation;
  offset_computation << "(" << tensor_struct << "->start_offset";

  for (size_t i = 0; i < offsets->elements_.size(); ++i) {
    offset_computation << " + " << codegen.GetExprAsCode(offsets->elements_[i]) << " * " << tensor_struct
                       << "->strides[" << i << "]";
  }

  offset_computation << ")";
  return offset_computation.str();
}

// Helper function for binary elementwise operations
static std::string MakeBinaryElementwiseCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                                   codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "Binary elementwise op requires 2 arguments";
  std::string lhs = codegen.GetExprAsCode(op->args_[0]);
  std::string rhs = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetCurrentResultTarget();
  codegen.Emit(cce_op_name + "(" + dst + ", " + lhs + ", " + rhs + ");");
  return "";
}

// Helper function for binary scalar operations
static std::string MakeBinaryScalarCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                              codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "Binary scalar op requires 2 arguments";
  std::string lhs = codegen.GetExprAsCode(op->args_[0]);
  std::string rhs = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetCurrentResultTarget();
  codegen.Emit(cce_op_name + "(" + dst + ", " + lhs + ", " + rhs + ");");
  return "";
}

// Helper function for unary operations
static std::string MakeUnaryCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                       codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "Unary op requires 1 argument";
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();
  codegen.Emit(cce_op_name + "(" + dst + ", " + src + ");");
  return "";
}

// Helper for block.cast - extract target_dtype from kwargs and use TCVT
static std::string MakeBlockCastCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "block.cast requires 1 argument";
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();
  int mode = op->GetKwarg<int>("mode");
  // TCVT signature: TCVT(dst, src, rmode)
  // Using default rounding mode (0 for round-to-nearest-even)
  codegen.Emit("TCVT(" + dst + ", " + src + ", " + codegen.GetTypeConverter().ConvertCastRoundMode(mode) +
               ");");
  return "";
}

// Helper for block.cmp/cmps - extract cmp_type from kwargs and use TCMP
static std::string MakeBlockCmpCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                          codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "block.cmp requires 2 arguments";
  std::string lhs = codegen.GetExprAsCode(op->args_[0]);
  std::string rhs = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetCurrentResultTarget();
  int cmp_type = op->GetKwarg<int>("cmp_type");
  // signature: TCMP/TCMPS(dst, src0, src1, cmpMode)
  // cmpMode: EQ=0, NE=1, LT=2, LE=3, GT=4, GE=5
  codegen.Emit(cce_op_name + "(" + dst + ", " + lhs + ", " + rhs + ", " + std::to_string(cmp_type) + ");");
  return "";
}

// Helper for block.expands/col_expand - expand scalar/col tile to tile
static std::string MakeBlockExpandsCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                              codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "block.expands/col_expand requires 2 arguments";

  std::string src1 = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetCurrentResultTarget();
  // FIX: this instruction is inplaced, dst and target addr should be same
  codegen.Emit(cce_op_name + "(" + dst + ", " + src1 + ");");
  return "";
}

// block.load: emit TASSIGN + TLOAD (same format as original IR layer codegen)
// IR signature: (tensor, offsets_tuple, shapes_tuple) = 3 args
static std::string MakeBlockLoadCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 4) << "block.load requires 4 arguments: tensor, offsets, shapes, validshape";

  auto src_tensor_var_ptr = std::dynamic_pointer_cast<const ir::Var>(op->args_[0]);
  CHECK(src_tensor_var_ptr != nullptr) << "block.load source tensor must be a Var";

  // Extract offsets tuple
  auto offsets_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[1]);
  CHECK(offsets_tuple != nullptr) << "block.load second argument must be a tuple (offsets)";
  CHECK(!offsets_tuple->elements_.empty()) << "block.load offsets tuple must have at least 1 element";

  // Extract shapes tuple
  auto shapes_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[2]);
  CHECK(shapes_tuple != nullptr) << "block.load third argument must be a tuple (shapes)";

  std::string src_tensor_var = codegen.GetVarName(src_tensor_var_ptr);

  auto src_tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(src_tensor_var_ptr->GetType());
  CHECK(src_tensor_type != nullptr) << "block.load source must be TensorType";

  // compute stride-based offset
  std::string offset = ComputeStrideBasedOffset(codegen, src_tensor_var, offsets_tuple, src_tensor_type);

  // Get buffer address from Tensor struct
  std::string src_ptr = codegen.GetPointer(src_tensor_var);
  std::string var_name = codegen.GetCurrentResultTarget();

  codegen.Emit("TASSIGN(" + src_tensor_var + ", " + src_ptr + " + " + offset + ");");
  codegen.Emit("TLOAD(" + var_name + ", " + src_tensor_var + ");");
  return "";
}

// block.store: emit TASSIGN + TSTORE + RegisterOutputPointer (same format as original IR layer codegen)
// IR signature: (tile, offsets_tuple, shapes_tuple, output_tensor) = 4 args
static std::string MakeBlockStoreCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 4) << "block.store requires 4 arguments: tile, offsets, shapes, output_tensor";

  std::string src_tile = codegen.GetExprAsCode(op->args_[0]);

  // Extract offsets tuple
  auto offsets_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[1]);
  CHECK(offsets_tuple != nullptr) << "block.store second argument must be a tuple (offsets)";
  CHECK(!offsets_tuple->elements_.empty()) << "block.store offsets tuple must have at least 1 element";

  // Extract shapes tuple
  auto shapes_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[2]);
  CHECK(shapes_tuple != nullptr) << "block.store third argument must be a tuple (shapes)";

  auto dst_tensor_var_ptr = std::dynamic_pointer_cast<const ir::Var>(op->args_[3]);
  CHECK(dst_tensor_var_ptr != nullptr) << "block.store destination tensor must be a Var";

  std::string dst_tensor_var = codegen.GetVarName(dst_tensor_var_ptr);

  auto dst_tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(dst_tensor_var_ptr->GetType());
  CHECK(dst_tensor_type != nullptr) << "block.store destination must be TensorType";

  // compute stride-based offset
  std::string offset = ComputeStrideBasedOffset(codegen, dst_tensor_var, offsets_tuple, dst_tensor_type);

  // Get buffer address from Tensor struct
  std::string dst_ptr = codegen.GetPointer(dst_tensor_var);
  std::string var_name = codegen.GetCurrentResultTarget();

  codegen.Emit("TASSIGN(" + dst_tensor_var + ", " + dst_ptr + " + " + offset + ");");
  codegen.Emit("TSTORE(" + dst_tensor_var + ", " + src_tile + ");");
  if (!var_name.empty()) {
    codegen.RegisterOutputPointer(var_name, dst_tensor_var);
    codegen.RegisterOutputTensorStruct(var_name, dst_tensor_var);
    codegen.Emit("auto " + var_name + " = " + dst_tensor_var + ";");
  }
  return "";
}

// Helper function for block.l0c_store
// IR signature: (tile, offsets_tuple, shapes_tuple, output_tensor) = 4 args
static std::string MakeBlockL0CStoreCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 4)
      << "block.l0c_store requires 4 arguments: tile, offsets, shapes, output_tensor";

  std::string src_tile = codegen.GetExprAsCode(op->args_[0]);

  // Extract offsets tuple
  auto offsets_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[1]);
  CHECK(offsets_tuple != nullptr) << "block.l0c_store second argument must be a tuple (offsets)";
  CHECK(!offsets_tuple->elements_.empty()) << "block.l0c_store offsets tuple must have at least 1 element";

  // Extract shapes tuple
  auto shapes_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[2]);
  CHECK(shapes_tuple != nullptr) << "block.l0c_store third argument must be a tuple (shapes)";

  auto dst_tensor_var_ptr = std::dynamic_pointer_cast<const ir::Var>(op->args_[3]);
  CHECK(dst_tensor_var_ptr != nullptr) << "block.l0c_store destination tensor must be a Var";

  std::string dst_tensor_var = codegen.GetVarName(dst_tensor_var_ptr);

  auto dst_tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(dst_tensor_var_ptr->GetType());
  CHECK(dst_tensor_type != nullptr) << "block.l0c_store destination must be TensorType";

  // compute stride-based offset
  std::string offset = ComputeStrideBasedOffset(codegen, dst_tensor_var, offsets_tuple, dst_tensor_type);

  // Get buffer address from Tensor struct
  std::string dst_ptr = codegen.GetPointer(dst_tensor_var);
  std::string var_name = codegen.GetCurrentResultTarget();

  codegen.Emit("TASSIGN(" + dst_tensor_var + ", " + dst_ptr + " + " + offset + ");");
  codegen.Emit("TSTORE(" + dst_tensor_var + ", " + src_tile + ");");
  if (!var_name.empty()) {
    codegen.RegisterOutputPointer(var_name, dst_tensor_var);
    codegen.RegisterOutputTensorStruct(var_name, dst_tensor_var);
    codegen.Emit("auto " + var_name + " = " + dst_tensor_var + ";");
  }
  return "";
}

// Helper function for block.move
static std::string MakeBlockMoveCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "block.move requires 1 argument: src";

  // Validate memory locations: can't Vec→Vec copies
  auto src_type = ir::As<ir::TileType>(op->args_[0]->GetType());
  INTERNAL_CHECK(src_type != nullptr) << "Internal error: block.move source must be TileType";
  INTERNAL_CHECK(src_type->memref_.has_value())
      << "Internal error: block.move source TileType must have MemRef (InitMemRef pass should have run)";

  ir::MemorySpace target_memory = op->GetKwarg<ir::MemorySpace>("target_memory");
  ir::MemorySpace src_mem =
      (*src_type->memref_)->memory_space_;  // NOLINT(bugprone-unchecked-optional-access)
  CHECK(!(src_mem == ir::MemorySpace::Vec && target_memory == ir::MemorySpace::Vec))
      << "block.move: Vec to Vec move should use block.vec_move";

  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();

  codegen.Emit("TMOV(" + dst + ", " + src + ");");

  return "";
}

// Helper function for block.vec_move (Vec to Vec copy only)
static std::string MakeBlockUbCopyCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "block.vec_move requires 1 argument: src";

  // Validate memory locations: ONLY support Vec→Vec copies
  auto src_type = ir::As<ir::TileType>(op->args_[0]->GetType());
  INTERNAL_CHECK(src_type != nullptr) << "Internal error: block.vec_move source must be TileType";
  INTERNAL_CHECK(src_type->memref_.has_value())
      << "Internal error: block.vec_move source TileType must have MemRef (InitMemRef pass should have run)";

  // Verify source is on Vec
  ir::MemorySpace src_mem =
      (*src_type->memref_)->memory_space_;  // NOLINT(bugprone-unchecked-optional-access)
  CHECK(src_mem == ir::MemorySpace::Vec)
      << "block.vec_move: source must be on Vec memory, got " << ir::MemorySpaceToString(src_mem);

  // Get source and destination expressions
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();

  // Emit TMOV instruction for Vec→Vec copy
  codegen.Emit("TMOV(" + dst + ", " + src + ");");

  return "";
}

// Helper function for block.alloc (no-op: allocation handled elsewhere)
static std::string MakeBlockAllocCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  (void)op;
  (void)codegen_base;
  return "";  // No C++ emission - MemRef/Tile setup handled in prologue
}

// Helper function for block.get_block_idx (returns value expression)
static std::string MakeBlockGetBlockIdxCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  CHECK(op->args_.size() == 0) << "block.get_block_idx requires no arguments";
  return "get_block_idx()";
}

// Helper function for block.make_tile (no-op: allocation handled elsewhere)
static std::string MakeBlockCreateTileCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  (void)op;
  (void)codegen_base;
  return "";  // No C++ emission - Tile declaration handled in prologue
}

// Helper function for block.full
static std::string MakeBlockFullCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  std::string dst = codegen.GetCurrentResultTarget();
  std::string scalar = codegen.GetExprAsCode(op->args_[1]);
  codegen.Emit("TEXPANDS(" + dst + ", " + scalar + ");");
  return "";
}

// ============================================================================
// Matmul Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "block.matmul")
    .set_pipe(ir::PipeType::M)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) -> std::string {
      CHECK(op->args_.size() == 2) << "block.matmul requires 2 arguments: lhs, rhs";

      std::string lhs = codegen.GetExprAsCode(op->args_[0]);
      std::string rhs = codegen.GetExprAsCode(op->args_[1]);
      std::string dst = codegen.GetCurrentResultTarget();

      codegen.Emit("TMATMUL(" + dst + ", " + lhs + ", " + rhs + ");");

      return "";  // Statement-emitting mode
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.matmul_acc")
    .set_pipe(ir::PipeType::M)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) -> std::string {
      CHECK(op->args_.size() == 3) << "block.matmul_acc requires 3 arguments: acc, lhs, rhs";

      [[maybe_unused]] std::string acc = codegen.GetExprAsCode(op->args_[0]);
      std::string lhs = codegen.GetExprAsCode(op->args_[1]);
      std::string rhs = codegen.GetExprAsCode(op->args_[2]);
      std::string dst = codegen.GetCurrentResultTarget();

      // TMATMUL_ACC accumulates into dst, which should be initialized from acc
      // In CCE ISA, this is typically: TMATMUL_ACC(dst, acc, lhs, rhs)
      codegen.Emit("TMATMUL_ACC(" + dst + ", " + acc + ", " + lhs + ", " + rhs + ");");

      return "";  // Statement-emitting mode
    });

// ============================================================================
// Elementwise Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "block.mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TMUL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TADD", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TDIV", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TSUB", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.maximum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TMAX", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.minimum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TMIN", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.muls")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryScalarCodegenCCE("TMULS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.adds")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryScalarCodegenCCE("TADDS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.divs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryScalarCodegenCCE("TDIVS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.subs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryScalarCodegenCCE("TSUBS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.cmp")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockCmpCodegenCCE("TCMP", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.cmps")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockCmpCodegenCCE("TCMPS", op, codegen);
    });

// ============================================================================
// Unary Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "block.exp")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TEXP", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.neg")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TNEG", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.recip")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TRECIP", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.rsqrt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TRSQRT", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.sqrt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TSQRT", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.log")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TLOG", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.abs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TABS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.relu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TRELU", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockCastCodegenCCE(op, codegen);
    });

// ============================================================================
// Memory Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "block.alloc")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockAllocCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.make_tile")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockCreateTileCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.load")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockLoadCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.store")
    .set_pipe(ir::PipeType::MTE3)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockStoreCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.l0c_store")
    .set_pipe(ir::PipeType::FIX)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockL0CStoreCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.move")
    .set_pipe(ir::PipeType::MTE1)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockMoveCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.vec_move")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockUbCopyCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.get_block_idx")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockGetBlockIdxCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.full")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockFullCodegenCCE(op, codegen);
    });

// ============================================================================
// Reduction Operations
// ============================================================================

static std::string MakeBlockRowReductionCodegenCCE(const std::string& op_prefix, const ir::CallPtr& op,
                                                   codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "TROW" << op_prefix << " requires 2 arguments";
  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string tmp_tile = codegen.GetExprAsCode(op->args_[1]);
  std::string result = codegen.GetCurrentResultTarget();

  codegen.Emit("TROW" + op_prefix + "(" + result + ", " + tile + ", " + tmp_tile + ");");
  return "";
}

static std::string MakeBlockColReductionCodegenCCE(const std::string& op_prefix, const ir::CallPtr& op,
                                                   codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "TCOL" << op_prefix << " requires 1 argument";
  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string result = codegen.GetCurrentResultTarget();

  codegen.Emit("TCOL" + op_prefix + "(" + result + ", " + tile + ");");
  return "";
}

// Helper function for reduction operations (sum, max)
static std::string MakeBlockReductionCodegenCCE(const std::string& op_prefix, const ir::CallPtr& op,
                                                codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  int axis = op->GetKwarg<int>("axis");
  if (axis == 0) {
    return MakeBlockColReductionCodegenCCE(op_prefix, op, codegen_base);
  } else {
    return MakeBlockRowReductionCodegenCCE(op_prefix, op, codegen_base);
  }
}

REGISTER_BACKEND_OP(Backend910B_CCE, "block.sum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockReductionCodegenCCE("SUM", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.max")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockReductionCodegenCCE("MAX", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_sum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockRowReductionCodegenCCE("SUM", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_max")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockRowReductionCodegenCCE("MAX", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.min")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockReductionCodegenCCE("MIN", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_min")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockRowReductionCodegenCCE("MIN", op, codegen);
    });

// ============================================================================
// Broadcast Operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_expand_mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TROWEXPANDMUL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_expand_div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TROWEXPANDDIV", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_expand_sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TROWEXPANDSUB", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.row_expand_add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TROWEXPANDADD", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.fillpad")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeUnaryCodegenCCE("TFILLPAD", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.col_expand")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockExpandsCodegenCCE("TCOLEXPAND", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.col_expand_add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TCOLEXPANDADD", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.col_expand_mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TCOLEXPANDMUL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.col_expand_div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TCOLEXPANDDIV", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.col_expand_sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBinaryElementwiseCodegenCCE("TCOLEXPANDSUB", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.expands")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockExpandsCodegenCCE("TEXPANDS", op, codegen);
    });

// ============================================================================
// Transform Operations (view/reshape/transpose: same buffer, reinterpret)
// ============================================================================

static std::string MakeBlockTransformCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() >= 1) << "block view/reshape/transpose require at least 1 argument";
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetCurrentResultTarget();
  codegen.Emit("TMOV(" + dst + ", " + src + ");");
  return "";
}

static std::string MakeTileReshapeCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  std::string target_var = codegen.GetCurrentResultTarget();
  std::string input_var = codegen.GetExprAsCode(op->args_[0]);

  codegen.Emit("TRESHAPE(" + target_var + ", " + input_var + ");");
  return "";
}

static std::string MakeTileTransposeCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  std::string target_var = codegen.GetCurrentResultTarget();
  std::string input_var = codegen.GetExprAsCode(op->args_[0]);
  auto axis1 = codegen.GetConstIntValue(op->args_[1]);
  auto axis2 = codegen.GetConstIntValue(op->args_[2]);
  size_t ndim = ir::As<ir::TileType>(op->args_[0]->GetType())->shape_.size();

  INTERNAL_CHECK(ndim == 2) << "Codegen only supports 2D tiles, but got " << ndim << "D tile";
  INTERNAL_CHECK(axis1 != axis2) << "tile.transpose: axis1 and axis2 must be different, but got axis1=axis2="
                                 << axis1;
  INTERNAL_CHECK(axis1 >= 0 && axis1 < ndim && axis2 >= 0 && axis2 < ndim)
      << "tile.transpose: axis1 and axis2 must be in range [0, " << ndim << "), but got axis1=" << axis1
      << ", axis2=" << axis2;

  codegen.Emit("TTRANS(" + target_var + ", " + input_var + ");");
  return "";
}

REGISTER_BACKEND_OP(Backend910B_CCE, "block.reshape")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileReshapeCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.transpose")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileReshapeCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.view")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockTransformCodegenCCE(op, codegen);
    });

// ============================================================================
// Sync / Barrier Operations (inserted by insert_sync_pass)
// ============================================================================

static std::string PipeTypeToCCEString(ir::PipeType pipe) {
  switch (pipe) {
    case ir::PipeType::MTE1:
      return "PIPE_MTE1";
    case ir::PipeType::MTE2:
      return "PIPE_MTE2";
    case ir::PipeType::MTE3:
      return "PIPE_MTE3";
    case ir::PipeType::M:
      return "PIPE_M";
    case ir::PipeType::V:
      return "PIPE_V";
    case ir::PipeType::S:
      return "PIPE_S";
    case ir::PipeType::FIX:
      return "PIPE_FIX";
    case ir::PipeType::ALL:
      return "PIPE_ALL";
    default:
      return "PIPE_V";
  }
}

static std::string MakeSyncCodegenCCE(const std::string& isa_name, const ir::CallPtr& op,
                                      codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  auto set_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("set_pipe"));
  auto wait_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("wait_pipe"));
  int event_id = op->GetKwarg<int>("event_id");
  std::string set_pipe_str = PipeTypeToCCEString(set_pipe);
  std::string wait_pipe_str = PipeTypeToCCEString(wait_pipe);
  std::string event_id_str = "EVENT_ID" + std::to_string(event_id);
  codegen.Emit(isa_name + "(" + set_pipe_str + ", " + wait_pipe_str + ", " + event_id_str + ");");
  return "";
}

REGISTER_BACKEND_OP(Backend910B_CCE, "system.sync_src")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeSyncCodegenCCE("set_flag", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.sync_dst")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeSyncCodegenCCE("wait_flag", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.bar_v")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_V);");
      return "";
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.bar_m")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_M);");
      return "";
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.bar_all")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      dynamic_cast<codegen::CCECodegen&>(codegen_base).Emit("pipe_barrier(PIPE_ALL);");
      return "";
    });

// ============================================================================
// Cross-core Sync Operations
// ============================================================================

// Cross-core SET: ffts_cross_core_sync(PIPE_xxx, getFFTSMsg(FFTS_MODE_VAL, event_id))
// Cross-core WAIT: wait_flag_dev(event_id)
// SET signals completion from a pipe; WAIT blocks until the other core signals.

static std::string MakeCrossCoreSetCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                               bool is_dynamic) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  auto pipe = op->GetKwarg<int>("pipe");
  std::string pipe_str = PipeTypeToCCEString(static_cast<ir::PipeType>(pipe));
  bool is_a5 = (codegen.GetArch() == "a5");
  if (is_a5) {
    // a5: use set_intra_block (called twice: id and id+16)
    if (is_dynamic) {
      std::string event_id = codegen.GetExprAsCode(op->args_[0]);
      codegen.Emit("set_intra_block(" + pipe_str + ", " + event_id + ");");
      codegen.Emit("set_intra_block(" + pipe_str + ", " + event_id + " + 16);");
    } else {
      int event_id = op->GetKwarg<int>("event_id");
      codegen.Emit("set_intra_block(" + pipe_str + ", " + std::to_string(event_id) + ");");
      codegen.Emit("set_intra_block(" + pipe_str + ", " + std::to_string(event_id + 16) + ");");
    }
  } else {
    if (is_dynamic) {
      std::string event_id = codegen.GetExprAsCode(op->args_[0]);
      codegen.Emit("ffts_cross_core_sync(" + pipe_str + ", getFFTSMsg(FFTS_MODE_VAL, " + event_id + "));");
    } else {
      int event_id = op->GetKwarg<int>("event_id");
      codegen.Emit("ffts_cross_core_sync(" + pipe_str + ", getFFTSMsg(FFTS_MODE_VAL, " +
                    std::to_string(event_id) + "));");
    }
  }
  return "";
}

static std::string MakeCrossCoreWaitCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base,
                                                bool is_dynamic) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  auto pipe = op->GetKwarg<int>("pipe");
  std::string pipe_str = PipeTypeToCCEString(static_cast<ir::PipeType>(pipe));
  bool is_a5 = (codegen.GetArch() == "a5");
  if (is_a5) {
    // a5: use wait_intra_block
    if (is_dynamic) {
      std::string event_id = codegen.GetExprAsCode(op->args_[0]);
      codegen.Emit("wait_intra_block(" + pipe_str + ", " + event_id + ");");
    } else {
      int event_id = op->GetKwarg<int>("event_id");
      codegen.Emit("wait_intra_block(" + pipe_str + ", " + std::to_string(event_id) + ");");
    }
  } else {
    if (is_dynamic) {
      std::string event_id = codegen.GetExprAsCode(op->args_[0]);
      codegen.Emit("wait_flag_dev(" + event_id + ");");
    } else {
      int event_id = op->GetKwarg<int>("event_id");
      codegen.Emit("wait_flag_dev(" + std::to_string(event_id) + ");");
    }
  }
  return "";
}

REGISTER_BACKEND_OP(Backend910B_CCE, "system.set_cross_core")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeCrossCoreSetCodegenCCE(op, codegen, false);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.wait_cross_core")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeCrossCoreWaitCodegenCCE(op, codegen, false);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.set_cross_core_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeCrossCoreSetCodegenCCE(op, codegen, true);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "system.wait_cross_core_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeCrossCoreWaitCodegenCCE(op, codegen, true);
    });

// ============================================================================
// Dynamic event_id sync operations
// ============================================================================

static std::string MakeSyncSrcDynCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  auto set_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("set_pipe"));
  auto wait_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("wait_pipe"));
  std::string event_id = codegen.GetExprAsCode(op->args_[0]);
  std::string set_pipe_str = PipeTypeToCCEString(set_pipe);
  std::string wait_pipe_str = PipeTypeToCCEString(wait_pipe);
  // EventId array access (contains '[') already returns event_t — no cast needed
  if (event_id.find('[') != std::string::npos) {
    codegen.Emit("set_flag(" + set_pipe_str + ", " + wait_pipe_str + ", " + event_id + ");");
  } else {
    codegen.Emit("set_flag(" + set_pipe_str + ", " + wait_pipe_str + ", (event_t)" + event_id + ");");
  }
  return "";
}

REGISTER_BACKEND_OP(Backend910B_CCE, "system.sync_src_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeSyncSrcDynCodegenCCE(op, codegen);
    });

static std::string MakeSyncDstDynCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  auto set_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("set_pipe"));
  auto wait_pipe = static_cast<ir::PipeType>(op->GetKwarg<int>("wait_pipe"));
  std::string event_id = codegen.GetExprAsCode(op->args_[0]);
  std::string set_pipe_str = PipeTypeToCCEString(set_pipe);
  std::string wait_pipe_str = PipeTypeToCCEString(wait_pipe);
  // EventId array access (contains '[') already returns event_t — no cast needed
  if (event_id.find('[') != std::string::npos) {
    codegen.Emit("wait_flag(" + set_pipe_str + ", " + wait_pipe_str + ", " + event_id + ");");
  } else {
    codegen.Emit("wait_flag(" + set_pipe_str + ", " + wait_pipe_str + ", (event_t)" + event_id + ");");
  }
  return "";
}

REGISTER_BACKEND_OP(Backend910B_CCE, "system.sync_dst_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeSyncDstDynCodegenCCE(op, codegen);
    });

// ============================================================================
// Missing block operations: get_block_num, get_subblock_idx, index_cast
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "block.get_block_num")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      CHECK(op->args_.size() == 0) << "block.get_block_num requires no arguments";
      return std::string("get_block_num()");
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.get_subblock_idx")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      CHECK(op->args_.size() == 0) << "block.get_subblock_idx requires no arguments";
      return std::string("get_subblockid()");
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "block.index_cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
      CHECK(op->args_.size() == 1) << "block.index_cast requires 1 argument";
      std::string value = codegen.GetExprAsCode(op->args_[0]);
      return "(int32_t)(" + value + ")";
    });

static std::string MakeTensorDimCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  std::string target_var = codegen.GetCurrentResultTarget();
  int64_t axis = codegen.GetConstIntValue(op->args_[1]);

  auto input_tensor = ir::As<ir::TensorType>(op->args_[0]->GetType());
  CHECK(input_tensor) << "tensor.dim need TensorType for first arg, but got "
                      << op->args_[0]->GetType()->TypeName();
  auto ndims = static_cast<int64_t>(input_tensor->shape_.size());
  int64_t pad_dims = 5 - ndims;  // pto-isa pad shape to 5 dims

  // get axis in GlobalTensor 5 dims
  if (axis < 0) {
    axis += ndims;
  }
  int64_t gt_dim = pad_dims + axis;

  // get GlobalTensor of input_tensor
  auto input_tensor_var = ir::As<ir::Var>(op->args_[0]);
  CHECK(input_tensor_var) << "tensor.dim need var with TensorType for first arg";
  std::string input_tensor_var_name = codegen.GetVarName(input_tensor_var);

  codegen.Emit("int " + target_var + " = " + input_tensor_var_name + ".GetShape(GlobalTensorDim::DIM_" +
               std::to_string(gt_dim) + ");");
  return "";
}

REGISTER_BACKEND_OP(Backend910B_CCE, "tensor.dim")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTensorDimCodegenCCE(op, codegen);
    });

}  // namespace backend
}  // namespace pypto
