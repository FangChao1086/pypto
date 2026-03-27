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
 * @file backend_910b_cce_manual_ops.cpp
 * @brief CCE backend op registration for manual (non-SSA) operations.
 *
 * Manual ops carry an explicit pre-allocated output tile as their last argument.
 * Convention: args = [input0, input1, ..., output_tile]
 * The helpers here extract output from args.back() instead of GetCurrentResultTarget().
 */

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
// Helper: Manual binary op — args = [lhs, rhs, dst]
// Emits: OP(dst, lhs, rhs);
// ============================================================================
static std::string MakeManualBinaryCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                               codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << cce_op_name << ": expected 3 args (lhs, rhs, dst), got " << op->args_.size();
  std::string lhs = codegen.GetExprAsCode(op->args_[0]);
  std::string rhs = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetExprAsCode(op->args_[2]);
  codegen.Emit(cce_op_name + "(" + dst + ", " + lhs + ", " + rhs + ");");
  return "";
}

// ============================================================================
// Helper: Manual unary op — args = [src, dst]
// Emits: OP(dst, src);
// ============================================================================
static std::string MakeManualUnaryCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                              codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << cce_op_name << ": expected 2 args (src, dst), got " << op->args_.size();
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetExprAsCode(op->args_[1]);
  codegen.Emit(cce_op_name + "(" + dst + ", " + src + ");");
  return "";
}

// ============================================================================
// Helper: Manual scalar op — args = [tile, scalar, dst]
// Emits: OP(dst, tile, scalar);
// ============================================================================
static std::string MakeManualScalarCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                               codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << cce_op_name << ": expected 3 args (tile, scalar, dst), got "
                               << op->args_.size();
  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string scalar = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetExprAsCode(op->args_[2]);
  codegen.Emit(cce_op_name + "(" + dst + ", " + tile + ", " + scalar + ");");
  return "";
}

// ============================================================================
// Helper: Manual reduction op — args = [tile, tmp, dst]
// Emits: OP(dst, tile, tmp);
// ============================================================================
static std::string MakeManualRowReductionCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                                     codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << cce_op_name << ": expected 3 args (tile, tmp, dst), got "
                               << op->args_.size();
  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string tmp = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetExprAsCode(op->args_[2]);
  codegen.Emit(cce_op_name + "(" + dst + ", " + tile + ", " + tmp + ");");
  return "";
}

// ============================================================================
// Helper: Manual row-expand op — args = [tile, reduction_tile, dst]
// Emits: OP(dst, tile, reduction_tile);
// ============================================================================
static std::string MakeManualRowExpandCodegenCCE(const std::string& cce_op_name, const ir::CallPtr& op,
                                                  codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << cce_op_name << ": expected 3 args (tile, red, dst), got "
                               << op->args_.size();
  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string red = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetExprAsCode(op->args_[2]);
  codegen.Emit(cce_op_name + "(" + dst + ", " + tile + ", " + red + ");");
  return "";
}

// Helper function to compute stride-based offset
// In single-file mode: compute strides from IR tensor shape (no Tensor struct available)
// In normal mode: use Tensor struct strides (same as backend_910b_cce_ops.cpp)
static std::string ComputeManualOffset(codegen::CCECodegen& codegen, const std::string& tensor_var_name,
                                       const ir::MakeTuplePtr& offsets,
                                       const ir::TensorTypePtr& tensor_type) {
  if (codegen.IsSingleFileMode()) {
    return codegen.ComputeIRBasedOffset(tensor_type, offsets);
  }
  // Normal mode: use Tensor struct strides
  std::string tensor_struct = codegen.GetTensorStruct(tensor_var_name);
  std::ostringstream offset_computation;
  offset_computation << "(" << tensor_struct << "->start_offset";
  for (size_t i = 0; i < offsets->elements_.size(); ++i) {
    offset_computation << " + " << codegen.GetExprAsCode(offsets->elements_[i]) << " * " << tensor_struct
                       << "->strides[" << i << "]";
  }
  offset_computation << ")";
  return offset_computation.str();
}

// ============================================================================
// manual.load — args = [tensor, offsets, out_tile]
// Emits: TASSIGN(tensor_global, ptr + offset); TLOAD(out_tile, tensor_global);
// ============================================================================
static std::string MakeManualLoadCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "manual.load requires 3 arguments: tensor, offsets, out_tile";

  auto src_tensor_var_ptr = std::dynamic_pointer_cast<const ir::Var>(op->args_[0]);
  CHECK(src_tensor_var_ptr != nullptr) << "manual.load source tensor must be a Var";

  auto offsets_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[1]);
  CHECK(offsets_tuple != nullptr) << "manual.load second argument must be a tuple (offsets)";

  auto out_tile = std::dynamic_pointer_cast<const ir::Var>(op->args_[2]);
  CHECK(out_tile != nullptr) << "manual.load third argument (out) must be a Var";

  std::string src_tensor_var = codegen.GetVarName(src_tensor_var_ptr);
  auto src_tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(src_tensor_var_ptr->GetType());
  CHECK(src_tensor_type != nullptr) << "manual.load source must be TensorType";

  std::string offset = ComputeManualOffset(codegen, src_tensor_var, offsets_tuple, src_tensor_type);
  std::string src_ptr = codegen.GetPointer(src_tensor_var);
  std::string out_name = codegen.GetExprAsCode(op->args_[2]);

  codegen.Emit("TASSIGN(" + src_tensor_var + ", " + src_ptr + " + " + offset + ");");
  codegen.Emit("TLOAD(" + out_name + ", " + src_tensor_var + ");");
  return "";
}

// ============================================================================
// manual.store — args = [tile, offsets, output_tensor]
// Emits: TASSIGN(tensor_global, ptr + offset); TSTORE(tensor_global, tile);
// ============================================================================
static std::string MakeManualStoreCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "manual.store requires 3 arguments: tile, offsets, output_tensor";

  std::string src_tile = codegen.GetExprAsCode(op->args_[0]);

  auto offsets_tuple = std::dynamic_pointer_cast<const ir::MakeTuple>(op->args_[1]);
  CHECK(offsets_tuple != nullptr) << "manual.store second argument must be a tuple (offsets)";

  auto dst_tensor_var_ptr = std::dynamic_pointer_cast<const ir::Var>(op->args_[2]);
  CHECK(dst_tensor_var_ptr != nullptr) << "manual.store destination tensor must be a Var";

  std::string dst_tensor_var = codegen.GetVarName(dst_tensor_var_ptr);
  auto dst_tensor_type = std::dynamic_pointer_cast<const ir::TensorType>(dst_tensor_var_ptr->GetType());
  CHECK(dst_tensor_type != nullptr) << "manual.store destination must be TensorType";

  std::string offset = ComputeManualOffset(codegen, dst_tensor_var, offsets_tuple, dst_tensor_type);
  std::string dst_ptr = codegen.GetPointer(dst_tensor_var);

  codegen.Emit("TASSIGN(" + dst_tensor_var + ", " + dst_ptr + " + " + offset + ");");
  codegen.Emit("TSTORE(" + dst_tensor_var + ", " + src_tile + ");");
  return "";
}

// ============================================================================
// manual.l0c_store — args = [tile, offsets, output_tensor]
// Same as manual.store but for ACC→GM transfers
// ============================================================================
static std::string MakeManualL0CStoreCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  // Same as manual.store — TSTORE handles ACC→GM
  return MakeManualStoreCodegenCCE(op, codegen_base);
}

// ============================================================================
// manual.make_tile — no-op (tile already declared in prologue)
// ============================================================================
static std::string MakeManualMakeTileCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  (void)op;
  (void)codegen_base;
  return "";
}

// ============================================================================
// manual.move — args = [src, dst]
// Emits: TMOV(dst, src);
// ============================================================================
static std::string MakeManualMoveCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "manual.move: expected 2 args (src, dst), got " << op->args_.size();
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetExprAsCode(op->args_[1]);

  // Check for acc_to_vec_mode kwarg
  if (op->HasKwarg("acc_to_vec_mode")) {
    const std::string& mode_str = op->GetKwarg<std::string>("acc_to_vec_mode");
    std::string mode_enum;
    if (mode_str == "single_vec0") {
      mode_enum = "AccToVecMode::SingleModeVec0";
    } else if (mode_str == "single_vec1") {
      mode_enum = "AccToVecMode::SingleModeVec1";
    } else if (mode_str == "dual_split_m") {
      mode_enum = "AccToVecMode::DualModeSplitM";
    } else if (mode_str == "dual_split_n") {
      mode_enum = "AccToVecMode::DualModeSplitN";
    } else {
      throw pypto::ValueError("Invalid acc_to_vec_mode: " + mode_str);
    }
    // Generate TMOV with explicit template parameters: <DstType, SrcType, AccToVecMode>
    codegen.Emit("TMOV<decltype(" + dst + "), decltype(" + src + "), " + mode_enum + ">(" + dst + ", " + src + ");");
  } else {
    codegen.Emit("TMOV(" + dst + ", " + src + ");");
  }
  return "";
}

// ============================================================================
// manual.matmul — args = [left, right, dst]
// Emits: TMATMUL(dst, left, right);
// ============================================================================
static std::string MakeManualMatmulCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "manual.matmul: expected 3 args (left, right, dst), got "
                               << op->args_.size();
  std::string left = codegen.GetExprAsCode(op->args_[0]);
  std::string right = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetExprAsCode(op->args_[2]);
  codegen.Emit("TMATMUL(" + dst + ", " + left + ", " + right + ");");
  return "";
}

// manual.matmul_acc — args = [acc, left, right, dst]
static std::string MakeManualMatmulAccCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 4) << "manual.matmul_acc: expected 4 args, got " << op->args_.size();
  std::string acc = codegen.GetExprAsCode(op->args_[0]);
  std::string left = codegen.GetExprAsCode(op->args_[1]);
  std::string right = codegen.GetExprAsCode(op->args_[2]);
  std::string dst = codegen.GetExprAsCode(op->args_[3]);
  codegen.Emit("TMATMUL_ACC(" + dst + ", " + acc + ", " + left + ", " + right + ");");
  return "";
}

// manual.cast — args = [src, dst] + mode kwarg
static std::string MakeManualCastCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "manual.cast: expected 2 args (src, dst), got " << op->args_.size();
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetExprAsCode(op->args_[1]);

  // Get round mode from kwargs — manual.cast uses a string mode
  const std::string& mode_str = op->GetKwarg<std::string>("mode");
  static const std::vector<std::string> kModeNames = {"none", "rint",  "round", "floor",
                                                       "ceil", "trunc", "odd",   "cast_rint"};
  int mode_idx = -1;
  for (int i = 0; i < static_cast<int>(kModeNames.size()); ++i) {
    if (kModeNames[i] == mode_str) {
      mode_idx = i;
      break;
    }
  }
  CHECK(mode_idx >= 0) << "manual.cast: unknown round mode '" << mode_str << "'";

  codegen.Emit("TCVT(" + dst + ", " + src + ", " +
               codegen.GetTypeConverter().ConvertCastRoundMode(mode_idx) + ");");
  return "";
}

// manual.full/expands — args = [scalar, dst]
static std::string MakeManualExpandsCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "manual.full/expands: expected 2 args (scalar, dst), got "
                               << op->args_.size();
  std::string scalar = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetExprAsCode(op->args_[1]);
  codegen.Emit("TEXPANDS(" + dst + ", " + scalar + ");");
  return "";
}

// manual.fillpad — args = [src, dst]
static std::string MakeManualFillpadCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  return MakeManualUnaryCodegenCCE("TFILLPAD", op, codegen_base);
}

// manual.reshape — args = [src, shape, dst]
static std::string MakeManualReshapeCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "manual.reshape: expected 3 args (src, shape, dst), got "
                               << op->args_.size();
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetExprAsCode(op->args_[2]);
  codegen.Emit("TRESHAPE(" + dst + ", " + src + ");");
  return "";
}

// manual.transpose — args = [src, dst]
static std::string MakeManualTransposeCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "manual.transpose: expected 2 args (src, dst), got " << op->args_.size();
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetExprAsCode(op->args_[1]);
  codegen.Emit("TTRANS(" + dst + ", " + src + ");");
  return "";
}

// manual.cmp — args = [lhs, rhs, dst] + cmp_type kwarg
static std::string MakeManualCmpCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "manual.cmp: expected 3 args (lhs, rhs, dst), got " << op->args_.size();
  std::string lhs = codegen.GetExprAsCode(op->args_[0]);
  std::string rhs = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetExprAsCode(op->args_[2]);
  int cmp_type = op->GetKwarg<int>("cmp_type");
  codegen.Emit("TCMP(" + dst + ", " + lhs + ", " + rhs + ", " + std::to_string(cmp_type) + ");");
  return "";
}

// manual.cmps — args = [tile, scalar, dst] + cmp_type kwarg
static std::string MakeManualCmpsCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "manual.cmps: expected 3 args (tile, scalar, dst), got "
                               << op->args_.size();
  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string scalar = codegen.GetExprAsCode(op->args_[1]);
  std::string dst = codegen.GetExprAsCode(op->args_[2]);
  int cmp_type = op->GetKwarg<int>("cmp_type");
  codegen.Emit("TCMPS(" + dst + ", " + tile + ", " + scalar + ", " + std::to_string(cmp_type) + ");");
  return "";
}

// manual.ub_copy — args = [src, dst] (Vec→Vec copy)
static std::string MakeManualUbCopyCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  return MakeManualMoveCodegenCCE(op, codegen_base);
}

// manual.col_expand — args = [src, dst]
static std::string MakeManualColExpandCodegenCCE(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "manual.col_expand: expected 2 args (src, dst), got " << op->args_.size();
  // TCOLEXPAND is inplace: dst and src share same buffer
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetExprAsCode(op->args_[1]);
  codegen.Emit("TCOLEXPAND(" + dst + ", " + src + ");");
  return "";
}

// manual.row_expand — args = [src, dst]
static std::string MakeManualRowExpandUnaryCodegenCCE(const ir::CallPtr& op,
                                                       codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::CCECodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "manual.row_expand: expected 2 args (src, dst), got " << op->args_.size();
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string dst = codegen.GetExprAsCode(op->args_[1]);
  codegen.Emit("TROWEXPAND(" + dst + ", " + src + ");");
  return "";
}

// ============================================================================
// Op registrations — Memory
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.load")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualLoadCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.store")
    .set_pipe(ir::PipeType::MTE3)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualStoreCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.l0c_store")
    .set_pipe(ir::PipeType::FIX)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualL0CStoreCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.make_tile")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualMakeTileCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.move")
    .set_pipe(ir::PipeType::MTE1)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualMoveCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.ub_copy")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUbCopyCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.full")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualExpandsCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.fillpad")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualFillpadCodegenCCE(op, codegen);
    });

// ============================================================================
// Op registrations — Tile x Tile binary
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TADD", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TSUB", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TMUL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TDIV", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.rem")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TREM", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.maximum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TMAX", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.minimum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TMIN", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.and")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TAND", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.or")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TOR", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.shl")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TSHL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.shr")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCodegenCCE("TSHR", op, codegen);
    });

// ============================================================================
// Op registrations — Tile x Scalar binary
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.adds")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TADDS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.subs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TSUBS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.muls")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TMULS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.divs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TDIVS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.rems")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TREMS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.ands")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TANDS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.ors")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TORS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.shls")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TSHLS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.shrs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TSHRS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.maxs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TMAXS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.mins")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TMINS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.lrelu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualScalarCodegenCCE("TLRELU", op, codegen);
    });

// ============================================================================
// Op registrations — Unary
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.neg")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryCodegenCCE("TNEG", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.exp")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryCodegenCCE("TEXP", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.sqrt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryCodegenCCE("TSQRT", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.rsqrt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryCodegenCCE("TRSQRT", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.recip")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryCodegenCCE("TRECIP", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.log")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryCodegenCCE("TLOG", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.abs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryCodegenCCE("TABS", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.relu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryCodegenCCE("TRELU", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.not")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryCodegenCCE("TNOT", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualCastCodegenCCE(op, codegen);
    });

// ============================================================================
// Op registrations — Comparison
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.cmp")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualCmpCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.cmps")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualCmpsCodegenCCE(op, codegen);
    });

// ============================================================================
// Op registrations — Scalar broadcast
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.expands")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualExpandsCodegenCCE(op, codegen);
    });

// ============================================================================
// Op registrations — Reductions (tile, tmp, out)
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.row_sum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualRowReductionCodegenCCE("TROWSUM", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.row_max")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualRowReductionCodegenCCE("TROWMAX", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.row_min")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualRowReductionCodegenCCE("TROWMIN", op, codegen);
    });

// ============================================================================
// Op registrations — Broadcast expansion
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.row_expand")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualRowExpandUnaryCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.row_expand_add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualRowExpandCodegenCCE("TROWEXPANDADD", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.row_expand_sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualRowExpandCodegenCCE("TROWEXPANDSUB", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.row_expand_mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualRowExpandCodegenCCE("TROWEXPANDMUL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.row_expand_div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualRowExpandCodegenCCE("TROWEXPANDDIV", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.col_expand")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualColExpandCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.col_expand_mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualRowExpandCodegenCCE("TCOLEXPANDMUL", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.col_expand_div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualRowExpandCodegenCCE("TCOLEXPANDDIV", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.col_expand_sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualRowExpandCodegenCCE("TCOLEXPANDSUB", op, codegen);
    });

// ============================================================================
// Op registrations — Matrix multiplication
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.matmul")
    .set_pipe(ir::PipeType::M)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualMatmulCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.matmul_acc")
    .set_pipe(ir::PipeType::M)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualMatmulAccCodegenCCE(op, codegen);
    });

// ============================================================================
// Op registrations — Layout operations
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.reshape")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualReshapeCodegenCCE(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_CCE, "manual.transpose")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTransposeCodegenCCE(op, codegen);
    });

}  // namespace backend
}  // namespace pypto
