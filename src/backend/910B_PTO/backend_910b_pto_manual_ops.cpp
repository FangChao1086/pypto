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
 * @file backend_910b_pto_manual_ops.cpp
 * @brief PTOAS IR code generation for manual (non-SSA) block operations.
 *
 * Each "manual.*" op carries an explicit pre-allocated output tile as its last
 * argument.  The helpers here build the ``ins(...)  outs(...)`` clause by
 * treating args[0..n-1] as inputs and args[n] as the outs target, rather than
 * calling GetCurrentResultTarget() for the output.
 *
 * Mapping mirrors the existing ``block.*`` → ``pto.*`` mapping in
 * backend_910b_pto_ops.cpp; only the argument counting and outs-clause
 * construction differ.
 */

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

#include "pypto/backend/910B_PTO/backend_910b_pto.h"
#include "pypto/backend/common/backend.h"
#include "pypto/codegen/codegen_base.h"
#include "pypto/codegen/pto/pto_codegen.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

using ir::As;
using ir::CallPtr;
using ir::PipeType;
using ir::TensorType;
using ir::Var;

// ============================================================================
// Shared mode tables (identical to backend_910b_pto_ops.cpp)
// ============================================================================

static const std::vector<std::string> kManualCmpModes = {"EQ", "NE", "LT", "LE", "GT", "GE"};
static const std::vector<std::string> kManualRoundModes = {"NONE", "RINT",  "ROUND", "FLOOR",
                                                           "CEIL", "TRUNC", "ODD",   "CAST_RINT"};

// ============================================================================
// Core helper: build ins(…) outs(…) for manual ops
//
// n_ins:    number of leading input arguments
// out_idx:  index of the explicit output tile argument (== n_ins)
// config_attr: optional attribute string inserted between ins type list
//              and the closing ')' e.g. "{cmpMode = #pto<cmp EQ>}"
// ============================================================================

static std::string GenerateManualInsOutsClause(const CallPtr& op, codegen::PTOCodegen& codegen,
                                               size_t n_ins,
                                               const std::string& config_attr = "") {
  CHECK(op->args_.size() == n_ins + 1)
      << "GenerateManualInsOutsClause: expected " << (n_ins + 1) << " args, got "
      << op->args_.size();

  std::ostringstream oss;

  // --- ins clause ---
  oss << "ins(";
  for (size_t i = 0; i < n_ins; ++i) {
    if (i > 0) oss << ", ";
    oss << codegen.GetExprAsCode(op->args_[i]);
  }
  // type annotations
  std::string type_annot;
  for (size_t i = 0; i < n_ins; ++i) {
    std::string annot = codegen.GetExprTypeAnnotation(op->args_[i]);
    if (!annot.empty()) {
      if (!type_annot.empty()) type_annot += ", ";
      type_annot += annot;
    }
  }
  if (!type_annot.empty()) oss << " : " << type_annot;
  if (!config_attr.empty()) oss << config_attr;

  // --- outs clause (explicit last arg) ---
  const size_t out_idx = n_ins;
  std::string out_name = codegen.GetExprAsCode(op->args_[out_idx]);
  std::string out_type = codegen.GetExprTypeAnnotation(op->args_[out_idx]);
  oss << ") outs(" << out_name;
  if (!out_type.empty()) oss << " : " << out_type;
  oss << ")";

  return oss.str();
}

// ============================================================================
// Arity-specific convenience wrappers
// ============================================================================

// Unary:  (src, out)
static std::string MakeManualUnaryPTO(const std::string& pto_op, const CallPtr& op,
                                      codegen::CodegenBase& cb) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(cb);
  CHECK(op->args_.size() == 2) << pto_op << ": expected 2 args (src, out), got " << op->args_.size();
  codegen.Emit(pto_op + " " + GenerateManualInsOutsClause(op, codegen, 1));
  return "";
}

// Binary: (lhs, rhs, out)
static std::string MakeManualBinaryPTO(const std::string& pto_op, const CallPtr& op,
                                       codegen::CodegenBase& cb) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(cb);
  CHECK(op->args_.size() == 3) << pto_op << ": expected 3 args (lhs, rhs, out), got "
                               << op->args_.size();
  codegen.Emit(pto_op + " " + GenerateManualInsOutsClause(op, codegen, 2));
  return "";
}

// Ternary: (a, b, c, out)
static std::string MakeManualTernaryPTO(const std::string& pto_op, const CallPtr& op,
                                        codegen::CodegenBase& cb) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(cb);
  CHECK(op->args_.size() == 4) << pto_op << ": expected 4 args (a, b, c, out), got "
                               << op->args_.size();
  codegen.Emit(pto_op + " " + GenerateManualInsOutsClause(op, codegen, 3));
  return "";
}

// Quaternary: (a, b, c, d, out)
static std::string MakeManualQuaternaryPTO(const std::string& pto_op, const CallPtr& op,
                                           codegen::CodegenBase& cb) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(cb);
  CHECK(op->args_.size() == 5) << pto_op << ": expected 5 args (a, b, c, d, out), got "
                               << op->args_.size();
  codegen.Emit(pto_op + " " + GenerateManualInsOutsClause(op, codegen, 4));
  return "";
}

// Comparison (binary + cmp_mode attr): (lhs, rhs, out) + cmp_type kwarg
static std::string MakeManualCmpPTO(const std::string& pto_op, const CallPtr& op,
                                    codegen::CodegenBase& cb) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(cb);
  CHECK(op->args_.size() == 3) << pto_op << ": expected 3 args (lhs, rhs, out)";
  int mode = op->GetKwarg<int>("cmp_type");
  CHECK(mode >= 0 && mode < static_cast<int>(kManualCmpModes.size()))
      << pto_op << ": cmp_type out of range: " << mode;
  std::string attr = "{cmpMode = #pto<cmp " + kManualCmpModes.at(mode) + ">}";
  codegen.Emit(pto_op + " " + GenerateManualInsOutsClause(op, codegen, 2, attr));
  return "";
}

// cmps: (tile, scalar, out) + cmp_type kwarg
static std::string MakeManualCmpsPTO(const std::string& pto_op, const CallPtr& op,
                                     codegen::CodegenBase& cb) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(cb);
  CHECK(op->args_.size() == 3) << pto_op << ": expected 3 args (tile, scalar, out)";
  int mode = op->GetKwarg<int>("cmp_type");
  CHECK(mode >= 0 && mode < static_cast<int>(kManualCmpModes.size()))
      << pto_op << ": cmp_type out of range: " << mode;
  std::string attr = "{cmpMode = #pto<cmp " + kManualCmpModes.at(mode) + ">}";
  codegen.Emit(pto_op + " " + GenerateManualInsOutsClause(op, codegen, 2, attr));
  return "";
}

// Cast / type-convert (unary + round_mode attr): (src, out) + mode kwarg
static std::string MakeManualCvtPTO(const std::string& pto_op, const CallPtr& op,
                                    codegen::CodegenBase& cb) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(cb);
  CHECK(op->args_.size() == 2) << pto_op << ": expected 2 args (src, out)";
  const std::string& mode_str = op->GetKwarg<std::string>("mode");
  // Map mode string → enum index for PTOAS IR attribute.
  static const std::vector<std::string> kModeNames = {"none", "rint",  "round", "floor",
                                                      "ceil", "trunc", "odd",   "cast_rint"};
  int mode_idx = -1;
  for (int i = 0; i < static_cast<int>(kModeNames.size()); ++i) {
    if (kModeNames[i] == mode_str) { mode_idx = i; break; }
  }
  CHECK(mode_idx >= 0) << pto_op << ": unknown round mode '" << mode_str << "'";
  std::string attr = "{rmode = #pto<round_mode " + kManualRoundModes.at(mode_idx) + ">}";
  codegen.Emit(pto_op + " " + GenerateManualInsOutsClause(op, codegen, 1, attr));
  return "";
}

// Scalar-to-tile broadcast: (scalar, out)
// Generates: pto.texpands ins(%scalar : scalar_type) outs(%out : tile_type)
static std::string MakeManualExpandsPTO(const std::string& pto_op, const CallPtr& op,
                                        codegen::CodegenBase& cb) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(cb);
  CHECK(op->args_.size() == 2) << pto_op << ": expected 2 args (scalar, out)";
  std::string scalar = codegen.GetExprAsCode(op->args_[0]);
  std::string scalar_type = codegen.GetExprTypeAnnotation(op->args_[0]);
  std::string out = codegen.GetExprAsCode(op->args_[1]);
  std::string out_type = codegen.GetExprTypeAnnotation(op->args_[1]);
  std::ostringstream oss;
  oss << pto_op << " ins(" << scalar;
  if (!scalar_type.empty()) oss << " : " << scalar_type;
  oss << ") outs(" << out;
  if (!out_type.empty()) oss << " : " << out_type;
  oss << ")";
  codegen.Emit(oss.str());
  return "";
}

// ============================================================================
// manual.load codegen
//
// Emits:
//   %pv = pto.partition_view %tensor_view, offsets=[...], sizes=[...] : T -> PTV
//   pto.tload ins(%pv : PTV) outs(%out : TileBufType)
// ============================================================================

static std::string MakeManualLoadCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);

  auto tensor = As<Var>(op->args_[0]);
  INTERNAL_CHECK(tensor) << "manual.load: first argument must be a Var";

  auto offsets_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK(offsets_tuple) << "manual.load: second argument must be a MakeTuple (offsets)";

  auto shapes_tuple = As<ir::MakeTuple>(op->args_[2]);
  INTERNAL_CHECK(shapes_tuple) << "manual.load: third argument must be a MakeTuple (shapes)";

  auto out_tile = As<Var>(op->args_[3]);
  INTERNAL_CHECK(out_tile) << "manual.load: fourth argument (out) must be a Var";

  auto tile_type = As<ir::TileType>(out_tile->GetType());
  INTERNAL_CHECK(tile_type) << "manual.load: fourth argument (out) must be a Tile";

  auto row_off = codegen.GetExprAsCode(offsets_tuple->elements_[0]);
  auto col_off = codegen.GetExprAsCode(offsets_tuple->elements_[1]);
  int64_t height  = codegen.GetConstIntValue(tile_type->shape_[0]);
  int64_t width   = codegen.GetConstIntValue(tile_type->shape_[1]);

  auto tensor_type = As<TensorType>(tensor->GetType());
  INTERNAL_CHECK(tensor_type) << "manual.load: tensor argument must have TensorType";

  std::string tensor_view = codegen.GetOrCreateTensorView(tensor);
  std::string dtype_str   = codegen.GetTypeString(tensor_type->dtype_);
  std::string tile_buf    = codegen.GetVarName(out_tile);

  std::string tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());
  std::string partition_type   = "!pto.partition_tensor_view<" + std::to_string(height) + "x" +
                                 std::to_string(width) + "x" + dtype_str + ">";

  // Emit partition_view.
  std::string partition_view = codegen.NewTemp();
  std::ostringstream pv_line;
  pv_line << partition_view << " = pto.partition_view " << tensor_view
          << ", offsets = [" << row_off << ", " << col_off << "]"
          << ", sizes = ["   << codegen.GetIndexConstant(height)  << ", "
          << codegen.GetIndexConstant(width) << "]"
          << " : " << tensor_view_type << " -> " << partition_type;
  codegen.Emit(pv_line.str());

  // Derive tile_buf type from the out_tile's TileType memref (may be empty before alloc pass).
  std::string tile_buf_type = codegen.GetTileBufTypeStringFromTileType(tile_type);

  // Emit set_validshape only for dynamic tiles.
  // Static tiles don't need valid shape; ptoas also doesn't support pto.set_validshape for them.
  if (!shapes_tuple->elements_.empty()) {
    bool is_dynamic = tile_buf_type.find("v_row=?, v_col=?") != std::string::npos;
    INTERNAL_CHECK(is_dynamic) << "manual.load: only dynamic tile can set valid shape";
    std::ostringstream set_line;
    auto valid_row = codegen.GetExprAsCode(shapes_tuple->elements_[0]);
    auto valid_col = codegen.GetExprAsCode(shapes_tuple->elements_[1]);
    set_line << "pto.set_validshape " << tile_buf << ", "
              << valid_row << ", " << valid_col << " : " << tile_buf_type;
    codegen.Emit(set_line.str());
  }

  // Emit tload using the explicit out_tile as outs target.
  std::ostringstream tload_line;
  tload_line << "pto.tload ins(" << partition_view << " : " << partition_type
             << ") outs(" << tile_buf << " : " << tile_buf_type << ")";
  codegen.Emit(tload_line.str());

  return "";
}

// ============================================================================
// manual.store codegen
//
// Emits:
//   %pv = pto.partition_view %tensor_view, offsets=[...], sizes=[...] : T -> PTV
//   (optional) pto.set_validshape %tile_buf, row, col : TileBufType
//   pto.tstore ins(%tile_buf : TileBufType) outs(%pv : PTV)
// ============================================================================

static std::string MakeManualStoreCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);

  auto tile = As<Var>(op->args_[0]);
  INTERNAL_CHECK(tile) << "manual.store: first argument must be a Var";

  auto tile_type = As<ir::TileType>(tile->GetType());
  INTERNAL_CHECK(tile_type) << "manual.store: first argument must be a Tile";

  auto offsets_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK(offsets_tuple) << "manual.store: second argument must be a MakeTuple (offsets)";

  auto shapes_tuple = As<ir::MakeTuple>(op->args_[2]);
  INTERNAL_CHECK(shapes_tuple) << "manual.store: third argument must be a MakeTuple (shapes)";

  auto output_tensor = As<Var>(op->args_[3]);
  INTERNAL_CHECK(output_tensor) << "manual.store: fourth argument must be a Var";

  auto tensor_type = As<TensorType>(output_tensor->GetType());
  INTERNAL_CHECK(tensor_type) << "manual.store: fourth argument must have TensorType";

  auto row_off = codegen.GetExprAsCode(offsets_tuple->elements_[0]);
  auto col_off = codegen.GetExprAsCode(offsets_tuple->elements_[1]);
  int64_t height = codegen.GetConstIntValue(tile_type->shape_[0]);
  int64_t width  = codegen.GetConstIntValue(tile_type->shape_[1]);

  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
  std::string tensor_view = codegen.GetOrCreateTensorView(output_tensor);
  std::string tile_buf = codegen.GetVarName(tile);

  std::string tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());
  std::string partition_type   = "!pto.partition_tensor_view<" + std::to_string(height) + "x" +
                                 std::to_string(width) + "x" + dtype_str + ">";

  std::string tile_buf_type = codegen.GetTileBufTypeStringFromTileType(tile_type);

  std::string partition_view = codegen.NewTemp();
  std::ostringstream pv_line;
  pv_line << partition_view << " = pto.partition_view " << tensor_view
          << ", offsets = [" << row_off << ", " << col_off << "]"
          << ", sizes = [" << codegen.GetIndexConstant(height) << ", "
          << codegen.GetIndexConstant(width) << "]"
          << " : " << tensor_view_type << " -> " << partition_type;
  codegen.Emit(pv_line.str());

  // Emit set_validshape only for dynamic tiles.
  // Static tiles don't need valid shape; ptoas also doesn't support pto.set_validshape for them.
  if (!shapes_tuple->elements_.empty()) {
    bool is_dynamic = tile_buf_type.find("v_row=?, v_col=?") != std::string::npos;
    if (is_dynamic) {
      std::ostringstream set_line;
      auto valid_row = codegen.GetExprAsCode(shapes_tuple->elements_[0]);
      auto valid_col = codegen.GetExprAsCode(shapes_tuple->elements_[1]);
      set_line << "pto.set_validshape " << tile_buf << ", "
               << valid_row << ", " << valid_col << " : " << tile_buf_type;
      codegen.Emit(set_line.str());
    }
  }

  std::ostringstream tstore_line;
  tstore_line << "pto.tstore ins(" << tile_buf;
  if (!tile_buf_type.empty()) {
    tstore_line << " : " << tile_buf_type;
  }
  tstore_line << ") outs(" << partition_view << " : " << partition_type << ")";
  codegen.Emit(tstore_line.str());

  return "";
}

// ============================================================================
// Op registrations
// ============================================================================

// ----------------------------------------------------------------------------
// Memory
// ----------------------------------------------------------------------------

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.load")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualLoadCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.store")
    .set_pipe(ir::PipeType::MTE3)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualStoreCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.move")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryPTO("pto.tmov", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.ub_copy")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryPTO("pto.tmov", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.full")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualExpandsPTO("pto.texpands", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.fillpad")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryPTO("pto.tfillpad", op, codegen);
    });

// ----------------------------------------------------------------------------
// Tile x Tile binary
// ----------------------------------------------------------------------------

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tadd", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tsub", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tmul", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tdiv", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.rem")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.trem", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.maximum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tmax", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.minimum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tmin", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.and")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tand", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.or")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tor", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.shl")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tshl", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.shr")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tshr", op, codegen);
    });

// ----------------------------------------------------------------------------
// Tile x Scalar binary
// ----------------------------------------------------------------------------

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.adds")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tadds", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.subs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tsubs", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.muls")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tmuls", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.divs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tdivs", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.rems")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.trems", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.ands")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tands", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.ors")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tors", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.shls")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tshls", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.shrs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tshrs", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.maxs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tmaxs", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.mins")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tmins", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.lrelu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tlrelu", op, codegen);
    });

// ----------------------------------------------------------------------------
// Unary operations
// ----------------------------------------------------------------------------

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.neg")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryPTO("pto.tneg", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.exp")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryPTO("pto.texp", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.sqrt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryPTO("pto.tsqrt", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.rsqrt")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryPTO("pto.trsqrt", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.recip")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryPTO("pto.trecip", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.log")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryPTO("pto.tlog", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.abs")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryPTO("pto.tabs", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.relu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryPTO("pto.trelu", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.not")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryPTO("pto.tnot", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualCvtPTO("pto.tcvt", op, codegen);
    });

// ----------------------------------------------------------------------------
// Ternary / multi-input
// ----------------------------------------------------------------------------

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.xor")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryPTO("pto.txor", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.xors")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryPTO("pto.txors", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.prelu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryPTO("pto.tprelu", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.addc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryPTO("pto.taddc", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.subc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryPTO("pto.tsubc", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.addsc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryPTO("pto.taddsc", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.subsc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryPTO("pto.tsubsc", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.sel")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryPTO("pto.tsel", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.sels")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryPTO("pto.tsels", op, codegen);
    });

// ----------------------------------------------------------------------------
// Comparison
// ----------------------------------------------------------------------------

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.cmp")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualCmpPTO("pto.tcmp", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.cmps")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualCmpsPTO("pto.tcmps", op, codegen);
    });

// ----------------------------------------------------------------------------
// Scalar broadcast
// ----------------------------------------------------------------------------

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.expands")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualExpandsPTO("pto.texpands", op, codegen);
    });

// ----------------------------------------------------------------------------
// Reductions (tile, tmp, out)
// ----------------------------------------------------------------------------

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.row_sum")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.trowsum", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.row_max")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.trowmax", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.row_min")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.trowmin", op, codegen);
    });

// ----------------------------------------------------------------------------
// Broadcast expansion
// ----------------------------------------------------------------------------

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.row_expand")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryPTO("pto.trowexpand", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.row_expand_add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.trowexpandadd", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.row_expand_sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.trowexpandsub", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.row_expand_mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.trowexpandmul", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.row_expand_div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.trowexpanddiv", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.col_expand")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryPTO("pto.tcolexpand", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.col_expand_mul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tcolexpandmul", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.col_expand_div")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tcolexpanddiv", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.col_expand_sub")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tcolexpandsub", op, codegen);
    });

// ----------------------------------------------------------------------------
// Matrix multiplication
// ----------------------------------------------------------------------------

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.matmul")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tmatmul", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.matmul_acc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryPTO("pto.tmatmul.acc", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.matmul_bias")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryPTO("pto.tmatmul.bias", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.gemv")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tgemv", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.gemv_acc")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryPTO("pto.tgemv.acc", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.gemv_bias")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualTernaryPTO("pto.tgemv.bias", op, codegen);
    });

// ----------------------------------------------------------------------------
// Layout operations
// ----------------------------------------------------------------------------

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.reshape")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      // (src, shape_tuple, out): emit reshape with explicit out as outs target.
      auto& c = dynamic_cast<codegen::PTOCodegen&>(codegen);
      CHECK(op->args_.size() == 3) << "manual.reshape: expected 3 args (src, shape, out)";
      std::string src      = c.GetExprAsCode(op->args_[0]);
      std::string src_type = c.GetExprTypeAnnotation(op->args_[0]);
      std::string out      = c.GetExprAsCode(op->args_[2]);
      std::string out_type = c.GetExprTypeAnnotation(op->args_[2]);
      std::ostringstream oss;
      oss << "pto.treshape ins(" << src;
      if (!src_type.empty()) oss << " : " << src_type;
      oss << ") outs(" << out;
      if (!out_type.empty()) oss << " : " << out_type;
      oss << ")";
      c.Emit(oss.str());
      return "";
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.transpose")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryPTO("pto.ttrans", op, codegen);
    });

// ----------------------------------------------------------------------------
// Synchronization
// ----------------------------------------------------------------------------

static std::string MakeManualSyncPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  
  const std::string& sync_type = op->GetKwarg<std::string>("sync_type");
  
  std::ostringstream oss;
  
  if (sync_type == "inner_core_sync") {
    // 检查是否有pipeline参数
    if (op->HasKwarg("pipeline")) {
      const std::string& pipeline = op->GetKwarg<std::string>("pipeline");
      
      // 方案一：直接使用流水线类型，不需要映射到操作类型
      oss << "pto.barrier_sync [#pto<pipe " << pipeline << ">]";
    } else {
      // 向后兼容：无pipeline参数，默认PIPE_ALL
      oss << "pto.barrier_sync [#pto<pipe PIPE_ALL>]";
    }
  } else if (sync_type == "cross_core_sync_forward") {
    int event_id = op->GetKwarg<int>("event_id");
    oss << "pto.record_event [#pto<sync_op_type TMATMUL>, #pto<sync_op_type TVEC>, #pto<event EVENT_ID" << event_id << ">]";
  } else if (sync_type == "cross_core_sync_both") {
    int event_id = op->GetKwarg<int>("event_id");
    const std::string& direction = op->GetKwarg<std::string>("direction");
    
    if (direction == "record") {
      oss << "pto.record_event [#pto<sync_op_type TMATMUL>, #pto<sync_op_type TVEC>, #pto<event EVENT_ID" << event_id << ">]";
    } else if (direction == "allocate") {
      oss << "pto.wait_event [#pto<sync_op_type TMATMUL>, #pto<sync_op_type TVEC>, #pto<event EVENT_ID" << event_id << ">]";
    } else {
      LOG_ERROR << "manual.sync: unknown direction '" << direction << "'";
      return "";
    }
  } else {
    LOG_ERROR << "manual.sync: unknown sync_type '" << sync_type << "'";
    return "";
  }
  
  codegen.Emit(oss.str());
  return "";
}

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.sync")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualSyncPTO(op, codegen);
    });

}  // namespace backend
}  // namespace pypto
