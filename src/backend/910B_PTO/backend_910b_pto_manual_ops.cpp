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
// config_attr: optional attribute string inserted after the ins operand list
//              and before the type annotation, e.g. "{cmpMode = #pto<cmp EQ>}"
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
  if (!config_attr.empty()) oss << config_attr;
  if (!type_annot.empty()) oss << " : " << type_annot;

  // --- outs clause (explicit last arg) ---
  const size_t out_idx = n_ins;
  std::string out_name = codegen.GetExprAsCode(op->args_[out_idx]);
  std::string out_type = codegen.GetExprTypeAnnotation(op->args_[out_idx]);
  oss << ") outs(" << out_name;
  if (!out_type.empty()) oss << " : " << out_type;
  oss << ")";

  return oss.str();
}

static std::string GenerateManualInsMidOutsClause(const CallPtr& op, codegen::PTOCodegen& codegen,
                                                  size_t n_ins,
                                                  const std::string& config_attr = "") {
  CHECK(op->args_.size() == n_ins + 1)
      << "GenerateManualInsMidOutsClause: expected " << (n_ins + 1) << " args, got "
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
  const size_t out_idx = 0;
  std::string out_name = codegen.GetExprAsCode(op->args_[out_idx]);
  std::string out_type = codegen.GetExprTypeAnnotation(op->args_[out_idx]);
  oss << ") outs(" << out_name;
  if (!out_type.empty()) oss << " : " << out_type;
  oss << ")";

  return oss.str();
}

static std::string GenerateManualMixSelfInsOutsClause(const CallPtr& op, codegen::PTOCodegen& codegen,
                                                      size_t n_ins, size_t out_ins,
                                                      const std::string& config_attr = "") {
  CHECK(op->args_.size() == n_ins + 1)
      << "GenerateManualInsMidOutsClause: expected " << (n_ins + 1) << " args, got "
      << op->args_.size();

  std::ostringstream oss;

  // --- ins clause ---
  oss << "ins(";
  for (size_t i = 0; i < n_ins + 1; i+=2) {
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
  const size_t out_idx = out_ins;
  std::string out_name = codegen.GetExprAsCode(op->args_[out_idx]);
  std::string out_type = codegen.GetExprTypeAnnotation(op->args_[out_idx]);
  oss << ") outs(" << out_name;
  if (!out_type.empty()) oss << " : " << out_type;
  oss << ")";

  return oss.str();
}

static std::string GenerateManualMixInsOutsClause(const CallPtr& op, codegen::PTOCodegen& codegen,
                                                  size_t n_ins,
                                                  const std::string& config_attr = "") {
  std::ostringstream oss;

  // --- ins clause ---
  oss << "ins(";
  oss << codegen.GetExprAsCode(op->args_[0]);
  if (!config_attr.empty()) oss << config_attr;
  // type annotations
  std::string type_annot;
  std::string annot = codegen.GetExprTypeAnnotation(op->args_[0]);
  if (!annot.empty()) {
    if (!type_annot.empty()) type_annot += ", ";
    type_annot += annot;
  }
  if (!type_annot.empty()) oss << " : " << type_annot;
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

// Mix Binary: (lhs, rhs, out)
static std::string MakeManualMixBinaryPTO(const std::string& pto_op1, const std::string& pto_op2,
                                          const CallPtr& op, codegen::CodegenBase& cb) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(cb);
  CHECK(op->args_.size() == 3) << pto_op1 << ": expected 3 args (lhs, rhs, out), got "
                                << op->args_.size();
  codegen.Emit(pto_op1 + " " + GenerateManualInsMidOutsClause(op, codegen, 2));
  codegen.Emit(pto_op2 + " " + GenerateManualMixInsOutsClause(op, codegen, 2));
  return "";
}

// Mix Self Binary: (lhs, rhs, out)
static std::string MakeManualMixSelfBinaryPTO(const std::string& pto_op1, const std::string& pto_op2,
                                          const CallPtr& op, codegen::CodegenBase& cb) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(cb);
  CHECK(op->args_.size() == 3) << pto_op1 << ": expected 3 args (lhs, rhs, out), got "
                                << op->args_.size();
  codegen.Emit(pto_op1 + " " + GenerateManualInsMidOutsClause(op, codegen, 2));
  codegen.Emit(pto_op2 + " " + GenerateManualMixSelfInsOutsClause(op, codegen, 2, 2));
  return "";
}

// Mix Binary Swap: (lhs, rhs, out) - lhs used in both ops, rhs used in second op
// Generates: pto_op1(lhs, out, out), pto_op2(rhs, out, out)
static std::string MakeManualMixFusedBinaryPTO(const std::string& pto_op1, const std::string& pto_op2,
                                          const CallPtr& op, codegen::CodegenBase& cb) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(cb);
  CHECK(op->args_.size() == 3) << pto_op1 << ": expected 3 args (lhs, rhs, out), got "
                                << op->args_.size();
  codegen.Emit(pto_op1 + " " + GenerateManualMixSelfInsOutsClause(op, codegen, 2, 0));
  codegen.Emit(pto_op2 + " " + GenerateManualInsOutsClause(op, codegen, 2));
  return "";
}

// Mix Binary Swap with ReLU: (lhs, rhs, out) - lhs used in both ops, rhs used in second op, then ReLU
// Generates: pto_op1(lhs, out, out), pto_op2(rhs, out, out), pto_op3(out, out)
static std::string MakeManualMixFusedReluBinaryReluPTO(const std::string& pto_op1, const std::string& pto_op2,
                                          const std::string& pto_op3, const CallPtr& op,
                                          codegen::CodegenBase& cb) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(cb);
  CHECK(op->args_.size() == 3) << pto_op1 << ": expected 3 args (lhs, rhs, out), got "
                                << op->args_.size();
  codegen.Emit(pto_op1 + " " + GenerateManualMixSelfInsOutsClause(op, codegen, 2, 0));
  codegen.Emit(pto_op2 + " " + GenerateManualInsMidOutsClause(op, codegen, 2));
  codegen.Emit(pto_op3 + " " + GenerateManualMixInsOutsClause(op, codegen, 2));
  return "";
}

// Mix Binary with Cast: (lhs, rhs, out) + target_type, mode kwargs
// Generates: pto_op1(lhs, rhs, out), pto_op2(out, out), pto_op3cvt(out, out) with cast attributes
static std::string MakeManualMixBinaryCvtPTO(const std::string& pto_op1, const std::string& pto_op2,
                                             const std::string& pto_op3cvt, const CallPtr& op,
                                             codegen::CodegenBase& cb) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(cb);
  CHECK(op->args_.size() == 3) << pto_op1 << ": expected 3 args (lhs, rhs, out), got "
                               << op->args_.size();
  
  // Step 1: First binary op (e.g., pto.tadd)
  codegen.Emit(pto_op1 + " " + GenerateManualInsMidOutsClause(op, codegen, 2));
  
  // Step 2: Second unary op (e.g., pto.trelu)
  codegen.Emit(pto_op2 + " " + GenerateManualMixInsOutsClause(op, codegen, 0));
  
  // Step 3: Cast op with mode attribute
  const std::string& mode_str = op->GetKwarg<std::string>("mode");
  static const std::vector<std::string> kModeNames = {"none", "rint",  "round", "floor",
                                                      "ceil", "trunc", "odd",   "cast_rint"};
  int mode_idx = -1;
  for (int i = 0; i < static_cast<int>(kModeNames.size()); ++i) {
    if (kModeNames[i] == mode_str) { mode_idx = i; break; }
  }
  CHECK(mode_idx >= 0) << pto_op3cvt << ": unknown round mode '" << mode_str << "'";
  std::string attr = "{rmode = #pto<round_mode " + kManualRoundModes.at(mode_idx) + ">}";
  codegen.Emit(pto_op3cvt + " " + GenerateManualMixInsOutsClause(op, codegen, 2, attr));
  
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

// Binary with Cast: (lhs, rhs, out) + target_type, mode kwargs
// Generates: pto_op1(lhs, rhs, out), pto_op2cvt(out, out) with cast attributes
static std::string MakeManualBinaryCvtPTO(const std::string& pto_op1, const std::string& pto_op2cvt,
                                          const CallPtr& op, codegen::CodegenBase& cb) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(cb);
  CHECK(op->args_.size() == 3) << pto_op1 << ": expected 3 args (lhs, rhs, out), got "
                               << op->args_.size();
  
  // Step 1: First binary op (e.g., pto.tmul)
  codegen.Emit(pto_op1 + " " + GenerateManualInsMidOutsClause(op, codegen, 2));
  
  // Step 2: Cast op with mode attribute
  const std::string& mode_str = op->GetKwarg<std::string>("mode");
  static const std::vector<std::string> kModeNames = {"none", "rint",  "round", "floor",
                                                      "ceil", "trunc", "odd",   "cast_rint"};
  int mode_idx = -1;
  for (int i = 0; i < static_cast<int>(kModeNames.size()); ++i) {
    if (kModeNames[i] == mode_str) { mode_idx = i; break; }
  }
  CHECK(mode_idx >= 0) << pto_op2cvt << ": unknown round mode '" << mode_str << "'";
  std::string attr = "{rmode = #pto<round_mode " + kManualRoundModes.at(mode_idx) + ">}";
  codegen.Emit(pto_op2cvt + " " + GenerateManualMixInsOutsClause(op, codegen, 2, attr));
  
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

  auto out_tile = As<Var>(op->args_[2]);
  INTERNAL_CHECK(out_tile) << "manual.load: third argument (out) must be a Var";

  auto tile_type = As<ir::TileType>(out_tile->GetType());
  INTERNAL_CHECK(tile_type) << "manual.load: third argument (out) must be a Tile";

  auto tensor_type = As<TensorType>(tensor->GetType());
  INTERNAL_CHECK(tensor_type) << "manual.load: tensor argument must have TensorType";

  std::string dtype_str   = codegen.GetTypeString(tensor_type->dtype_);
  std::string tile_buf    = codegen.GetVarName(out_tile);
  std::string tile_buf_type = codegen.GetTileBufTypeStringFromTileType(tile_type);

  // Check for DN (column-major) layout
  bool is_dn = op->HasKwarg("layout") && op->GetKwarg<std::string>("layout") == "dn";

  std::string view_for_partition;
  std::string tensor_view_type;
  std::string row_off, col_off;

  if (is_dn) {
    // DN layout: emit a transposed make_tensor_view from the raw pointer.
    // Original tensor shape: [dim0, dim1], strides: [dim1, 1] (row-major)
    // DN view: shape = [dim1, dim0], strides = [1, dim1] (column-major)
    // This tells TLOAD to read column-major, achieving on-chip transpose.
    std::string raw_ptr = codegen.GetTensorPtr(tensor);

    // Get original tensor dimensions
    INTERNAL_CHECK(tensor_type->shape_.size() == 2)
        << "manual.load DN layout: tensor must be 2D";
    std::string orig_dim0, orig_dim1;
    if (auto var0 = As<ir::Var>(tensor_type->shape_[0])) {
      orig_dim0 = codegen.GetVarName(var0);
    } else {
      orig_dim0 = codegen.GetIndexConstant(codegen.GetConstIntValue(tensor_type->shape_[0]));
    }
    if (auto var1 = As<ir::Var>(tensor_type->shape_[1])) {
      orig_dim1 = codegen.GetVarName(var1);
    } else {
      orig_dim1 = codegen.GetIndexConstant(codegen.GetConstIntValue(tensor_type->shape_[1]));
    }

    // Emit transposed tensor_view: shape=[dim1, dim0], strides=[1, dim1], layout=DN
    std::string dn_view = codegen.NewTemp();
    std::string c1 = codegen.GetIndexConstant(1);
    tensor_view_type = "!pto.tensor_view<?x?x" + dtype_str + ">";
    std::ostringstream tv_line;
    tv_line << dn_view << " = pto.make_tensor_view " << raw_ptr
            << ", shape = [" << orig_dim1 << ", " << orig_dim0 << "],"
            << " strides = [" << c1 << ", " << orig_dim1 << "]"
            << " {layout = #pto.layout<dn>}"
            << " : " << tensor_view_type;
    codegen.Emit(tv_line.str());
    view_for_partition = dn_view;

    // Swap offsets: user's [dim0_off, dim1_off] → [dim1_off, dim0_off]
    col_off = codegen.GetExprAsCode(offsets_tuple->elements_[0]);
    row_off = codegen.GetExprAsCode(offsets_tuple->elements_[1]);
  } else {
    // Standard ND path
    view_for_partition = codegen.GetOrCreateTensorView(tensor);
    tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());
    row_off = codegen.GetExprAsCode(offsets_tuple->elements_[0]);
    col_off = codegen.GetExprAsCode(offsets_tuple->elements_[1]);

    // Static bounds check: tile dimensions must not exceed tensor dimensions.
    // Catches cases like loading a [64,128] tile from a [128,64] tensor (column overflow).
    INTERNAL_CHECK(tensor_type->shape_.size() == 2)
        << "manual.load ND layout: tensor must be 2D";
    for (size_t d = 0; d < 2; ++d) {
      auto tensor_dim = As<ir::ConstInt>(tensor_type->shape_[d]);
      auto tile_dim = As<ir::ConstInt>(tile_type->shape_[d]);
      if (tensor_dim && tile_dim) {
        CHECK(tile_dim->value_ <= tensor_dim->value_)
            << "manual.load: tile dimension " << d << " (" << tile_dim->value_
            << ") exceeds tensor dimension (" << tensor_dim->value_
            << "). If the tensor needs transposing, use layout=\"dn\".";
      }
    }
  }

  std::string partition_view = codegen.NewTemp();
  std::string partition_type;
  std::ostringstream pv_line;

  bool is_dynamic = codegen.IsDynamicTileType(tile_buf_type);
  if (is_dynamic) {
    // Dynamic tile: get partition_view sizes from tile→valid_shape mapping
    auto [cur_row, cur_col] = codegen.GetTileValidShape(tile_buf);

    partition_type = "!pto.partition_tensor_view<?x?x" + dtype_str + ">";
    pv_line << partition_view << " = pto.partition_view " << view_for_partition
            << ", offsets = [" << row_off << ", " << col_off << "]"
            << ", sizes = ["   << cur_row  << ", " << cur_col << "]"
            << " : " << tensor_view_type << " -> " << partition_type;
    codegen.Emit(pv_line.str());
  } else {
    // Static tile: get partition_view sizes from tile shape constants
    auto cur_row = codegen.GetConstIntValue(tile_type->shape_[0]);
    auto cur_col = codegen.GetConstIntValue(tile_type->shape_[1]);

    partition_type = "!pto.partition_tensor_view<" + std::to_string(cur_row) + "x" +
                                 std::to_string(cur_col) + "x" + dtype_str + ">";
    pv_line << partition_view << " = pto.partition_view " << view_for_partition
          << ", offsets = [" << row_off << ", " << col_off << "]"
          << ", sizes = ["   << codegen.GetIndexConstant(cur_row)  << ", "
          << codegen.GetIndexConstant(cur_col) << "]"
          << " : " << tensor_view_type << " -> " << partition_type;
    codegen.Emit(pv_line.str());
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

  auto output_tensor = As<Var>(op->args_[2]);
  INTERNAL_CHECK(output_tensor) << "manual.store: third argument must be a Var";

  auto tensor_type = As<TensorType>(output_tensor->GetType());
  INTERNAL_CHECK(tensor_type) << "manual.store: third argument must have TensorType";

  auto row_off = codegen.GetExprAsCode(offsets_tuple->elements_[0]);
  auto col_off = codegen.GetExprAsCode(offsets_tuple->elements_[1]);

  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
  std::string tensor_view = codegen.GetOrCreateTensorView(output_tensor);
  std::string tile_buf = codegen.GetVarName(tile);

  std::string tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());

  std::string tile_buf_type = codegen.GetTileBufTypeStringFromTileType(tile_type);

  std::string partition_view = codegen.NewTemp();
  std::string partition_type;
  std::ostringstream pv_line;

  bool is_dynamic = codegen.IsDynamicTileType(tile_buf_type);
  if (is_dynamic) {
    // Dynamic tile: get partition_view sizes from tile→valid_shape mapping
    auto [cur_row, cur_col] = codegen.GetTileValidShape(tile_buf);

    partition_type = "!pto.partition_tensor_view<?x?x" + dtype_str + ">";
    pv_line << partition_view << " = pto.partition_view " << tensor_view
            << ", offsets = [" << row_off << ", " << col_off << "]"
            << ", sizes = ["   << cur_row  << ", " << cur_col << "]"
            << " : " << tensor_view_type << " -> " << partition_type;
    codegen.Emit(pv_line.str());
  } else {
    // Static tile: get partition_view sizes from tile shape constants
    auto cur_row = codegen.GetConstIntValue(tile_type->shape_[0]);
    auto cur_col = codegen.GetConstIntValue(tile_type->shape_[1]);

    partition_type = "!pto.partition_tensor_view<" + std::to_string(cur_row) + "x" +
                                 std::to_string(cur_col) + "x" + dtype_str + ">";
    pv_line << partition_view << " = pto.partition_view " << tensor_view
          << ", offsets = [" << row_off << ", " << col_off << "]"
          << ", sizes = ["   << codegen.GetIndexConstant(cur_row)  << ", "
          << codegen.GetIndexConstant(cur_col) << "]"
          << " : " << tensor_view_type << " -> " << partition_type;
    codegen.Emit(pv_line.str());
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

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.add_relu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualMixBinaryPTO("pto.tadd", "pto.trelu", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.sub_relu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualMixBinaryPTO("pto.tsub", "pto.trelu", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.add_relu_cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualMixBinaryCvtPTO("pto.tadd", "pto.trelu", "pto.tcvt", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.sub_relu_cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualMixBinaryCvtPTO("pto.tsub", "pto.trelu", "pto.tcvt", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.mul_cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryCvtPTO("pto.tmul", "pto.tcvt", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.mul_add_dst")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualMixSelfBinaryPTO("pto.tmul", "pto.tadd", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.fused_mul_add")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualMixFusedBinaryPTO("pto.tmul", "pto.tadd", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.fused_mul_add_relu")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualMixFusedReluBinaryReluPTO("pto.tmul", "pto.tadd", "pto.trelu", op, codegen);
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
      // After parser rewriting for manual ops, manual.reshape reaches backend as:
      //   (src, shape_tuple, out)
      // Emit result-style reshape:
      //   %new = pto.treshape %src : src_type -> dst_type
      // and rebind the explicit output tile variable to %new.
      auto& c = dynamic_cast<codegen::PTOCodegen&>(codegen);
      CHECK(op->args_.size() == 3) << "manual.reshape: expected 3 args (src, shape, out)";

      auto out_var = As<Var>(op->args_[2]);
      CHECK(out_var) << "manual.reshape: out must be a Var";

      std::string src = c.GetExprAsCode(op->args_[0]);
      std::string src_type = c.GetExprTypeAnnotation(op->args_[0]);

      auto out_tile_type = As<ir::TileType>(out_var->GetType());
      CHECK(out_tile_type) << "manual.reshape: out must have TileType";

      std::string out_type = c.GetTileBufTypeStringFromTileType(out_tile_type);
      std::string result = c.NewTemp();

      std::ostringstream oss;
      oss << result << " = pto.treshape " << src;
      if (!src_type.empty()) {
        oss << " : " << src_type;
      }
      if (!out_type.empty()) {
        oss << " -> " << out_type;
      }
      c.Emit(oss.str());

      c.SetVarMlirName(out_var->name_, result);
      c.SetCurrentResultBuf(result);
      return "";
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.transpose")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualUnaryPTO("pto.ttrans", op, codegen);
    });
  
// Gather: (src, indices, out) or (src, indices, tmp, out)
static std::string MakeManualGatherPTO(const std::string& pto_op, const CallPtr& op,
                                       codegen::CodegenBase& cb) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(cb);
  CHECK(op->args_.size() == 2 || op->args_.size() == 4)
      << "manual.gather: expected 3 or 4 args (src, indices, out) or (src, indices, tmp, out), got "
      << op->args_.size();

  if (op->args_.size() == 2) {
    // Index gather without tmp: pto.tgather ins(%src, %indices) outs(%dst)
    codegen.Emit(pto_op + " " + GenerateManualInsOutsClause(op, codegen, 1));
  } else {
    // Index gather with tmp: pto.tgather ins(%src, %indices, %tmp) outs(%dst)
    codegen.Emit(pto_op + " " + GenerateManualInsOutsClause(op, codegen, 3));
  }
  return "";
}

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.gather")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualGatherPTO("pto.tgather", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.gatherb")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeManualBinaryPTO("pto.tgatherb", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "manual.set_validshape")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      auto& c = dynamic_cast<codegen::PTOCodegen&>(codegen);
      CHECK(op->args_.size() == 3)
          << "manual.set_validshape: expected 3 args (row, col, tile), got "
          << op->args_.size();
      auto tile_var = As<Var>(op->args_[2]);
      CHECK(tile_var) << "manual.set_validshape: 3rd arg must be a var";
      auto tile_buf_type = As<ir::TileType>(tile_var->GetType());
      CHECK(tile_buf_type) << "manual.set_validshape: 3rd arg must be a tile";

      std::string row = c.GetExprAsCode(op->args_[0]);
      std::string col = c.GetExprAsCode(op->args_[1]);
      std::string tile_buf = c.GetExprAsCode(op->args_[2]);
      std::string tile_type = c.GetExprTypeAnnotation(op->args_[2]);
      CHECK(c.IsDynamicTileType(tile_type)) << "manual.set_validshape: only dynamic tile can set valid shape";

      std::ostringstream oss;
      oss << "pto.set_validshape " << tile_buf << ", " << row << ", " << col;
      if (!tile_type.empty()) oss << " : " << tile_type;
      c.Emit(oss.str());

      // Update tile→valid_shape mapping so load/store can retrieve it
      c.UpdateTileValidShape(tile_buf, row, col);
      return "";
    });

}  // namespace backend
}  // namespace pypto
