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

#include <algorithm>
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
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace backend {

using ir::As;
using ir::CallPtr;
using ir::PipeType;
using ir::PtrType;
using ir::TileType;
using ir::TensorType;
using ir::Var;

// ============================================================================
// Helper Functions for PTO Code Generation
// ============================================================================

const std::vector<std::string> cmp_modes = {"eq", "ne", "lt", "le", "gt", "ge"};
const std::vector<std::string> round_modes = {"NONE", "RINT",  "ROUND", "FLOOR",
                                              "CEIL", "TRUNC", "ODD",   "CAST_RINT"};

// Helper function for input & output generation (with type annotations)
static std::string GenerateInsOutsClause(const CallPtr& op, codegen::PTOCodegen& codegen,
                                         const std::string& config_attr = "") {
  size_t args_num = op->args_.size();
  std::ostringstream oss;

  // Build ins clause with operand names
  oss << "ins(";
  for (size_t input_idx = 0; input_idx < args_num; ++input_idx) {
    std::string operand = codegen.GetExprAsCode(op->args_[input_idx]);
    if (input_idx == 0) {
      oss << operand;
    } else {
      oss << ", " << operand;
    }
  }

  if (!config_attr.empty()) {
    oss << config_attr;
  }

  // Add type annotations after colon
  std::string type_annot;
  for (size_t input_idx = 0; input_idx < args_num; ++input_idx) {
    std::string annot = codegen.GetExprTypeAnnotation(op->args_[input_idx]);
    if (!annot.empty()) {
      if (!type_annot.empty()) type_annot += ", ";
      type_annot += annot;
    }
  }
  if (!type_annot.empty()) {
    oss << " : " << type_annot;
  }

  // Build outs clause with type annotation
  std::string result_target = codegen.GetCurrentResultTarget();
  std::string result_type = codegen.GetCurrentResultTileBufTypeString();
  oss << ") outs(" << result_target;
  if (!result_type.empty()) {
    oss << " : " << result_type;
  }
  oss << ")";
  return oss.str();
}

// Helper function for N-ary operations (unary, binary, ternary, etc.)
static std::string MakeNaryCodegenPTO(const std::string& pto_op_name, size_t arity, const CallPtr& op,
                                      codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == arity) << "Operation:[" << pto_op_name << "] requires " << arity << " argument"
                                   << (arity != 1 ? "s" : "") << ", but got " << op->args_.size();
  codegen.Emit(pto_op_name + " " + GenerateInsOutsClause(op, codegen));
  return "";
}

// Helper function for StoreFP
static std::string MakeStoreFPCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                         codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "Operation:[" << pto_op_name << "] requires 3 arguments, but got "
                               << op->args_.size();
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string fp = codegen.GetExprAsCode(op->args_[1]);
  std::string mem = codegen.GetExprAsCode(op->args_[2]);
  codegen.Emit(pto_op_name + " ins(" + src + ", " + fp + ") outs(" + mem + ")");
  return "";
}

// Helper function for Binary Tile cmp operations
static std::string MakeTileCmpCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                         codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "Operation:[" << pto_op_name << "] requires 2 arguments, but got "
                               << op->args_.size();
  int mode = op->GetKwarg<int>("mode");
  CHECK(mode >= 0 && mode < static_cast<int>(cmp_modes.size())) << "Tile cmp mode out of range: " << mode;
  std::string config_attr = "{cmpMode = #pto<cmp " + cmp_modes.at(mode) + ">}";
  codegen.Emit(pto_op_name + " " + GenerateInsOutsClause(op, codegen, config_attr));
  return "";
}

// Helper function for Tile cvt operations
static std::string MakeTileCvtCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                         codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "Operation:[" << pto_op_name << "] requires 1 argument, but got "
                               << op->args_.size();
  int mode = op->GetKwarg<int>("mode");
  CHECK(mode >= 0 && mode < static_cast<int>(round_modes.size())) << "Round mode out of range: " << mode;
  std::string config_attr = "{rmode = #pto<round_mode " + round_modes.at(mode) + ">}";
  codegen.Emit(pto_op_name + " " + GenerateInsOutsClause(op, codegen, config_attr));
  return "";
}

// Helper function for full op
static std::string MakeFullCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                      codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "Operation:[" << pto_op_name << "] requires 2 arguments, but got "
                               << op->args_.size();
  std::string scalar = codegen.GetExprAsCode(op->args_[1]);
  std::string scalar_type = codegen.GetExprTypeAnnotation(op->args_[1]);
  std::string dst = codegen.GetCurrentResultTarget();
  std::string dst_type = codegen.GetCurrentResultTileBufTypeString();
  std::ostringstream oss;
  oss << pto_op_name << " ins(" << scalar;
  if (!scalar_type.empty()) oss << " : " << scalar_type;
  oss << ") outs(" << dst;
  if (!dst_type.empty()) oss << " : " << dst_type;
  oss << ")";
  codegen.Emit(oss.str());
  return "";
}

// Helper function for cmps
static std::string MakeCmpsCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                      codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "Operation:[" << pto_op_name << "] requires 2 arguments, but got "
                               << op->args_.size();
  int mode = op->GetKwarg<int>("mode");
  CHECK(mode >= 0 && mode < static_cast<int>(cmp_modes.size())) << "Tile cmp mode out of range: " << mode;
  std::string config_attr = "{cmpMode = #pto<cmp " + cmp_modes.at(mode) + ">}";
  codegen.Emit(pto_op_name + " " + GenerateInsOutsClause(op, codegen, config_attr));
  return "";
}

// Helper function for Assign
static std::string MakeAssignCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                        codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "Operation:[" << pto_op_name << "] requires 2 arguments, but got "
                               << op->args_.size();
  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string addr = codegen.GetExprAsCode(op->args_[1]);
  codegen.Emit(pto_op_name + " ins(" + tile + ", " + addr + ")");
  return "";
}

// Helper function for Ci
static std::string MakeCiCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                    codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "Operation:[" << pto_op_name << "] requires 1 argument, but got "
                               << op->args_.size();
  bool descending = op->GetKwarg<bool>("descending");
  std::string src = codegen.GetExprAsCode(op->args_[0]);
  std::string config_attr = descending ? "{descending = true}" : "{descending = false}";
  std::string dst = codegen.GetCurrentResultTarget();
  codegen.Emit(pto_op_name + " ins(" + src + " " + config_attr + ") outs(" + dst + ")");
  return "";
}

// TODO(guoliwei): Sorting operations typically have multiple outputs, which has not yet been addressed.
// Helper function for Sort32
static std::string MakeSort32CodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                        codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "Operation:[" << pto_op_name << "] requires 1 argument, but got "
                               << op->args_.size();
  // std::string src = codegen.GetExprAsCode(op->args_[0]);
  // std::string dst = codegen.GetCurrentResultTarget();
  codegen.Emit(pto_op_name);
  return "";
}

// TODO(guoliwei): Sorting operations typically have multiple outputs, which has not yet been addressed.
// Helper function for MrgSort
static std::string MakeMrgSortCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                         codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "Operation:[" << pto_op_name << "] requires 2 arguments, but got "
                               << op->args_.size();
  // std::string src = codegen.GetExprAsCode(op->args_[0]);
  // std::string blockLen = codegen.GetExprAsCode(op->args_[1]);
  // std::string dst = codegen.GetCurrentResultTarget();
  codegen.Emit(pto_op_name);
  return "";
}

// Helper function for Print
static std::string MakePrintCodegenPTO(const std::string& pto_op_name, const CallPtr& op,
                                       codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 1 || op->args_.size() == 3)
      << "Operation:[" << pto_op_name << "] requires 1 argument (tile) or 3 arguments "
      << "(tile, offsets, shapes), but got " << op->args_.size();

  auto memory_space_to_mlir = [](ir::MemorySpace space) {
    if (space == ir::MemorySpace::DDR) return std::string("gm");
    if (space == ir::MemorySpace::Vec) return std::string("vec");
    if (space == ir::MemorySpace::Mat) return std::string("mat");
    if (space == ir::MemorySpace::Left) return std::string("left");
    if (space == ir::MemorySpace::Right) return std::string("right");
    if (space == ir::MemorySpace::Acc) return std::string("acc");
    return std::string("vec");
  };
  auto tile_layout_to_str = [](ir::TileLayout layout) {
    if (layout == ir::TileLayout::row_major) return std::string("row_major");
    if (layout == ir::TileLayout::col_major) return std::string("col_major");
    return std::string("none_box");
  };
  auto get_const_or_default = [](const ir::ExprPtr& expr, int64_t default_value) {
    if (auto const_int = As<ir::ConstInt>(expr)) {
      return const_int->value_;
    }
    return default_value;
  };
  auto build_tile_buf_type = [&](const TileType* tile_type, const ir::MakeTuple* shapes_tuple = nullptr,
                                 const ir::MakeTuple* offsets_tuple = nullptr) {
    INTERNAL_CHECK(tile_type) << "tile type must not be null";
    INTERNAL_CHECK(tile_type->shape_.size() == 2) << "block.print tile window currently only supports 2D tiles";
    std::string loc = tile_type->memref_.has_value()
                          ? memory_space_to_mlir(tile_type->memref_.value()->memory_space_)
                          : "vec";
    std::string dtype_str = codegen.GetTypeString(tile_type->dtype_);
    int64_t rows = get_const_or_default(tile_type->shape_[0], 32);
    int64_t cols = get_const_or_default(tile_type->shape_[1], 32);
    int64_t v_row = rows;
    int64_t v_col = cols;
    ir::TileLayout blayout = ir::TileLayout::row_major;
    ir::TileLayout slayout = ir::TileLayout::none_box;
    uint64_t fractal = 512;
    ir::TilePad pad = ir::TilePad::null;
    if (tile_type->tile_view_.has_value()) {
      const auto& tile_view = tile_type->tile_view_.value();
      if (tile_view.valid_shape.size() == 2) {
        v_row = get_const_or_default(tile_view.valid_shape[0], v_row);
        v_col = get_const_or_default(tile_view.valid_shape[1], v_col);
      }
      blayout = tile_view.blayout;
      slayout = tile_view.slayout;
      fractal = tile_view.fractal;
      pad = tile_view.pad;
    }
    if (shapes_tuple != nullptr) {
      rows = get_const_or_default(shapes_tuple->elements_[0], rows);
      cols = get_const_or_default(shapes_tuple->elements_[1], cols);
      v_row = rows;
      v_col = cols;
      if (offsets_tuple != nullptr && tile_type->tile_view_.has_value() &&
          tile_type->tile_view_->valid_shape.size() == 2) {
        int64_t src_valid_row = get_const_or_default(tile_type->tile_view_->valid_shape[0], rows);
        int64_t src_valid_col = get_const_or_default(tile_type->tile_view_->valid_shape[1], cols);
        int64_t row_off = get_const_or_default(offsets_tuple->elements_[0], 0);
        int64_t col_off = get_const_or_default(offsets_tuple->elements_[1], 0);
        v_row = std::max<int64_t>(0, std::min<int64_t>(rows, src_valid_row - row_off));
        v_col = std::max<int64_t>(0, std::min<int64_t>(cols, src_valid_col - col_off));
      }
    }
    std::ostringstream oss;
    oss << "!pto.tile_buf<loc=" << loc << ", dtype=" << dtype_str;
    oss << ", rows=" << rows << ", cols=" << cols;
    oss << ", v_row=" << v_row << ", v_col=" << v_col;
    oss << ", blayout=" << tile_layout_to_str(blayout);
    oss << ", slayout=" << tile_layout_to_str(slayout);
    oss << ", fractal=" << fractal << ", pad=" << static_cast<int>(pad) << ">";
    return oss.str();
  };

  std::string src = codegen.GetExprAsCode(op->args_[0]);
  if (op->args_.size() == 3) {
    auto tile_type = As<TileType>(op->args_[0]->GetType());
    INTERNAL_CHECK(tile_type) << "block.print first argument must have TileType";
    auto offsets_tuple = As<ir::MakeTuple>(op->args_[1]);
    INTERNAL_CHECK(offsets_tuple) << "block.print second argument must be a tuple (offsets)";
    auto shapes_tuple = As<ir::MakeTuple>(op->args_[2]);
    INTERNAL_CHECK(shapes_tuple) << "block.print third argument must be a tuple (shapes)";

    std::string src_type = codegen.GetExprTypeAnnotation(op->args_[0]);
    if (src_type.empty()) {
      src_type = build_tile_buf_type(tile_type.get());
    }
    std::string subset_type = build_tile_buf_type(tile_type.get(), shapes_tuple.get(), offsets_tuple.get());
    std::string subset = codegen.NewTemp();
    std::ostringstream subset_line;
    subset_line << subset << " = pto.subset " << src << "[";
    for (size_t i = 0; i < offsets_tuple->elements_.size(); ++i) {
      if (i > 0) subset_line << ", ";
      subset_line << codegen.GetExprAsCode(offsets_tuple->elements_[i]);
    }
    subset_line << "] sizes [";
    for (size_t i = 0; i < shapes_tuple->elements_.size(); ++i) {
      if (i > 0) subset_line << ", ";
      auto dim = As<ir::ConstInt>(shapes_tuple->elements_[i]);
      INTERNAL_CHECK(dim) << "block.print shape must be static ConstInt at axis " << i;
      subset_line << dim->value_;
    }
    subset_line << "] : " << src_type;
    codegen.Emit(subset_line.str());
    codegen.Emit(pto_op_name + " ins(" + subset + " : " + subset_type + ")");
    return "";
  }

  std::string src_type = codegen.GetExprTypeAnnotation(op->args_[0]);
  if (!src_type.empty()) {
    codegen.Emit(pto_op_name + " ins(" + src + " : " + src_type + ")");
  } else if (auto tile_type = As<TileType>(op->args_[0]->GetType()); tile_type &&
             tile_type->shape_.size() == 2) {
    codegen.Emit(pto_op_name + " ins(" + src + " : " + build_tile_buf_type(tile_type.get()) + ")");
  } else {
    codegen.Emit(pto_op_name + " ins(" + src + ")");
  }
  return "";
}

static std::string GetStaticPartitionType(const ir::MakeTuple* shapes_tuple, const std::string& dtype_str) {
  std::ostringstream oss;
  oss << "!pto.partition_tensor_view<";
  for (size_t i = 0; i < shapes_tuple->elements_.size(); ++i) {
    if (i > 0) oss << "x";
    auto dim = As<ir::ConstInt>(shapes_tuple->elements_[i]);
    INTERNAL_CHECK(dim) << "partition shape must be static ConstInt at axis " << i;
    oss << dim->value_;
  }
  oss << "x" << dtype_str << ">";
  return oss.str();
}

static std::string MakeTensorPrintCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "tensor.print requires 3 arguments, but got " << op->args_.size();

  auto tensor = As<Var>(op->args_[0]);
  INTERNAL_CHECK(tensor) << "tensor.print first argument must be a Var";

  auto offsets_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK(offsets_tuple) << "tensor.print second argument must be a tuple (offsets)";

  auto shapes_tuple = As<ir::MakeTuple>(op->args_[2]);
  INTERNAL_CHECK(shapes_tuple) << "tensor.print third argument must be a tuple (shapes)";

  auto tensor_type = As<TensorType>(tensor->GetType());
  INTERNAL_CHECK(tensor_type) << "tensor.print tensor argument must have TensorType";

  std::string tensor_view = codegen.GetOrCreateTensorView(tensor);
  std::string tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());
  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
  std::string partition_type = GetStaticPartitionType(shapes_tuple.get(), dtype_str);

  std::string partition_view = codegen.NewTemp();
  std::ostringstream partition_line;
  partition_line << partition_view << " = pto.partition_view " << tensor_view;
  partition_line << ", offsets = [";
  for (size_t i = 0; i < offsets_tuple->elements_.size(); ++i) {
    if (i > 0) partition_line << ", ";
    partition_line << codegen.GetExprAsCode(offsets_tuple->elements_[i]);
  }
  partition_line << "], sizes = [";
  for (size_t i = 0; i < shapes_tuple->elements_.size(); ++i) {
    if (i > 0) partition_line << ", ";
    partition_line << codegen.GetExprAsCode(shapes_tuple->elements_[i]);
  }
  partition_line << "] : " << tensor_view_type << " -> " << partition_type;
  codegen.Emit(partition_line.str());

  codegen.Emit("pto.tprint ins(" + partition_view + " : " + partition_type + ")");
  return "";
}

// block.load: emit pto.subview + pto.tload (same format as original IR layer codegen)
static std::string MakeBlockLoadCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  auto tensor = As<Var>(op->args_[0]);
  INTERNAL_CHECK(tensor) << "block.load first argument must be a Var";

  // Extract offsets tuple
  auto offsets_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK(offsets_tuple) << "block.load second argument must be a tuple (offsets)";

  // Extract shapes tuple
  auto shapes_tuple = As<ir::MakeTuple>(op->args_[2]);
  INTERNAL_CHECK(shapes_tuple) << "block.load third argument must be a tuple (shapes)";

  // Extract 2D offset and size values from tuples
  auto row_off = codegen.GetExprAsCode(offsets_tuple->elements_[0]);
  auto col_off = codegen.GetExprAsCode(offsets_tuple->elements_[1]);
  int64_t height = codegen.GetConstIntValue(shapes_tuple->elements_[0]);
  int64_t width = codegen.GetConstIntValue(shapes_tuple->elements_[1]);

  auto tensor_type = As<TensorType>(tensor->GetType());
  INTERNAL_CHECK(tensor_type) << "block.load tensor argument must have TensorType";

  std::string tensor_view = codegen.GetOrCreateTensorView(tensor);
  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
  std::string tile_buf = codegen.GetCurrentResultTarget();
  INTERNAL_CHECK(!tile_buf.empty()) << "block.load requires assignment target (tile_buf)";

  std::string tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());
  std::string tile_buf_type = codegen.GetCurrentResultTileBufTypeString();
  std::string partition_type = "!pto.partition_tensor_view<" + std::to_string(height) + "x" +
                               std::to_string(width) + "x" + dtype_str + ">";

  std::string partition_view = codegen.NewTemp();
  std::ostringstream partition_line;
  partition_line << partition_view << " = pto.partition_view " << tensor_view;
  partition_line << ", offsets = [" << row_off << ", " << col_off << "]";
  partition_line << ", sizes = [" << codegen.GetIndexConstant(height) << ", ";
  partition_line << codegen.GetIndexConstant(width) << "]";
  partition_line << " : " << tensor_view_type << " -> " << partition_type;
  codegen.Emit(partition_line.str());

  std::ostringstream tload_line;
  tload_line << "pto.tload ins(" << partition_view << " : " << partition_type << ") outs(";
  tload_line << tile_buf << " : " << tile_buf_type << ")";
  codegen.Emit(tload_line.str());
  return "";  // Multi-line emission
}

// block.store: emit pto.partition_view + pto.tstore
static std::string MakeBlockStoreCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  auto tile = As<Var>(op->args_[0]);
  INTERNAL_CHECK(tile) << "block.store first argument must be a Var";

  // Extract offsets tuple
  auto offsets_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK(offsets_tuple) << "block.store second argument must be a tuple (offsets)";

  // Extract shapes tuple
  auto shapes_tuple = As<ir::MakeTuple>(op->args_[2]);
  INTERNAL_CHECK(shapes_tuple) << "block.store third argument must be a tuple (shapes)";

  // Extract 2D offset and size values from tuples
  auto row_off = codegen.GetExprAsCode(offsets_tuple->elements_[0]);
  auto col_off = codegen.GetExprAsCode(offsets_tuple->elements_[1]);
  int64_t height = codegen.GetConstIntValue(shapes_tuple->elements_[0]);
  int64_t width = codegen.GetConstIntValue(shapes_tuple->elements_[1]);
  auto output_tensor = As<Var>(op->args_[3]);
  INTERNAL_CHECK(output_tensor) << "block.store output_tensor must be a Var";

  auto tensor_type = As<TensorType>(output_tensor->GetType());
  INTERNAL_CHECK(tensor_type) << "block.store output_tensor must have TensorType";

  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
  std::string tensor_view = codegen.GetOrCreateTensorView(output_tensor);
  std::string tile_buf = codegen.GetVarName(tile);

  std::string tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());
  std::string partition_type = "!pto.partition_tensor_view<" + std::to_string(height) + "x" +
                               std::to_string(width) + "x" + dtype_str + ">";

  // Get tile_buf type via GetExprTypeAnnotation which correctly handles
  // dynamically-allocated buffers (e.g., reshape outputs in extra_tile_buf_types_)
  std::string tile_buf_type = codegen.GetExprTypeAnnotation(op->args_[0]);

  std::string partition_view = codegen.NewTemp();
  std::ostringstream partition_line;
  partition_line << partition_view << " = pto.partition_view " << tensor_view;
  partition_line << ", offsets = [" << row_off << ", " << col_off << "]";
  partition_line << ", sizes = [" << codegen.GetIndexConstant(height) << ", ";
  partition_line << codegen.GetIndexConstant(width) << "]";
  partition_line << " : " << tensor_view_type << " -> " << partition_type;
  codegen.Emit(partition_line.str());

  std::ostringstream tstore_line;
  tstore_line << "pto.tstore ins(" << tile_buf;
  if (!tile_buf_type.empty()) {
    tstore_line << " : " << tile_buf_type;
  }
  tstore_line << ") outs(" << partition_view << " : " << partition_type << ")";
  codegen.Emit(tstore_line.str());
  return "";
}

// Helper function for block.alloc (no-op: allocation handled elsewhere)
static std::string MakeBlockAllocCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  (void)op;
  (void)codegen_base;
  return "";  // No MLIR emission - pto.alloc_tile generated from MemRefs in TileTypes
}

static std::string MakeTensorDimCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "tensor.dim requires 2 arguments, but got " << op->args_.size();
  auto input_tensor = ir::As<ir::TensorType>(op->args_[0]->GetType());
  CHECK(input_tensor) << "tensor.dim need TensorType for first arg, but got "
                      << op->args_[0]->GetType()->TypeName();
  auto axis = codegen.GetConstIntValue(op->args_[1]);
  CHECK(axis >= 0 && static_cast<size_t>(axis) < input_tensor->shape_.size())
      << "tensor.dim axis " << axis << " out of range for tensor with rank " << input_tensor->shape_.size();
  auto shape = input_tensor->shape_[axis];
  std::string shape_name;
  // dynamic shape
  if (auto dyn_shape = ir::As<ir::Var>(shape)) {
    shape_name = codegen.GetVarName(dyn_shape);
  } else if (auto static_shape = ir::As<ir::ConstInt>(shape)) {  // constant shape
    shape_name = codegen.GetIndexConstant(static_shape->value_);
  } else {
    INTERNAL_CHECK(false) << "Internal error: tensor.dim shape is neither Var nor ConstInt";
  }
  // register target var to shape name so later uses (e.g., pl.range(M)) resolve correctly
  auto target_var_name = codegen.GetCurrentResultTarget();
  if (!target_var_name.empty() && !shape_name.empty()) {
    codegen.RegisterVarToMlir(target_var_name, shape_name);
  }

  return "";
}

// ============================================================================
// Table-driven registration for simple N-ary operations
// ============================================================================

struct SimpleOpEntry {
  const char* op_name;
  const char* pto_op_name;
  size_t arity;
  PipeType pipe = PipeType::V;
};

// clang-format off
static const SimpleOpEntry kSimpleOps[] = {
    // Tile utility operations
    {"block.getval",          "pto.tgetval",          2},
    {"block.setval",          "pto.tsetval",          2},
    // Memory operations
    {"block.mgather",         "pto.tmgather",         2},
    {"block.mscatter",        "pto.tmscatter",        2},
    // Tile x Tile arithmetic operations
    {"block.add",             "pto.tadd",             2},
    {"block.sub",             "pto.tsub",             2},
    {"block.mul",             "pto.tmul",             2},
    {"block.div",             "pto.tdiv",             2},
    {"block.rem",             "pto.trem",             2},
    // Tile x Tile bitwise operations
    {"block.and",             "pto.tand",             2},
    {"block.or",              "pto.tor",              2},
    {"block.xor",             "pto.txor",             2},
    {"block.shl",             "pto.tshl",             2},
    {"block.shr",             "pto.tshr",             2},
    // Tile x Tile comparison/selection operations
    {"block.maximum",         "pto.tmax",             2},
    {"block.minimum",         "pto.tmin",             2},
    {"block.prelu",           "pto.tprelu",           2},
    // Unary operations
    {"block.abs",             "pto.tabs",             1},
    {"block.exp",             "pto.texp",             1},
    {"block.log",             "pto.tlog",             1},
    {"block.sqrt",            "pto.tsqrt",            1},
    {"block.rsqrt",           "pto.trsqrt",           1},
    {"block.recip",           "pto.trecip",           1},
    {"block.neg",             "pto.tneg",             1},
    {"block.not",             "pto.tnot",             1},
    {"block.relu",            "pto.trelu",            1},
    // Ternary operations (tile x tile + carry/select)
    {"block.addc",            "pto.taddc",            3},
    {"block.subc",            "pto.tsubc",            3},
    {"block.sel",             "pto.tsel",             3},
    // Tile x Scalar operations
    {"block.adds",            "pto.tadds",            2},
    {"block.subs",            "pto.tsubs",            2},
    {"block.muls",            "pto.tmuls",            2},
    {"block.divs",            "pto.tdivs",            2},
    {"block.rems",            "pto.trems",            2},
    {"block.ands",            "pto.tands",            2},
    {"block.ors",             "pto.tors",             2},
    {"block.xors",            "pto.txors",            2},
    {"block.shls",            "pto.tshls",            2},
    {"block.shrs",            "pto.tshrs",            2},
    {"block.maxs",            "pto.tmaxs",            2},
    {"block.mins",            "pto.tmins",            2},
    {"block.lrelu",           "pto.tlrelu",           2},
    // Ternary scalar operations (tile x scalar + carry/select)
    {"block.addsc",           "pto.taddsc",           3},
    {"block.subsc",           "pto.tsubsc",           3},
    {"block.selc",            "pto.tselc",            3},
    // Axis reduction/expansion operations
    {"block.row_sum",         "pto.trowsum",          2},
    {"block.row_max",         "pto.trowmax",          2},
    {"block.row_min",         "pto.trowmin",          2},
    {"block.row_expand",      "pto.trowexpand",       1},
    {"block.col_sum",         "pto.tcolsum",          1},
    {"block.col_max",         "pto.tcolmax",          1},
    {"block.col_min",         "pto.tcolmin",          1},
    {"block.col_expand",      "pto.tcolexpand",       2},
    {"block.row_expand_div",  "pto.trowexpanddiv",    2},
    {"block.row_expand_mul",  "pto.trowexpandmul",    2},
    {"block.row_expand_sub",  "pto.trowexpandsub",    2},
    // Padding operations
    {"block.fillpad",         "pto.tfillpad",         1},
    // Matrix multiplication operations (PipeType::M → CUBE/AIC core)
    {"block.matmul",          "pto.tmatmul",          2, PipeType::M},
    {"block.matmul_mx",       "pto.tmatmul.mx",       4, PipeType::M},
    {"block.matmul_mx_acc",   "pto.tmatmul.mx.acc",   5, PipeType::M},
    {"block.matmul_mx_bias",  "pto.tmatmul.mx.bias",  5, PipeType::M},
    {"block.matmul_acc",      "pto.tmatmul.acc",      3, PipeType::M},
    {"block.matmul_bias",     "pto.tmatmul.bias",     3, PipeType::M},
    {"block.gemv",            "pto.tgemv",            2, PipeType::M},
    {"block.gemv_acc",        "pto.tgemv.acc",        3, PipeType::M},
    {"block.gemv_bias",       "pto.tgemv.bias",       3, PipeType::M},
    // Data movement/layout operations (PipeType::MTE1 → memory transfer, not V/M)
    {"block.move",            "pto.tmov",             1, PipeType::MTE1},
    {"block.move_fp",         "pto.tmov.fp",          2, PipeType::MTE1},
    {"block.transpose",       "pto.ttrans",           3},
    {"block.extract",         "pto.textract",         3},
    // Gather/scatter operations
    {"block.gather",          "pto.tgather",          2},
    {"block.gatherb",         "pto.tgatherb",         2},
    {"block.scatter",         "pto.tscatter",         2},
    // Partial reduction operations
    {"block.partadd",         "pto.tpartadd",         2},
    {"block.partmax",         "pto.tpartmax",         2},
    {"block.partmin",         "pto.tpartmin",         2},
};
// clang-format on

static void RegisterSimpleOps() {
  for (const auto& entry : kSimpleOps) {
    std::string pto_op = entry.pto_op_name;
    size_t arity = entry.arity;
    Backend910B_PTO::Instance()
        .RegisterOp(entry.op_name)
        .set_pipe(entry.pipe)
        .f_codegen([pto_op, arity](const CallPtr& op, codegen::CodegenBase& codegen) {
          return MakeNaryCodegenPTO(pto_op, arity, op, codegen);
        });
  }
}

static const bool kSimpleOpsRegistered = [] {
  RegisterSimpleOps();
  return true;
}();

// ============================================================================
// Operations with custom codegen logic
// ============================================================================

REGISTER_BACKEND_OP(Backend910B_PTO, "block.load")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockLoadCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.store")
    .set_pipe(ir::PipeType::MTE3)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockStoreCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.l0c_store")
    .set_pipe(ir::PipeType::MTE3)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockStoreCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.alloc")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockAllocCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.make_tile")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      (void)op;
      (void)codegen_base;
      return std::string("");  // No MLIR emission - tile allocation handled by pto.alloc_tile
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.store_fp")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeStoreFPCodegenPTO("pto.tstore.fp", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.cmp")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileCmpCodegenPTO("pto.tcmp", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTileCvtCodegenPTO("pto.tcvt", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.full")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeFullCodegenPTO("pto.texpands", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.cmps")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeCmpsCodegenPTO("pto.tcmps", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.assign")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeAssignCodegenPTO("pto.tassign", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.ci")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeCiCodegenPTO("pto.tci", op, codegen);
    });

// TODO(guoliwei): Sorting operations typically have multiple outputs, which has not yet been addressed.
REGISTER_BACKEND_OP(Backend910B_PTO, "block.sort32")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeSort32CodegenPTO("pto.tsort32", op, codegen);
    });

// TODO(guoliwei): Sorting operations typically have multiple outputs, which has not yet been addressed.
REGISTER_BACKEND_OP(Backend910B_PTO, "block.mrgsort")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeMrgSortCodegenPTO("pto.tmrgsort", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.print")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakePrintCodegenPTO("pto.tprint", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "tensor.dim")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTensorDimCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "tensor.print")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTensorPrintCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.reshape")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
      auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
      CHECK(op->args_.size() == 2) << "Operation:[block.reshape] requires 2 arguments (tile, shape), but got "
                                   << op->args_.size();
      // Only use the first argument (source tile); shape tuple is metadata
      std::string src = codegen.GetExprAsCode(op->args_[0]);
      std::string result_target = codegen.GetCurrentResultTarget();
      std::string result_type = codegen.GetCurrentResultTileBufTypeString();
      // Get the correct input type directly from the source variable's TileType,
      // bypassing the memref_to_tile_type_ lookup which may return the wrong shape
      // when input and output share the same MemRef.
      std::string src_type;
      if (auto src_var = ir::As<Var>(op->args_[0])) {
        if (auto tile_type = ir::As<ir::TileType>(src_var->GetType())) {
          if (tile_type->memref_.has_value()) {
            src_type = codegen.GetTileBufTypeStringFromTileType(tile_type);
          }
        }
      }
      // PTO bytecode requires distinct tile buffers for reshape input/output.
      // When both resolve to the same buffer (shared MemRef), allocate a new output buffer.
      if (src == result_target && !result_type.empty()) {
        result_target = codegen.AllocNewTileBuf(result_type);
        codegen.SetCurrentResultBuf(result_target);
      }
      std::ostringstream oss;
      oss << "pto.treshape ins(" << src;
      if (!src_type.empty()) oss << " : " << src_type;
      oss << ") outs(" << result_target;
      if (!result_type.empty()) oss << " : " << result_type;
      oss << ")";
      codegen.Emit(oss.str());
      return std::string("");
    });

// Helper function for block.get_block_idx
static std::string MakeBlockGetBlockIdxCodegenPTO(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 0) << "block.get_block_idx requires no arguments";

  // Create a new SSA variable for the scalar result
  std::string result = codegen.NewTemp();
  codegen.Emit(result + " = pto.get_block_idx");

  // Register the result variable mapping
  codegen.SetVarMlirName(codegen.GetCurrentResultVarName(), result);

  return "";
}

REGISTER_BACKEND_OP(Backend910B_PTO, "block.get_block_idx")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockGetBlockIdxCodegenPTO(op, codegen);
    });

// Helper function for block.get_subblock_idx
static std::string MakeBlockGetSubblockIdxIdxCodegenPTO(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 0) << "block.get_subblock_idx requires no arguments";

  // Create a new SSA variable for the scalar result
  std::string result = codegen.NewTemp();
  codegen.Emit(result + " = pto.get_subblock_idx");

  // Register the result variable mapping
  codegen.SetVarMlirName(codegen.GetCurrentResultVarName(), result);
  return "";
}

REGISTER_BACKEND_OP(Backend910B_PTO, "block.get_subblock_idx")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockGetSubblockIdxIdxCodegenPTO(op, codegen);
    });

// Helper function for block.get_block_num
static std::string MakeBlockGetBlockNumCodegenPTO(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 0) << "block.get_block_num requires no arguments";

  // Create a new SSA variable for the scalar result
  std::string result = codegen.NewTemp();
  codegen.Emit(result + " = pto.get_block_num");

  // Register the result variable mapping
  codegen.SetVarMlirName(codegen.GetCurrentResultVarName(), result);

  return "";
}

REGISTER_BACKEND_OP(Backend910B_PTO, "block.get_block_num")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockGetBlockNumCodegenPTO(op, codegen);
    });

// Helper function for block.index_cast
static std::string MakeBlockIndexCastCodegenPTO(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 1) << "block.index_cast requires 1 argument";

  std::string idx = codegen.GetExprAsCode(op->args_[0]);

  // Create a new SSA variable for the result
  std::string result = codegen.NewTemp();

  // Emit arith.index_cast operation
  codegen.Emit(result + " = arith.index_cast " + idx + " : i64 to index");

  // Register result variable mapping
  codegen.SetVarMlirName(codegen.GetCurrentResultVarName(), result);

  return "";
}

REGISTER_BACKEND_OP(Backend910B_PTO, "block.index_cast")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockIndexCastCodegenPTO(op, codegen);
    });

// ptr.make_tensor: emit pto.make_tensor_view in function body
static std::string MakePtrMakeTensorCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  std::string ptr_name = codegen.GetExprAsCode(op->args_[0]);
  auto result_type = As<TensorType>(op->GetType());
  INTERNAL_CHECK(result_type) << "ptr.make_tensor result must be TensorType";
  auto shape_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK(shape_tuple) << "ptr.make_tensor shape must be MakeTuple";
  auto stride_tuple = As<ir::MakeTuple>(op->args_[2]);
  INTERNAL_CHECK(stride_tuple) << "ptr.make_tensor stride must be MakeTuple";
  std::string view_name = codegen.NewTemp();
  codegen.SetTensorViewName(codegen.GetCurrentResultVarName(), view_name);
  std::ostringstream oss;
  oss << view_name << " = pto.make_tensor_view " << ptr_name << ", shape = [";
  for (size_t j = 0; j < shape_tuple->elements_.size(); j++) {
    if (j > 0) oss << ", ";
    oss << codegen.GetExprAsCode(shape_tuple->elements_[j]);
  }
  oss << "], strides = [";
  for (size_t j = 0; j < stride_tuple->elements_.size(); j++) {
    if (j > 0) oss << ", ";
    oss << codegen.GetExprAsCode(stride_tuple->elements_[j]);
  }
  oss << "] : " << codegen.GetTensorViewTypeString(result_type.get());
  codegen.Emit(oss.str());
  return "";
}

// ptr.addptr: emit pto.addptr
static std::string MakePtrAddPtrCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  std::string ptr_name = codegen.GetExprAsCode(op->args_[0]);
  std::string offset_name = codegen.GetExprAsCode(op->args_[1]);
  std::string result_name = codegen.NewTemp();
  auto result_type = As<PtrType>(op->GetType());
  INTERNAL_CHECK(result_type) << "ptr.addptr result must be PtrType";
  codegen.SetVarMlirName(codegen.GetCurrentResultVarName(), result_name);
  std::string ptr_type_str = "!pto.ptr<" + codegen.GetTypeString(result_type->dtype_) + ">";
  codegen.Emit(result_name + " = pto.addptr " + ptr_name + ", " + offset_name + " : " +
               ptr_type_str + " -> " + ptr_type_str);
  return "";
}

REGISTER_BACKEND_OP(Backend910B_PTO, "ptr.make_tensor")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakePtrMakeTensorCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "ptr.addptr")
    .set_pipe(ir::PipeType::MTE2)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakePtrAddPtrCodegenPTO(op, codegen);
    });
// System synchronization operations
static std::string GetPipeTypeName(ir::PipeType pipe) {
  switch (pipe) {
    case ir::PipeType::MTE1: return "MTE1";
    case ir::PipeType::MTE2: return "MTE2";
    case ir::PipeType::MTE3: return "MTE3";
    case ir::PipeType::M: return "M";
    case ir::PipeType::V: return "V";
    case ir::PipeType::S: return "S";
    case ir::PipeType::FIX: return "FIX";
    case ir::PipeType::ALL: return "ALL";
    default: return "UNKNOWN";
  }
}

static std::string MakeSyncSrcCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  auto set_pipe = op->GetKwarg<int>("set_pipe");
  auto wait_pipe = op->GetKwarg<int>("wait_pipe");
  auto event_id = op->GetKwarg<int>("event_id");
  std::ostringstream oss;
  oss << "pto.set_flag[<PIPE_" << GetPipeTypeName(static_cast<ir::PipeType>(set_pipe))
      << ">, <PIPE_" << GetPipeTypeName(static_cast<ir::PipeType>(wait_pipe))
      << ">, <EVENT_ID" << event_id << ">]";
  codegen.Emit(oss.str());
  return "";
}

static std::string MakeSyncDstCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  auto set_pipe = op->GetKwarg<int>("set_pipe");
  auto wait_pipe = op->GetKwarg<int>("wait_pipe");
  auto event_id = op->GetKwarg<int>("event_id");
  std::ostringstream oss;
  oss << "pto.wait_flag[<PIPE_" << GetPipeTypeName(static_cast<ir::PipeType>(set_pipe))
      << ">, <PIPE_" << GetPipeTypeName(static_cast<ir::PipeType>(wait_pipe))
      << ">, <EVENT_ID" << event_id << ">]";
  codegen.Emit(oss.str());
  return "";
}

REGISTER_BACKEND_OP(Backend910B_PTO, "system.sync_src")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeSyncSrcCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "system.sync_dst")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeSyncDstCodegenPTO(op, codegen);
    });
// Barrier operations
static std::string MakeBarVCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  codegen.Emit("pto.barrier #pto.pipe<PIPE_V>");
  return "";
}

static std::string MakeBarMCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  codegen.Emit("pto.barrier #pto.pipe<PIPE_M>");
  return "";
}

static std::string MakeBarAllCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  codegen.Emit("pto.barrier #pto.pipe<PIPE_ALL>");
  return "";
}

REGISTER_BACKEND_OP(Backend910B_PTO, "system.bar_v")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBarVCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "system.bar_m")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBarMCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "system.bar_all")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBarAllCodegenPTO(op, codegen);
    });

static std::string MakeSetCrossCoreCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  auto pipe = op->GetKwarg<int>("pipe");
  auto event_id = op->GetKwarg<int>("event_id");
  std::ostringstream oss;
  oss << "pto.sync.set #pto.pipe<PIPE_" << GetPipeTypeName(static_cast<ir::PipeType>(pipe))
      << ">, " << event_id;
  codegen.Emit(oss.str());
  return "";
}

REGISTER_BACKEND_OP(Backend910B_PTO, "system.set_cross_core")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeSetCrossCoreCodegenPTO(op, codegen);
    });

static std::string MakeWaitCrossCoreCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  auto pipe = op->GetKwarg<int>("pipe");
  auto event_id = op->GetKwarg<int>("event_id");
  std::ostringstream oss;
  oss << "pto.sync.wait #pto.pipe<PIPE_" << GetPipeTypeName(static_cast<ir::PipeType>(pipe))
      << ">, " << event_id;
  codegen.Emit(oss.str());
  return "";
}

REGISTER_BACKEND_OP(Backend910B_PTO, "system.wait_cross_core")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeWaitCrossCoreCodegenPTO(op, codegen);
    });
}  // namespace backend
}  // namespace pypto
