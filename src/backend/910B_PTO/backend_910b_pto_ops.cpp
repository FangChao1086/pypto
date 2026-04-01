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
#include <cctype>
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

// CVSyncEvent
enum CVSyncEvent : uint16_t {
    SYNC_AIC_FLAG = 11,
    SYNC_AIV_FLAG = 12,
    SYNC_AIC_AIV_FLAG = 13,
    SYNC_AIV_ONLY_ALL = 14,
    SYNC_FLAG_ID_MAX = 16,
};

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
  auto is_dynamic_dim = [](const ir::ExprPtr& expr) {
    if (auto const_int = As<ir::ConstInt>(expr)) {
      return const_int->value_ < 0;
    }
    return true;
  };
  auto build_tile_buf_type = [&](const TileType* tile_type, const ir::MakeTuple* shapes_tuple = nullptr,
                                 const ir::MakeTuple* offsets_tuple = nullptr) {
    INTERNAL_CHECK(tile_type) << "tile type must not be null";
    INTERNAL_CHECK(tile_type->shape_.size() == 2) << "debug.dump_tile window currently only supports 2D tiles";
    std::string loc = tile_type->memref_.has_value()
                          ? memory_space_to_mlir(tile_type->memref_.value()->memory_space_)
                          : "vec";
    std::string dtype_str = codegen.GetTypeString(tile_type->dtype_);
    int64_t rows = get_const_or_default(tile_type->shape_[0], 32);
    int64_t cols = get_const_or_default(tile_type->shape_[1], 32);
    int64_t v_row = rows;
    int64_t v_col = cols;
    bool v_row_dynamic = false;
    bool v_col_dynamic = false;
    ir::TileLayout blayout = ir::TileLayout::row_major;
    ir::TileLayout slayout = ir::TileLayout::none_box;
    uint64_t fractal = 512;
    ir::TilePad pad = ir::TilePad::null;
    ir::CompactMode compact = ir::CompactMode::null;
    if (tile_type->tile_view_.has_value()) {
      const auto& tile_view = tile_type->tile_view_.value();
      if (tile_view.valid_shape.size() == 2) {
        if (is_dynamic_dim(tile_view.valid_shape[0])) {
          v_row_dynamic = true;
        } else {
          v_row = get_const_or_default(tile_view.valid_shape[0], v_row);
        }
        if (is_dynamic_dim(tile_view.valid_shape[1])) {
          v_col_dynamic = true;
        } else {
          v_col = get_const_or_default(tile_view.valid_shape[1], v_col);
        }
      }
      blayout = tile_view.blayout;
      slayout = tile_view.slayout;
      fractal = tile_view.fractal;
      pad = tile_view.pad;
      compact = tile_view.compact;
    }
    if (shapes_tuple != nullptr) {
      rows = get_const_or_default(shapes_tuple->elements_[0], rows);
      cols = get_const_or_default(shapes_tuple->elements_[1], cols);
      v_row = rows;
      v_col = cols;
      v_row_dynamic = false;
      v_col_dynamic = false;
      if (offsets_tuple != nullptr && tile_type->tile_view_.has_value() &&
          tile_type->tile_view_->valid_shape.size() == 2) {
        auto src_valid_row = As<ir::ConstInt>(tile_type->tile_view_->valid_shape[0]);
        auto src_valid_col = As<ir::ConstInt>(tile_type->tile_view_->valid_shape[1]);
        auto row_off = As<ir::ConstInt>(offsets_tuple->elements_[0]);
        auto col_off = As<ir::ConstInt>(offsets_tuple->elements_[1]);
        if (src_valid_row && src_valid_row->value_ >= 0 && row_off) {
          v_row = std::max<int64_t>(0, std::min<int64_t>(rows, src_valid_row->value_ - row_off->value_));
        } else {
          v_row_dynamic = true;
        }
        if (src_valid_col && src_valid_col->value_ >= 0 && col_off) {
          v_col = std::max<int64_t>(0, std::min<int64_t>(cols, src_valid_col->value_ - col_off->value_));
        } else {
          v_col_dynamic = true;
        }
      }
    }
    std::ostringstream oss;
    oss << "!pto.tile_buf<loc=" << loc << ", dtype=" << dtype_str;
    oss << ", rows=" << rows << ", cols=" << cols;
    oss << ", v_row=" << (v_row_dynamic ? "?" : std::to_string(v_row));
    oss << ", v_col=" << (v_col_dynamic ? "?" : std::to_string(v_col));
    oss << ", blayout=" << tile_layout_to_str(blayout);
    oss << ", slayout=" << tile_layout_to_str(slayout);
    oss << ", fractal=" << fractal << ", pad=" << static_cast<int>(pad);
    if (compact != ir::CompactMode::null) {
      oss << ", compact=" << static_cast<int>(compact);
    }
    oss << ">";
    return oss.str();
  };

  std::string src = codegen.GetExprAsCode(op->args_[0]);
  if (op->args_.size() == 3) {
    auto tile_type = As<TileType>(op->args_[0]->GetType());
    INTERNAL_CHECK(tile_type) << "debug.dump_tile first argument must have TileType";
    auto offsets_tuple = As<ir::MakeTuple>(op->args_[1]);
    INTERNAL_CHECK(offsets_tuple) << "debug.dump_tile second argument must be a tuple (offsets)";
    auto shapes_tuple = As<ir::MakeTuple>(op->args_[2]);
    INTERNAL_CHECK(shapes_tuple) << "debug.dump_tile third argument must be a tuple (shapes)";

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
      INTERNAL_CHECK(dim) << "debug.dump_tile shape must be static ConstInt at axis " << i;
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

struct PrintfSegment {
  std::string format_segment;
  char conversion;
};

static bool IsUnsignedMlirIntType(const std::string& type) {
  return type == "ui8" || type == "ui16" || type == "ui32" || type == "ui64";
}

static std::string GetUnsignedPrintfTargetType(const std::string& type) {
  if (type == "ui8" || type == "ui16") {
    return "i32";
  }
  if (type == "ui32" || type == "ui64") {
    return "i64";
  }
  return "";
}

static bool IsSupportedPrintfConversion(char conversion) {
  return conversion == 'd' || conversion == 'i' || conversion == 'u' || conversion == 'x' ||
         conversion == 'f';
}

static size_t FindPrintfConversionIndex(const std::string& format_segment) {
  size_t i = 0;
  while (i < format_segment.size()) {
    if (format_segment[i] != '%') {
      ++i;
      continue;
    }
    CHECK(!(i + 1 < format_segment.size() && format_segment[i + 1] == '%'))
        << "debug.printf does not support literal '%%'";

    size_t j = i + 1;
    while (j < format_segment.size()) {
      char c = format_segment[j];
      if (c == '-' || c == '+' || c == ' ' || c == '#' || c == '0') {
        ++j;
      } else {
        break;
      }
    }
    while (j < format_segment.size() && std::isdigit(static_cast<unsigned char>(format_segment[j]))) {
      ++j;
    }
    if (j < format_segment.size() && format_segment[j] == '.') {
      ++j;
      CHECK(j < format_segment.size() && std::isdigit(static_cast<unsigned char>(format_segment[j])))
          << "debug.printf precision must be followed by digits";
      while (j < format_segment.size() && std::isdigit(static_cast<unsigned char>(format_segment[j]))) {
        ++j;
      }
    }
    CHECK(j < format_segment.size()) << "debug.printf format ends with an incomplete conversion";
    CHECK(IsSupportedPrintfConversion(format_segment[j]))
        << "debug.printf does not support conversion '%" << format_segment[j] << "'";
    return j;
  }
  CHECK(false) << "debug.printf format segment must contain a supported conversion";
  return std::string::npos;
}

static std::string RewritePrintfFormatForScalarType(const std::string& format_segment, char conversion,
                                                    const std::string& scalar_type) {
  if (scalar_type != "i64") {
    return format_segment;
  }

  std::string replacement;
  switch (conversion) {
    case 'd':
      replacement = "lld";
      break;
    case 'i':
      replacement = "lli";
      break;
    case 'u':
      replacement = "llu";
      break;
    case 'x':
      replacement = "llx";
      break;
    default:
      return format_segment;
  }

  size_t conv_idx = FindPrintfConversionIndex(format_segment);
  size_t percent_idx = format_segment.rfind('%', conv_idx);
  INTERNAL_CHECK(percent_idx != std::string::npos)
      << "debug.printf failed to locate '%' while rewriting 64-bit format segment";

  std::string rewritten = format_segment;
  rewritten.replace(conv_idx, 1, replacement);
  return rewritten;
}

static std::string EscapeMlirStringLiteral(const std::string& text) {
  std::ostringstream oss;
  oss << "\"";
  for (char c : text) {
    switch (c) {
      case '\\':
        oss << "\\\\";
        break;
      case '"':
        oss << "\\\"";
        break;
      case '\n':
        oss << "\\n";
        break;
      case '\t':
        oss << "\\t";
        break;
      case '\r':
        oss << "\\r";
        break;
      default:
        oss << c;
        break;
    }
  }
  oss << "\"";
  return oss.str();
}

static std::vector<PrintfSegment> ParsePrintfSegments(const std::string& format) {
  std::vector<PrintfSegment> segments;
  std::string pending_text;
  size_t i = 0;
  while (i < format.size()) {
    if (format[i] != '%') {
      pending_text.push_back(format[i]);
      ++i;
      continue;
    }
    if (i + 1 < format.size() && format[i + 1] == '%') {
      CHECK(false) << "debug.printf does not support literal '%%'";
    }

    size_t j = i + 1;
    while (j < format.size()) {
      char c = format[j];
      if (c == '-' || c == '+' || c == ' ' || c == '#' || c == '0') {
        ++j;
      } else {
        break;
      }
    }
    while (j < format.size() && std::isdigit(static_cast<unsigned char>(format[j]))) {
      ++j;
    }
    if (j < format.size() && format[j] == '.') {
      ++j;
      CHECK(j < format.size() && std::isdigit(static_cast<unsigned char>(format[j])))
          << "debug.printf precision must be followed by digits";
      while (j < format.size() && std::isdigit(static_cast<unsigned char>(format[j]))) {
        ++j;
      }
    }

    CHECK(j < format.size()) << "debug.printf format ends with an incomplete conversion";
    char conversion = format[j];
    CHECK(IsSupportedPrintfConversion(conversion))
        << "debug.printf does not support conversion '%" << conversion << "'";

    segments.push_back({pending_text + format.substr(i, j - i + 1), conversion});
    pending_text.clear();
    i = j + 1;
  }

  if (!pending_text.empty()) {
    if (segments.empty()) {
      return segments;
    }
    segments.back().format_segment += pending_text;
  }
  return segments;
}

static void EmitPrintfSegments(codegen::PTOCodegen& codegen, const std::string& format,
                               const std::vector<ir::ExprPtr>& args, size_t arg_offset,
                               const std::string& indent = "") {
  auto segments = ParsePrintfSegments(format);
  if (segments.empty()) {
    CHECK(args.size() == arg_offset) << "printf-like lowering format expects 0 scalar arguments, but got "
                                     << (args.size() - arg_offset);
    std::string dummy = codegen.NewTemp();
    codegen.Emit(indent + dummy + " = arith.constant 0 : i32");
    codegen.Emit(indent + "pto.print ins(" + EscapeMlirStringLiteral(format) + ", " + dummy + " : i32)");
    return;
  }
  CHECK(segments.size() == args.size() - arg_offset) << "printf-like lowering segment count (" << segments.size()
                                                     << ") must match scalar arg count ("
                                                     << (args.size() - arg_offset) << ")";

  for (size_t i = 0; i < segments.size(); ++i) {
    size_t arg_index = arg_offset + i;
    std::string scalar = codegen.GetExprAsCode(args[arg_index]);
    std::string scalar_type = codegen.GetExprTypeAnnotation(args[arg_index]);
    INTERNAL_CHECK(!scalar_type.empty()) << "debug.printf scalar argument " << i << " is missing type annotation";

    if ((segments[i].conversion == 'd' || segments[i].conversion == 'i' || segments[i].conversion == 'u') &&
        scalar_type == "i1") {
      std::string casted = codegen.NewTemp();
      codegen.Emit(indent + casted + " = arith.extui " + scalar + " : i1 to i32");
      scalar = casted;
      scalar_type = "i32";
    }

    if ((segments[i].conversion == 'd' || segments[i].conversion == 'i' ||
         segments[i].conversion == 'u' || segments[i].conversion == 'x') &&
        scalar_type == "index") {
      std::string casted = codegen.NewTemp();
      codegen.Emit(indent + casted + " = arith.index_cast " + scalar + " : index to i64");
      scalar = casted;
      scalar_type = "i64";
    }

    if ((segments[i].conversion == 'u' || segments[i].conversion == 'x') &&
        IsUnsignedMlirIntType(scalar_type)) {
      // Approximate unsigned printf support by materializing a wider signless
      // integer temporary before emitting pto.print:
      //   ui8/ui16 -> i32
      //   ui32/ui64 -> i64
      //
      // This keeps values stable for UINT8/16/32. UINT64 is only conditionally
      // trustworthy here: values <= INT64_MAX remain faithful after the cast,
      // while larger runtime UINT64 values are not guaranteed to print
      // correctly with the current route.
      std::string casted = codegen.NewTemp();
      std::string target_type = GetUnsignedPrintfTargetType(scalar_type);
      INTERNAL_CHECK(!target_type.empty())
          << "debug.printf failed to choose target signless type for unsigned scalar " << scalar_type;
      codegen.Emit(indent + casted + " = builtin.unrealized_conversion_cast " + scalar + " : " + scalar_type +
                   " to " + target_type);
      scalar = casted;
      scalar_type = target_type;
    }

    INTERNAL_CHECK(!(segments[i].conversion == 'x' && scalar_type == "i1"))
        << "debug.printf %x does not support bool scalars";
    INTERNAL_CHECK(!(segments[i].conversion == 'f' && scalar_type != "f32"))
        << "debug.printf %f requires f32 operand after frontend/IR validation, but got " << scalar_type;

    std::string rewritten_format =
        RewritePrintfFormatForScalarType(segments[i].format_segment, segments[i].conversion, scalar_type);

    std::ostringstream oss;
    oss << "pto.print ins(" << EscapeMlirStringLiteral(rewritten_format) << ", " << scalar << " : "
        << scalar_type << ")";
    codegen.Emit(indent + oss.str());
  }
}

static std::string MakeDebugPrintfCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);

  std::string format = op->GetKwarg<std::string>("format");
  EmitPrintfSegments(codegen, format, op->args_, 0);
  return "";
}

static std::string MakeDebugAssertCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() >= 1) << "debug.assert requires at least 1 condition argument, but got "
                               << op->args_.size();

  std::string condition = codegen.GetExprAsCode(op->args_[0]);
  std::string condition_type = codegen.GetExprTypeAnnotation(op->args_[0]);
  if (condition_type.empty()) {
    if (auto scalar_type = As<ir::ScalarType>(op->args_[0]->GetType());
        scalar_type && scalar_type->dtype_ == DataType::BOOL) {
      condition_type = "i1";
    }
  }
  CHECK(condition_type == "i1") << "debug.assert requires i1 condition after frontend/IR validation, but got "
                                << condition_type;

  std::string condition_text = op->GetKwarg<std::string>("condition_text");
  std::string format = op->GetKwarg<std::string>("format");

  std::string false_value = codegen.NewTemp();
  codegen.Emit(false_value + " = arith.constant false");

  std::string failed = codegen.NewTemp();
  codegen.Emit(failed + " = arith.cmpi eq, " + condition + ", " + false_value + " : i1");

  std::string printed_message = "[ASSERT] Assertion '" + condition_text + "'";
  if (!format.empty()) {
    printed_message += ", " + format;
  }
  if (printed_message.empty() || printed_message.back() != '\n') {
    printed_message += "\n";
  }

  codegen.Emit("scf.if " + failed + " {");
  EmitPrintfSegments(codegen, printed_message, op->args_, 1, "  ");
  codegen.Emit("  pto.trap");
  codegen.Emit("}");
  return "";
}

static std::string MakeTrapCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.empty()) << "debug.trap takes no arguments, but got " << op->args_.size();
  codegen.Emit("pto.trap");
  return "";
}

static std::string GetPartitionType(const ir::MakeTuple* shapes_tuple, const std::string& dtype_str) {
  std::ostringstream oss;
  oss << "!pto.partition_tensor_view<";
  for (size_t i = 0; i < shapes_tuple->elements_.size(); ++i) {
    if (i > 0) oss << "x";
    if (auto dim = As<ir::ConstInt>(shapes_tuple->elements_[i])) {
      oss << dim->value_;
    } else {
      oss << "?";
    }
  }
  oss << "x" << dtype_str << ">";
  return oss.str();
}

static std::string MakeTensorPrintCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "debug.dump_tensor requires 3 arguments, but got " << op->args_.size();

  auto tensor = As<Var>(op->args_[0]);
  INTERNAL_CHECK(tensor) << "debug.dump_tensor first argument must be a Var";

  auto offsets_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK(offsets_tuple) << "debug.dump_tensor second argument must be a tuple (offsets)";

  auto shapes_tuple = As<ir::MakeTuple>(op->args_[2]);
  INTERNAL_CHECK(shapes_tuple) << "debug.dump_tensor third argument must be a tuple (shapes)";

  auto tensor_type = As<TensorType>(tensor->GetType());
  INTERNAL_CHECK(tensor_type) << "debug.dump_tensor tensor argument must have TensorType";

  std::string tensor_view = codegen.GetOrCreateTensorView(tensor);
  std::string tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());
  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
  std::string partition_type = GetPartitionType(shapes_tuple.get(), dtype_str);

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

// block.load: emit pto.partition_view + pto.tload
// For N-dim tensor (N > 2), flatten first N-1 dims into row offset
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

  auto tensor_type = As<TensorType>(tensor->GetType());
  INTERNAL_CHECK(tensor_type) << "block.load tensor argument must have TensorType";

  size_t tensor_ndim = tensor_type->shape_.size();
  INTERNAL_CHECK(tensor_ndim >= 2) << "block.load: tensor must have at least 2 dimensions";

  // shapes must be 2D (tile shape)
  size_t shapes_ndim = shapes_tuple->elements_.size();
  INTERNAL_CHECK(shapes_ndim == 2)
      << "block.load: shapes tuple must have 2 elements (tile height and width)";

  // height and width are the two dimensions
  int64_t height = codegen.GetConstIntValue(shapes_tuple->elements_[0]);
  int64_t width = codegen.GetConstIntValue(shapes_tuple->elements_[1]);

  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
  std::string tile_buf = codegen.GetCurrentResultTarget();
  INTERNAL_CHECK(!tile_buf.empty()) << "block.load requires assignment target (tile_buf)";
  std::string tile_buf_type = codegen.GetCurrentResultTileBufTypeString();
  std::string partition_type = "!pto.partition_tensor_view<" + std::to_string(height) + "x" +
                               std::to_string(width) + "x" + dtype_str + ">";

  // Get all dimensions
  std::vector<std::string> dims(tensor_ndim);
  for (size_t i = 0; i < tensor_ndim; ++i) {
    if (auto var_i = As<ir::Var>(tensor_type->shape_[i])) {
      dims[i] = codegen.GetVarName(var_i);
    } else {
      dims[i] = codegen.GetIndexConstant(codegen.GetConstIntValue(tensor_type->shape_[i]));
    }
  }

  // Compute row_off and col_off
  std::string row_off, col_off;
  if (tensor_ndim == 2) {
    // 2D tensor: use offsets directly
    row_off = codegen.GetExprAsCode(offsets_tuple->elements_[0]);
    col_off = codegen.GetExprAsCode(offsets_tuple->elements_[1]);
  } else {
    // N-D tensor (N > 2): compute flattened row offset
    // row_offset = offset[0]
    // for i in 1..N-1: row_offset = row_offset * dim[i] + offset[i]
    row_off = codegen.GetExprAsCode(offsets_tuple->elements_[0]);
    for (size_t i = 1; i < tensor_ndim - 1; ++i) {
      std::string dim_i = dims[i];
      std::string off_i = codegen.GetExprAsCode(offsets_tuple->elements_[i]);
      std::string new_row_off = codegen.NewTemp();
      codegen.Emit(new_row_off + " = arith.muli " + row_off + ", " + dim_i + " : index");
      std::string tmp = new_row_off;
      new_row_off = codegen.NewTemp();
      codegen.Emit(new_row_off + " = arith.addi " + tmp + ", " + off_i + " : index");
      row_off = new_row_off;
    }
    col_off = codegen.GetExprAsCode(offsets_tuple->elements_[tensor_ndim - 1]);
  }

  // Create flattened 2D tensor_view for N-D tensor (N > 2)
  std::string tensor_view;
  std::string tensor_view_type;
  if (tensor_ndim == 2) {
    tensor_view = codegen.GetOrCreateTensorView(tensor);
    tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());
  } else {
    // Compute flattened row dimension: prod(dim[0..N-2])
    // For 4D tensor [B, N, Sq, D], this is B * N * Sq
    // row_off = ((b_idx * N + n_idx) * Sq + sq_off) is the linear index into [B*N*Sq]
    std::string flat_row_dim = dims[0];
    for (size_t i = 1; i < tensor_ndim - 1; ++i) {
      std::string new_dim = codegen.NewTemp();
      codegen.Emit(new_dim + " = arith.muli " + flat_row_dim + ", " + dims[i] + " : index");
      flat_row_dim = new_dim;
    }
    
    std::string col_dim = dims[tensor_ndim - 1];
    std::string c1 = codegen.GetIndexConstant(1);
    std::string raw_ptr = codegen.GetTensorPtr(tensor);
    
    // Create a 2D tensor_view with shape [flat_row_dim, col_dim]
    tensor_view_type = "!pto.tensor_view<?x?x" + dtype_str + ">";
    std::string nd_view = codegen.NewTemp();
    std::ostringstream tv_line;
    tv_line << nd_view << " = pto.make_tensor_view " << raw_ptr
            << ", shape = [" << flat_row_dim << ", " << col_dim << "],"
            << " strides = [" << col_dim << ", " << c1 << "]"
            << " : " << tensor_view_type;
    codegen.Emit(tv_line.str());
    tensor_view = nd_view;
  }

  // Build partition_view with 2 offsets and 2 sizes
  std::string partition_view = codegen.NewTemp();
  std::ostringstream partition_line;
  partition_line << partition_view << " = pto.partition_view " << tensor_view;
  partition_line << ", offsets = [" << row_off << ", " << col_off << "]";
  partition_line << ", sizes = [" << codegen.GetIndexConstant(height) << ", " << codegen.GetIndexConstant(width) << "]";
  partition_line << " : " << tensor_view_type << " -> " << partition_type;
  codegen.Emit(partition_line.str());

  std::ostringstream tload_line;
  tload_line << "pto.tload ins(" << partition_view << " : " << partition_type << ") outs(";
  tload_line << tile_buf << " : " << tile_buf_type << ")";
  codegen.Emit(tload_line.str());
  return "";  // Multi-line emission
}

// block.store: emit pto.partition_view + pto.tstore
// For N-dim tensor (N > 2), flatten first N-1 dims into row offset
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

  auto output_tensor = As<Var>(op->args_[3]);
  INTERNAL_CHECK(output_tensor) << "block.store output_tensor must be a Var";

  auto tensor_type = As<TensorType>(output_tensor->GetType());
  INTERNAL_CHECK(tensor_type) << "block.store output_tensor must have TensorType";

  size_t tensor_ndim = tensor_type->shape_.size();
  INTERNAL_CHECK(tensor_ndim >= 2) << "block.store: tensor must have at least 2 dimensions";

  // shapes must be 2D (tile shape)
  size_t shapes_ndim = shapes_tuple->elements_.size();
  INTERNAL_CHECK(shapes_ndim == 2)
      << "block.store: shapes tuple must have 2 elements (tile height and width)";

  // height and width are the two dimensions
  int64_t height = codegen.GetConstIntValue(shapes_tuple->elements_[0]);
  int64_t width = codegen.GetConstIntValue(shapes_tuple->elements_[1]);

  std::string dtype_str = codegen.GetTypeString(tensor_type->dtype_);
  std::string tile_buf = codegen.GetVarName(tile);

  std::string partition_type = "!pto.partition_tensor_view<" + std::to_string(height) + "x" +
                               std::to_string(width) + "x" + dtype_str + ">";

  // Get tile_buf type via GetExprTypeAnnotation which correctly handles
  // dynamically-allocated buffers (e.g., reshape outputs in extra_tile_buf_types_)
  std::string tile_buf_type = codegen.GetExprTypeAnnotation(op->args_[0]);

  // Get all dimensions
  std::vector<std::string> dims(tensor_ndim);
  for (size_t i = 0; i < tensor_ndim; ++i) {
    if (auto var_i = As<ir::Var>(tensor_type->shape_[i])) {
      dims[i] = codegen.GetVarName(var_i);
    } else {
      dims[i] = codegen.GetIndexConstant(codegen.GetConstIntValue(tensor_type->shape_[i]));
    }
  }

  // Compute row_off and col_off
  std::string row_off, col_off;
  if (tensor_ndim == 2) {
    // 2D tensor: use offsets directly
    row_off = codegen.GetExprAsCode(offsets_tuple->elements_[0]);
    col_off = codegen.GetExprAsCode(offsets_tuple->elements_[1]);
  } else {
    // N-D tensor (N > 2): compute flattened row offset
    // row_offset = offset[0]
    // for i in 1..N-1: row_offset = row_offset * dim[i] + offset[i]
    row_off = codegen.GetExprAsCode(offsets_tuple->elements_[0]);
    for (size_t i = 1; i < tensor_ndim - 1; ++i) {
      std::string dim_i = dims[i];
      std::string off_i = codegen.GetExprAsCode(offsets_tuple->elements_[i]);
      std::string new_row_off = codegen.NewTemp();
      codegen.Emit(new_row_off + " = arith.muli " + row_off + ", " + dim_i + " : index");
      std::string tmp = new_row_off;
      new_row_off = codegen.NewTemp();
      codegen.Emit(new_row_off + " = arith.addi " + tmp + ", " + off_i + " : index");
      row_off = new_row_off;
    }
    col_off = codegen.GetExprAsCode(offsets_tuple->elements_[tensor_ndim - 1]);
  }

  // Create flattened 2D tensor_view for N-D tensor (N > 2)
  std::string tensor_view;
  std::string tensor_view_type;
  if (tensor_ndim == 2) {
    tensor_view = codegen.GetOrCreateTensorView(output_tensor);
    tensor_view_type = codegen.GetTensorViewTypeString(tensor_type.get());
  } else {
    // Compute flattened row dimension: prod(dim[0..N-2])
    // For 4D tensor [B, N, Sq, D], this is B * N * Sq
    std::string flat_row_dim = dims[0];
    for (size_t i = 1; i < tensor_ndim - 1; ++i) {
      std::string new_dim = codegen.NewTemp();
      codegen.Emit(new_dim + " = arith.muli " + flat_row_dim + ", " + dims[i] + " : index");
      flat_row_dim = new_dim;
    }
    
    std::string col_dim = dims[tensor_ndim - 1];
    std::string c1 = codegen.GetIndexConstant(1);
    std::string raw_ptr = codegen.GetTensorPtr(output_tensor);
    
    // Create a 2D tensor_view with shape [flat_row_dim, col_dim]
    tensor_view_type = "!pto.tensor_view<?x?x" + dtype_str + ">";
    std::string nd_view = codegen.NewTemp();
    std::ostringstream tv_line;
    tv_line << nd_view << " = pto.make_tensor_view " << raw_ptr
            << ", shape = [" << flat_row_dim << ", " << col_dim << "],"
            << " strides = [" << col_dim << ", " << c1 << "]"
            << " : " << tensor_view_type;
    codegen.Emit(tv_line.str());
    tensor_view = nd_view;
  }

  // Build partition_view with 2 offsets and 2 sizes
  std::string partition_view = codegen.NewTemp();
  std::ostringstream partition_line;
  partition_line << partition_view << " = pto.partition_view " << tensor_view;
  partition_line << ", offsets = [" << row_off << ", " << col_off << "]";
  partition_line << ", sizes = [" << codegen.GetIndexConstant(height) << ", " << codegen.GetIndexConstant(width) << "]";
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

REGISTER_BACKEND_OP(Backend910B_PTO, "debug.dump_tile")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakePrintCodegenPTO("pto.tprint", op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "tensor.dim")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTensorDimCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "debug.dump_tensor")
    .set_pipe(ir::PipeType::V)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTensorPrintCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "debug.printf")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeDebugPrintfCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "debug.assert")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeDebugAssertCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "debug.trap")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTrapCodegenPTO(op, codegen);
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

  // Get the input type from the argument
  auto scalar_type = As<ir::ScalarType>(op->args_[0]->GetType());
  CHECK(scalar_type) << "block.index_cast requires argument to be ScalarType";
  std::string input_type = codegen.GetTypeString(scalar_type->dtype_);

  // Create a new SSA variable for the result
  std::string result = codegen.NewTemp();

  // Emit arith.index_cast operation with correct input type
  codegen.Emit(result + " = arith.index_cast " + idx + " : " + input_type + " to index");

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

// Dynamic event_id variants (intra-core)
static std::string MakeSyncSrcDynCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  auto set_pipe = op->GetKwarg<int>("set_pipe");
  auto wait_pipe = op->GetKwarg<int>("wait_pipe");
  std::string event_id = codegen.GetExprAsCode(op->args_[0]);
  std::ostringstream oss;
  oss << "pto.set_flag_dyn[<PIPE_" << GetPipeTypeName(static_cast<ir::PipeType>(set_pipe))
      << ">, <PIPE_" << GetPipeTypeName(static_cast<ir::PipeType>(wait_pipe))
      << ">, " << event_id << "]";
  codegen.Emit(oss.str());
  return "";
}

static std::string MakeSyncDstDynCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  auto set_pipe = op->GetKwarg<int>("set_pipe");
  auto wait_pipe = op->GetKwarg<int>("wait_pipe");
  std::string event_id = codegen.GetExprAsCode(op->args_[0]);
  std::ostringstream oss;
  oss << "pto.wait_flag_dyn[<PIPE_" << GetPipeTypeName(static_cast<ir::PipeType>(set_pipe))
      << ">, <PIPE_" << GetPipeTypeName(static_cast<ir::PipeType>(wait_pipe))
      << ">, " << event_id << "]";
  codegen.Emit(oss.str());
  return "";
}

REGISTER_BACKEND_OP(Backend910B_PTO, "system.sync_src_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeSyncSrcDynCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "system.sync_dst_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeSyncDstDynCodegenPTO(op, codegen);
    });
// Barrier operations
static std::string MakeBarVCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  const char* arch = std::getenv("npu_arch");
  std::string arch_val(arch);
  if (arch_val == "dav-c220") {
    codegen.Emit("pto.barrier #pto.pipe<PIPE_V>");
  }
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
  auto mode_id = op->GetKwarg<int>("mode_id");
  std::ostringstream oss;
  oss << "pto.sync.set #pto.pipe<PIPE_" << GetPipeTypeName(static_cast<ir::PipeType>(pipe))
      << ">, " << event_id <<  " {ffts_mode = " << mode_id <<" : i32}";
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

// Dynamic event_id variants (cross-core)
// PTOAS's pto.sync.set/wait only accept static I32Attr event_id.
// Lower dynamic event_id to an scf.if chain of static ops:
//   %cond0 = arith.cmpi eq, %eid, %c0 : index
//   scf.if %cond0 { pto.sync.set #pto.pipe<PIPE_X>, 0 }
//   %cond1 = arith.cmpi eq, %eid, %c1 : index
//   scf.if %cond1 { pto.sync.set #pto.pipe<PIPE_X>, 1 }
//   ...
static std::string MakeSetCrossCoreDynCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  auto pipe = op->GetKwarg<int>("pipe");
  int max_eid = op->GetKwarg<int>("max_event_id");
  std::string pipe_name = GetPipeTypeName(static_cast<ir::PipeType>(pipe));
  std::string event_id = codegen.GetExprAsCode(op->args_[0]);

  for (int i = 0; i < max_eid; ++i) {
    std::string ci = codegen.GetIndexConstant(static_cast<int64_t>(i));
    std::string cond = codegen.NewTemp();
    codegen.Emit(cond + " = arith.cmpi eq, " + event_id + ", " + ci + " : index");
    codegen.Emit("scf.if " + cond + " {");
    codegen.Emit("  pto.sync.set #pto.pipe<PIPE_" + pipe_name + ">, " + std::to_string(i));
    codegen.Emit("}");
  }
  return "";
}

static std::string MakeWaitCrossCoreDynCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  auto pipe = op->GetKwarg<int>("pipe");
  int max_eid = op->GetKwarg<int>("max_event_id");
  std::string pipe_name = GetPipeTypeName(static_cast<ir::PipeType>(pipe));
  std::string event_id = codegen.GetExprAsCode(op->args_[0]);

  for (int i = 0; i < max_eid; ++i) {
    std::string ci = codegen.GetIndexConstant(static_cast<int64_t>(i));
    std::string cond = codegen.NewTemp();
    codegen.Emit(cond + " = arith.cmpi eq, " + event_id + ", " + ci + " : index");
    codegen.Emit("scf.if " + cond + " {");
    codegen.Emit("  pto.sync.wait #pto.pipe<PIPE_" + pipe_name + ">, " + std::to_string(i));
    codegen.Emit("}");
  }
  return "";
}

REGISTER_BACKEND_OP(Backend910B_PTO, "system.set_cross_core_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeSetCrossCoreDynCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "system.wait_cross_core_dyn")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeWaitCrossCoreDynCodegenPTO(op, codegen);
    });

static std::string MakeSystemSyncAllCodegenPTO(const CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.empty()) << "system.sync_all takes no arguments, but got " << op->args_.size();
  const char* arch = std::getenv("npu_arch");
  std::string arch_val(arch);
  bool aiv_only = false;
  int trigger_pipe_val = 0, wait_pipe_val = 0;
  for (const auto& [key, value] : op->kwargs_) {
    if (key == "aiv_only") {
      aiv_only = std::any_cast<bool>(value);
    } else if (key == "trigger_pipe") {
      trigger_pipe_val = std::any_cast<int>(value);
    } else if (key == "wait_pipe") {
      wait_pipe_val = std::any_cast<int>(value);
    }
  }
  std::ostringstream oss;
  if (arch_val == "dav-c220") {
    codegen.Emit("pto.barrier #pto.pipe<PIPE_ALL>");
    if (aiv_only) {
      oss << "pto.sync.set #pto.pipe<PIPE_MTE3>, " << SYNC_AIV_ONLY_ALL << " {ffts_mode = 0 : i32}" << "\n      "
          << "pto.sync.wait #pto.pipe<PIPE_MTE3>, " << SYNC_AIV_ONLY_ALL;
      codegen.Emit(oss.str());
      return "";
    }
    // AIC
    oss << "pto.section.cube {" << "\n      "
        << "pto.sync.wait #pto.pipe<PIPE_S>, " << SYNC_AIV_FLAG << "\n      "
        << "pto.sync.set #pto.pipe<PIPE_FIX>, " << SYNC_AIC_FLAG << " {ffts_mode = 0 : i32}" << "\n      "
        << "pto.sync.wait #pto.pipe<PIPE_S>, " << SYNC_AIC_FLAG << "\n      "
        << "pto.sync.set #pto.pipe<PIPE_MTE3>, " << SYNC_AIC_AIV_FLAG << "\n    }";
    codegen.Emit(oss.str());
    // AIV
    oss.str("");
    oss << "pto.section.vector {" << "\n      "
        << "pto.sync.set #pto.pipe<PIPE_MTE3>, " << SYNC_AIV_FLAG << "\n      "
        << "pto.sync.wait #pto.pipe<PIPE_S>, " << SYNC_AIC_AIV_FLAG << "\n    }";
    codegen.Emit(oss.str());
  } else if (arch_val == "dav-c310") {
    if (aiv_only) {
      codegen.Emit("pto.section.vector {");
      std::string trigger_pipe_str = PipeTypeToString(static_cast<ir::PipeType>(trigger_pipe_val));
      codegen.Emit(" pto.barrier #pto.pipe<PIPE_" + trigger_pipe_str + ">");
      if (trigger_pipe_val == static_cast<int>(ir::PipeType::ALL)) {
        oss << " pto.sync.set #pto.pipe<PIPE_MTE3>, " << SYNC_AIV_ONLY_ALL << " {ffts_mode = 0 : i32}";
        codegen.Emit(oss.str());
      } else {
        oss << " pto.sync.set #pto.pipe<PIPE_" << trigger_pipe_str << ">, " << SYNC_AIV_ONLY_ALL << " {ffts_mode = 0 : i32}";
        codegen.Emit(oss.str());
      }
      if (wait_pipe_val == static_cast<int>(ir::PipeType::ALL)) {
        oss.str("");
        oss << " pto.sync.wait #pto.pipe<PIPE_S>, " << SYNC_AIV_ONLY_ALL;
        codegen.Emit(oss.str());
      } else {
        oss.str("");
        std::string wait_pipe_str = PipeTypeToString(static_cast<ir::PipeType>(wait_pipe_val));
        oss << " pto.sync.wait #pto.pipe<PIPE_" << wait_pipe_str << ">, " << SYNC_AIV_ONLY_ALL;
        codegen.Emit(oss.str());
      }
      codegen.Emit("}");
      return "";
    }
    codegen.Emit("pto.barrier #pto.pipe<PIPE_ALL>");
    // AIC
    oss << "pto.section.cube {" << "\n      "
        << "pto.sync.wait #pto.pipe<PIPE_S>, " << SYNC_AIV_FLAG << "\n      "
        << "pto.sync.wait #pto.pipe<PIPE_S>, " << (SYNC_AIV_FLAG + SYNC_FLAG_ID_MAX) << "\n      "
        // << "pto.sync.set #pto.pipe<PIPE_FIX>, " << SYNC_AIC_FLAG << " {ffts_mode = 0 : i32}" << "\n      "
        // << "pto.sync.wait #pto.pipe<PIPE_S>, " << SYNC_AIC_FLAG << "\n      "
        << "pto.sync.set #pto.pipe<PIPE_S>, " << SYNC_AIC_AIV_FLAG << "\n      "
        << "pto.sync.set #pto.pipe<PIPE_S>, " << (SYNC_AIC_AIV_FLAG + SYNC_FLAG_ID_MAX) << "\n    }";
    codegen.Emit(oss.str());
    // AIV
    oss.str("");
    oss << "pto.section.vector {" << "\n      "
        << "pto.sync.set #pto.pipe<PIPE_MTE3>, " << SYNC_AIV_FLAG << "\n      "
        << "pto.sync.wait #pto.pipe<PIPE_S>, " << SYNC_AIC_AIV_FLAG << "\n    }";
    codegen.Emit(oss.str());
  }
  return "";
}

REGISTER_BACKEND_OP(Backend910B_PTO, "system.sync_all")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeSystemSyncAllCodegenPTO(op, codegen);
    });

// Helper function for BlockGetVal
static std::string MakeBlockGetValCodegenPTO(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "block.getval requires 2 arguments (tile, index), but got "
                                << op->args_.size();

  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string index = codegen.GetExprAsCode(op->args_[1]);
  std::string tile_type = codegen.GetExprTypeAnnotation(op->args_[0]);

  // Create a new SSA variable for the scalar result
  std::string result = codegen.NewTemp();

  std::ostringstream oss;
  oss << result << " = pto.tgetval ins(" << tile << ", " << index;
  if (!tile_type.empty()) oss << " : " << tile_type;
  oss << ", index)";

  // Get the result type (scalar type matching tile)
  auto result_type = As<ir::ScalarType>(op->GetType());
  INTERNAL_CHECK(result_type) << "block.getval result must be ScalarType";
  std::string result_type_str = codegen.GetTypeString(result_type->dtype_);
  if (!result_type_str.empty()) oss << " outs : " << result_type_str;

  codegen.Emit(oss.str());

  // Register the result variable mapping
  codegen.SetVarMlirName(codegen.GetCurrentResultVarName(), result);

  return "";
}

// Helper function for BlockSetVal
static std::string MakeBlockSetValCodegenPTO(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "block.setval requires 3 arguments (tile, index, value), but got "
                                << op->args_.size();

  std::string tile = codegen.GetExprAsCode(op->args_[0]);
  std::string index = codegen.GetExprAsCode(op->args_[1]);
  std::string value = codegen.GetExprAsCode(op->args_[2]);
  std::string tile_type = codegen.GetExprTypeAnnotation(op->args_[0]);

  std::ostringstream oss;
  oss << "pto.tsetval ins(" << index << ", " << value;
  if (!tile_type.empty()) {
    auto value_type = As<ir::ScalarType>(op->args_[2]->GetType());
    INTERNAL_CHECK(value_type) << "block.setval value must be ScalarType";
    oss << " : index, " << codegen.GetTypeString(value_type->dtype_);
  }
  oss << ") outs(" << tile;
  if (!tile_type.empty()) oss << " : " << tile_type;
  oss << ")";

  codegen.Emit(oss.str());

  return "";
}

REGISTER_BACKEND_OP(Backend910B_PTO, "block.getval")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockGetValCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "block.setval")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeBlockSetValCodegenPTO(op, codegen);
    });

// Helper function for TensorGetVal
static std::string MakeTensorGetValCodegenPTO(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "tensor.getval requires 2 arguments (tensor, offset), but got "
                                << op->args_.size();

  std::string tensor = codegen.GetExprAsCode(op->args_[0]);
  std::string offset = codegen.GetExprAsCode(op->args_[1]);

  // Get the tensor pointer
  auto tensor_var = As<ir::Var>(op->args_[0]);
  INTERNAL_CHECK(tensor_var) << "tensor.getval requires tensor to be a Var";
  std::string tensor_ptr = codegen.GetTensorPtr(tensor_var);

  // Create a new SSA variable for the scalar result
  std::string result = codegen.NewTemp();

  // Get the result type (scalar type matching tensor)
  auto result_type = As<ir::ScalarType>(op->GetType());
  INTERNAL_CHECK(result_type) << "tensor.getval result must be ScalarType";
  std::string result_type_str = codegen.GetTypeString(result_type->dtype_);

  std::ostringstream oss;
  oss << result << " = pto.load_scalar " << tensor_ptr << "[" << offset << "] : !pto.ptr<" << result_type_str << "> -> " << result_type_str;

  codegen.Emit(oss.str());

  // Register the result variable mapping
  codegen.SetVarMlirName(codegen.GetCurrentResultVarName(), result);

  return "";
}

// Helper function for TensorSetVal
static std::string MakeTensorSetValCodegenPTO(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 3) << "tensor.setval requires 3 arguments (tensor, offset, value), but got "
                                << op->args_.size();

  std::string tensor = codegen.GetExprAsCode(op->args_[0]);
  std::string offset = codegen.GetExprAsCode(op->args_[1]);
  std::string value = codegen.GetExprAsCode(op->args_[2]);

  // Get the tensor pointer
  auto tensor_var = As<ir::Var>(op->args_[0]);
  INTERNAL_CHECK(tensor_var) << "tensor.setval requires tensor to be a Var";
  std::string tensor_ptr = codegen.GetTensorPtr(tensor_var);

  // Get the value type (scalar type matching tensor)
  auto tensor_type = As<ir::TensorType>(op->args_[0]->GetType());
  INTERNAL_CHECK(tensor_type) << "tensor.setval requires tensor to be TensorType";
  std::string value_type_str = codegen.GetTypeString(tensor_type->dtype_);

  std::ostringstream oss;
  oss << "pto.store_scalar " << value << ", " << tensor_ptr << "[" << offset << "] : !pto.ptr<" << value_type_str << ">, " << value_type_str;

  codegen.Emit(oss.str());

  return "";
}

REGISTER_BACKEND_OP(Backend910B_PTO, "tensor.getval")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTensorGetValCodegenPTO(op, codegen);
    });

REGISTER_BACKEND_OP(Backend910B_PTO, "tensor.setval")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTensorSetValCodegenPTO(op, codegen);
    });

// Helper function for TensorRead
static std::string MakeTensorReadCodegenPTO(const ir::CallPtr& op, codegen::CodegenBase& codegen_base) {
  auto& codegen = dynamic_cast<codegen::PTOCodegen&>(codegen_base);
  CHECK(op->args_.size() == 2) << "tensor.read requires 2 arguments (tensor, indices), but got "
                                << op->args_.size();

  // Get the tensor pointer
  auto tensor_var = As<ir::Var>(op->args_[0]);
  INTERNAL_CHECK(tensor_var) << "tensor.read requires tensor to be a Var";
  std::string tensor_ptr = codegen.GetTensorPtr(tensor_var);

  // Get tensor type for shape information
  auto tensor_type = As<ir::TensorType>(op->args_[0]->GetType());
  INTERNAL_CHECK(tensor_type) << "tensor.read requires tensor to be TensorType";

  // Get indices tuple
  auto indices_tuple = As<ir::MakeTuple>(op->args_[1]);
  INTERNAL_CHECK(indices_tuple) << "tensor.read indices must be MakeTuple";

  // Get the result type (scalar type matching tensor)
  auto result_type = As<ir::ScalarType>(op->GetType());
  INTERNAL_CHECK(result_type) << "tensor.read result must be ScalarType";
  std::string result_type_str = codegen.GetTypeString(result_type->dtype_);

  // Create a new SSA variable for the scalar result
  std::string result = codegen.NewTemp();

  // Calculate linear offset from multi-dimensional indices
  // For tensor with shape [d0, d1, ..., dn] and indices [i0, i1, ..., in]:
  // linear_offset = i0 * (d1 * d2 * ... * dn) + i1 * (d2 * ... * dn) + ... + in
  // All indices are converted to index type for consistent arithmetic
  std::string linear_offset;
  for (size_t i = 0; i < indices_tuple->elements_.size(); ++i) {
    std::string idx_str = codegen.GetExprAsCode(indices_tuple->elements_[i]);

    // Get the type of this index
    auto idx_type = As<ir::ScalarType>(indices_tuple->elements_[i]->GetType());
    std::string idx_mlir_type = idx_type ? codegen.GetTypeString(idx_type->dtype_) : "index";

    // Convert to index type if not already
    std::string idx_as_index;
    if (idx_mlir_type != "index") {
      idx_as_index = codegen.NewTemp();
      codegen.Emit(idx_as_index + " = arith.index_cast " + idx_str + " : " + idx_mlir_type + " to index");
    } else {
      idx_as_index = idx_str;
    }

    // Calculate stride: product of dimensions after this one
    int64_t stride = 1;
    for (size_t j = i + 1; j < tensor_type->shape_.size(); ++j) {
      auto dim_expr = tensor_type->shape_[j];
      if (auto const_int = As<ir::ConstInt>(dim_expr)) {
        stride *= const_int->value_;
      } else {
        // Dynamic dimension - need to compute at runtime
        stride = -1;
        break;
      }
    }

    if (stride > 0) {
      // Static stride - can compute directly
      std::string term;
      if (stride == 1) {
        term = idx_as_index;
      } else {
        std::string stride_var = codegen.NewTemp();
        codegen.Emit(stride_var + " = arith.constant " + std::to_string(stride) + " : index");
        std::string mul_result = codegen.NewTemp();
        codegen.Emit(mul_result + " = arith.muli " + idx_as_index + ", " + stride_var + " : index");
        term = mul_result;
      }

      if (linear_offset.empty()) {
        linear_offset = term;
      } else {
        std::string add_result = codegen.NewTemp();
        codegen.Emit(add_result + " = arith.addi " + linear_offset + ", " + term + " : index");
        linear_offset = add_result;
      }
    } else {
      // Dynamic stride - need more complex handling
      // For now, just use the index directly (fallback)
      if (linear_offset.empty()) {
        linear_offset = idx_as_index;
      } else {
        std::string add_result = codegen.NewTemp();
        codegen.Emit(add_result + " = arith.addi " + linear_offset + ", " + idx_as_index + " : index");
        linear_offset = add_result;
      }
    }
  }

  std::ostringstream oss;
  oss << result << " = pto.load_scalar " << tensor_ptr << "[" << linear_offset << "] : !pto.ptr<" << result_type_str << "> -> " << result_type_str;

  codegen.Emit(oss.str());

  // Set the result as the last assigned temp so it's returned by GetExprAsCode
  codegen.SetLastAssignedTemp(result);

  // Register the result variable mapping
  codegen.SetVarMlirName(codegen.GetCurrentResultVarName(), result);

  return "";
}

REGISTER_BACKEND_OP(Backend910B_PTO, "tensor.read")
    .set_pipe(ir::PipeType::S)
    .f_codegen([](const ir::CallPtr& op, codegen::CodegenBase& codegen) {
      return MakeTensorReadCodegenPTO(op, codegen);
    });

}  // namespace backend
}  // namespace pypto
