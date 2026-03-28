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

#include <any>
#include <cctype>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

bool IsSupportedPrintfConversion(char conversion) {
  return conversion == 'd' || conversion == 'i' || conversion == 'u' || conversion == 'x' ||
         conversion == 'f';
}

std::vector<char> ParsePrintfConversions(const std::string& format) {
  std::vector<char> conversions;
  size_t i = 0;
  while (i < format.size()) {
    if (format[i] != '%') {
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
    conversions.push_back(conversion);
    i = j + 1;
  }

  return conversions;
}

bool IsPrintfIntegerType(const DataType& dtype) {
  return dtype == DataType::INT8 || dtype == DataType::INT16 || dtype == DataType::INT32 ||
         dtype == DataType::INT64 || dtype == DataType::UINT8 || dtype == DataType::UINT16 ||
         dtype == DataType::UINT32 || dtype == DataType::UINT64;
}

bool IsPrintfSignedIntegerType(const DataType& dtype) {
  return dtype == DataType::INT8 || dtype == DataType::INT16 || dtype == DataType::INT32 ||
         dtype == DataType::INT64;
}

bool IsPrintfUnsignedIntegerType(const DataType& dtype) {
  return dtype == DataType::UINT8 || dtype == DataType::UINT16 || dtype == DataType::UINT32 ||
         dtype == DataType::UINT64;
}

bool IsPrintfIndexType(const DataType& dtype) {
  return dtype == DataType::INDEX;
}

bool IsPrintfBoolType(const DataType& dtype) {
  return dtype == DataType::BOOL;
}

TypePtr DeduceDebugDumpTensorType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 3) << "debug.dump_tensor requires exactly 3 arguments (tensor, offsets, shapes), but got "
                          << args.size();

  auto tensor_type = As<TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "debug.dump_tensor requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  auto offsets = As<MakeTuple>(args[1]);
  CHECK(offsets) << "debug.dump_tensor requires offsets to be a MakeTuple";

  auto shapes = As<MakeTuple>(args[2]);
  CHECK(shapes) << "debug.dump_tensor requires shapes to be a MakeTuple";

  const size_t rank = tensor_type->shape_.size();
  CHECK(offsets->elements_.size() == rank)
      << "debug.dump_tensor offsets count (" << offsets->elements_.size()
      << ") must match tensor rank (" << rank << ")";
  CHECK(shapes->elements_.size() == rank)
      << "debug.dump_tensor shapes count (" << shapes->elements_.size()
      << ") must match tensor rank (" << rank << ")";

  bool is_full_tensor_window = true;
  for (size_t i = 0; i < rank; ++i) {
    auto offset_const = As<ConstInt>(offsets->elements_[i]);
    if (!offset_const || offset_const->value_ != 0) {
      is_full_tensor_window = false;
      break;
    }
    auto shape_const = As<ConstInt>(shapes->elements_[i]);
    auto tensor_dim_const = As<ConstInt>(tensor_type->shape_[i]);
    if (!shape_const || !tensor_dim_const || shape_const->value_ != tensor_dim_const->value_) {
      is_full_tensor_window = false;
      break;
    }
  }

  if (!is_full_tensor_window && tensor_type->tensor_view_.has_value() &&
      !tensor_type->tensor_view_->stride.empty()) {
    const auto& last_stride = tensor_type->tensor_view_->stride.back();
    auto last_stride_const = As<ConstInt>(last_stride);
    CHECK(last_stride_const)
        << "debug.dump_tensor windowed mode requires the innermost stride to be statically 1";
    CHECK(last_stride_const->value_ == 1)
        << "debug.dump_tensor windowed mode requires innermost stride == 1, got "
        << last_stride_const->value_;
  }

  for (size_t i = 0; i < rank; ++i) {
    auto offset_scalar = As<ScalarType>(offsets->elements_[i]->GetType());
    CHECK(offset_scalar) << "debug.dump_tensor offset element " << i << " must be ScalarType, but got "
                         << offsets->elements_[i]->GetType()->TypeName();
    CHECK(offset_scalar->dtype_.IsInt())
        << "debug.dump_tensor offset element " << i << " must have integer dtype, but got "
        << offset_scalar->dtype_.ToString();
    CHECK(As<ConstInt>(offsets->elements_[i]))
        << "debug.dump_tensor currently only supports static offsets; axis " << i << " is dynamic";

    auto shape_scalar = As<ScalarType>(shapes->elements_[i]->GetType());
    CHECK(shape_scalar) << "debug.dump_tensor shape element " << i << " must be ScalarType, but got "
                        << shapes->elements_[i]->GetType()->TypeName();
    CHECK(shape_scalar->dtype_.IsInt())
        << "debug.dump_tensor shape element " << i << " must have integer dtype, but got "
        << shape_scalar->dtype_.ToString();
    auto shape_const = As<ConstInt>(shapes->elements_[i]);
    CHECK(shape_const) << "debug.dump_tensor currently only supports static shapes; axis " << i
                       << " is dynamic";
    CHECK(shape_const->value_ > 0) << "debug.dump_tensor shape element " << i
                                   << " must be positive, got " << shape_const->value_;
  }

  return GetUnknownType();
}

TypePtr DeduceDebugDumpTileType(const std::vector<ExprPtr>& args,
                                const std::vector<std::pair<std::string, std::any>>& kwargs) {
  CHECK(args.size() == 1 || args.size() == 3)
      << "debug.dump_tile requires 1 argument (tile) or 3 arguments (tile, offsets, shapes), but got "
      << args.size();

  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "debug.dump_tile requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  if (args.size() == 3) {
    auto offsets = As<MakeTuple>(args[1]);
    CHECK(offsets) << "debug.dump_tile requires second argument to be a MakeTuple (offsets)";

    auto shapes = As<MakeTuple>(args[2]);
    CHECK(shapes) << "debug.dump_tile requires third argument to be a MakeTuple (shapes)";

    const size_t rank = tile_type->shape_.size();
    CHECK(rank == 2) << "debug.dump_tile currently only supports 2D tile windows, but got rank " << rank;
    CHECK(offsets->elements_.size() == rank)
        << "debug.dump_tile offsets count (" << offsets->elements_.size()
        << ") must match tile rank (" << rank << ")";
    CHECK(shapes->elements_.size() == rank)
        << "debug.dump_tile shapes count (" << shapes->elements_.size()
        << ") must match tile rank (" << rank << ")";

    for (size_t i = 0; i < rank; ++i) {
      auto offset_scalar = As<ScalarType>(offsets->elements_[i]->GetType());
      CHECK(offset_scalar) << "debug.dump_tile offset element " << i << " must be ScalarType, but got "
                           << offsets->elements_[i]->GetType()->TypeName();
      CHECK(offset_scalar->dtype_.IsInt())
          << "debug.dump_tile offset element " << i << " must have integer dtype, but got "
          << offset_scalar->dtype_.ToString();
      CHECK(As<ConstInt>(offsets->elements_[i]))
          << "debug.dump_tile currently only supports static offsets; axis " << i << " is dynamic";

      auto shape_scalar = As<ScalarType>(shapes->elements_[i]->GetType());
      CHECK(shape_scalar) << "debug.dump_tile shape element " << i << " must be ScalarType, but got "
                          << shapes->elements_[i]->GetType()->TypeName();
      CHECK(shape_scalar->dtype_.IsInt())
          << "debug.dump_tile shape element " << i << " must have integer dtype, but got "
          << shape_scalar->dtype_.ToString();
      auto shape_const = As<ConstInt>(shapes->elements_[i]);
      CHECK(shape_const) << "debug.dump_tile currently only supports static shapes; axis " << i
                         << " is dynamic";
      CHECK(shape_const->value_ > 0) << "debug.dump_tile shape element " << i
                                     << " must be positive, got " << shape_const->value_;
    }
  }

  return GetUnknownType();
}

TypePtr DeduceDebugPrintfType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs) {
  bool found_format = false;
  std::string format;
  for (const auto& [key, value] : kwargs) {
    if (key == "format") {
      format = AnyCast<std::string>(value, "kwarg key: format");
      found_format = true;
      break;
    }
  }
  CHECK(found_format) << "debug.printf requires 'format' kwarg";

  auto conversions = ParsePrintfConversions(format);
  CHECK(conversions.size() == args.size()) << "debug.printf format expects " << conversions.size()
                                           << " scalar arguments, but got " << args.size();

  for (size_t i = 0; i < args.size(); ++i) {
    auto scalar_type = As<ScalarType>(args[i]->GetType());
    CHECK(scalar_type) << "debug.printf argument " << i << " must be ScalarType, but got "
                       << args[i]->GetType()->TypeName();

    const DataType& dtype = scalar_type->dtype_;
    char conversion = conversions[i];
    if (conversion == 'f') {
      CHECK(dtype == DataType::FP32)
          << "debug.printf conversion '%f' requires FP32 scalar, but got "
          << dtype.ToString();
    } else if (conversion == 'x') {
      CHECK(IsPrintfUnsignedIntegerType(dtype) || IsPrintfIndexType(dtype))
          << "debug.printf conversion '%" << conversion
          << "' requires unsigned integer or index scalar, but got "
          << dtype.ToString();
    } else if (conversion == 'u') {
      CHECK(IsPrintfUnsignedIntegerType(dtype) || IsPrintfBoolType(dtype) || IsPrintfIndexType(dtype))
          << "debug.printf conversion '%" << conversion
          << "' requires unsigned integer, bool, or index scalar, but got "
          << dtype.ToString();
    } else {
      CHECK(IsPrintfSignedIntegerType(dtype) || IsPrintfBoolType(dtype) || IsPrintfIndexType(dtype))
          << "debug.printf conversion '%" << conversion
          << "' requires signed integer, bool, or index scalar, but got "
          << dtype.ToString();
    }
  }

  return GetUnknownType();
}

}  // namespace

REGISTER_OP("debug.dump_tensor")
    .set_op_category("DebugOp")
    .set_description("Print a tensor or tensor window for debugging")
    .add_argument("tensor", "Input tensor (TensorType)")
    .add_argument("offsets", "Static offsets per dimension (MakeTuple of ConstInt)")
    .add_argument("shapes", "Static shape per dimension (MakeTuple of ConstInt)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceDebugDumpTensorType(args, kwargs);
    });

REGISTER_OP("debug.dump_tile")
    .set_op_category("DebugOp")
    .set_description("Print a tile or tile window for debugging")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("offsets", "Optional static offsets per dimension (MakeTuple of ConstInt)")
    .add_argument("shapes", "Optional static shape per dimension (MakeTuple of ConstInt)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceDebugDumpTileType(args, kwargs);
    });

REGISTER_OP("debug.printf")
    .set_op_category("DebugOp")
    .set_description("Print scalar values using a compile-time format string")
    .add_argument("scalars", "Scalar arguments consumed by format conversions (variadic)")
    .set_attr<std::string>("format")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceDebugPrintfType(args, kwargs);
    });

REGISTER_OP("debug.trap")
    .set_op_category("DebugOp")
    .set_description("Abort execution by inserting a trap")
    .no_argument()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      (void)args;
      (void)kwargs;
      return GetUnknownType();
    });

}  // namespace ir
}  // namespace pypto
