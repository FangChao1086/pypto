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
 * @file memory.cpp
 * @brief Memory block operations (get_block_idx, load, store)
 *
 * This file implements memory operations for block-level programming.
 * These operations handle data movement between tensors and unified buffers (tiles).
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

// Helper to get kwargs value with default (uses vector to preserve order)
template <typename T>
T GetKwarg(const std::vector<std::pair<std::string, std::any>>& kwargs, const std::string& key,
           const std::optional<T>& default_value = std::nullopt) {
  for (const auto& [k, v] : kwargs) {
    if (k == key) {
      return AnyCast<T>(v, "kwarg key: " + key);
    }
  }
  if (default_value) {
    return *default_value;
  }
  throw ValueError("Missing kwarg: " + key);
}

TypePtr DeduceBlockGetBlockIdxType(const std::vector<ExprPtr>& args,
                                   const std::vector<std::pair<std::string, std::any>>& kwargs,
                                   const std::string& op_name) {
  CHECK(args.size() == 0) << "The operator " << op_name << " requires no arguments, but got " << args.size();

  // get_block_idx returns UINT64 scalar
  return std::make_shared<ScalarType>(DataType::UINT64);
}

TypePtr DeduceBlockLoadType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                            const std::string& op_name) {
  // load signature: (tensor, offsets_tuple, shapes_tuple, valid_shapes_tuple)
  CHECK(args.size() == 4) << "The operator " << op_name
                          << " requires 4 arguments (tensor, offsets, shapes, valid_shapes), but got "
                          << args.size();

  // First argument must be TensorType
  auto tensor_type = As<TensorType>(args[0]->GetType());
  CHECK(tensor_type) << "The operator " << op_name << " requires first argument to be a TensorType, but got "
                     << args[0]->GetType()->TypeName();

  // Second argument must be TupleType (offsets)
  auto offsets_tuple = As<MakeTuple>(args[1]);
  CHECK(offsets_tuple) << "The operator " << op_name
                       << " requires second argument to be a tuple (offsets), but got "
                       << args[1]->GetType()->TypeName();

  // Third argument must be TupleType (shapes)
  auto shapes_tuple = As<MakeTuple>(args[2]);
  CHECK(shapes_tuple) << "The operator " << op_name
                      << " requires third argument to be a tuple (shapes), but got "
                      << args[2]->GetType()->TypeName();

  // Fourth argument must be TupleType (valid_shapes)
  auto valid_shapes_tuple = As<MakeTuple>(args[3]);
  CHECK(valid_shapes_tuple) << "The operator " << op_name
                            << " requires fourth argument to be a tuple (valid shapes), but got "
                            << args[3]->GetType()->TypeName();

  // Verify offsets, shapes and valid_shapes have same number of dimensions
  CHECK(offsets_tuple->elements_.size() == shapes_tuple->elements_.size())
      << "The operator " << op_name
      << " requires offsets and shapes to have same number of dimensions, but got "
      << offsets_tuple->elements_.size() << " offsets and " << shapes_tuple->elements_.size() << " shapes";
  CHECK(valid_shapes_tuple->elements_.size() == shapes_tuple->elements_.size())
      << "The operator " << op_name
      << " requires valid_shapes and shapes to have same number of dimensions, but got "
      << valid_shapes_tuple->elements_.size() << " valid_shapes and " << shapes_tuple->elements_.size()
      << " shapes";
  CHECK(shapes_tuple->elements_.size() > 0)
      << "The operator " << op_name << " requires at least one dimension, but got empty shapes tuple";

  // load to l1 need nz now
  auto target_memory = GetKwarg<MemorySpace>(kwargs, "target_memory");
  TileView tile_view;
  if (target_memory == MemorySpace::Mat) {
    tile_view.blayout = TileLayout::col_major;
    tile_view.slayout = TileLayout::row_major;
  }

  // Build tile shape from shapes tuple
  std::vector<ExprPtr> tile_shape;
  for (const auto& shape_expr : shapes_tuple->elements_) {
    tile_shape.push_back(shape_expr);
  }

  if (auto last_dim = As<ConstInt>(tile_shape.back()); last_dim && last_dim->value_ == 1) {
    tile_view.blayout = TileLayout::col_major;
  }

  // Build TileView with valid_shape: use valid_shapes arg if provided, else use shapes
  tile_view.valid_shape = valid_shapes_tuple->elements_;

  // Return TileType with same dtype as tensor and TileView containing valid_shape
  return std::make_shared<TileType>(tile_shape, tensor_type->dtype_, std::nullopt, tile_view);
}

TypePtr DeduceBlockStoreType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs,
                             const std::string& op_name) {
  // store signature: (tile, offsets_tuple, shapes_tuple, output_tensor)
  CHECK(args.size() == 4) << "The operator " << op_name
                          << " requires 4 arguments (tile, offsets, shapes, output_tensor), but got "
                          << args.size();

  // First argument must be TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Second argument must be TupleType (offsets)
  auto offsets_tuple = As<MakeTuple>(args[1]);
  CHECK(offsets_tuple) << "The operator " << op_name
                       << " requires second argument to be a tuple (offsets), but got "
                       << args[1]->GetType()->TypeName();

  // Third argument must be TupleType (shapes)
  auto shapes_tuple = As<MakeTuple>(args[2]);
  CHECK(shapes_tuple) << "The operator " << op_name
                      << " requires third argument to be a tuple (shapes), but got "
                      << args[2]->GetType()->TypeName();

  // Verify offsets and shapes have same number of dimensions.
  CHECK(offsets_tuple->elements_.size() == shapes_tuple->elements_.size())
      << "The operator " << op_name
      << " requires offsets and shapes to have same number of dimensions, but got "
      << offsets_tuple->elements_.size() << " offsets and " << shapes_tuple->elements_.size() << " shapes";

  // Fourth argument must be the output tensor
  auto output_tensor_type = As<TensorType>(args[3]->GetType());
  CHECK(output_tensor_type) << "The operator " << op_name
                            << " requires fourth argument to be a TensorType, but got "
                            << args[3]->GetType()->TypeName();

  // store returns the output tensor (same type)
  return output_tensor_type;
}

TypePtr DeduceBlockMoveType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                            const std::string& op_name) {
  // Validate args: expect exactly 1 argument (tile)
  CHECK(args.size() == 1) << "The operator " << op_name << " requires 1 argument, but got " << args.size();

  // Validate first argument is TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Return TileType with same shape and dtype (no explicit MemRef).
  // The actual TMOV variant is determined by the output tile's memory space at codegen time.
  return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_);
}

TypePtr DeduceBlockUbCopyType(const std::vector<ExprPtr>& args,
                              const std::vector<std::pair<std::string, std::any>>& kwargs,
                              const std::string& op_name) {
  // Validate exactly 1 argument
  CHECK(args.size() == 1) << "The operator " << op_name << " requires 1 argument, but got " << args.size();

  // Validate argument is TileType
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  // Return TileType with same shape and dtype
  return std::make_shared<TileType>(tile_type->shape_, tile_type->dtype_);
}

TypePtr DeduceBlockAllocType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs,
                             const std::string& op_name) {
  // alloc signature: (memory_space, addr, size, id)
  // Takes MemRef fields as arguments and returns MemRefType
  CHECK(args.size() == 4) << "The operator " << op_name << " requires exactly 4 arguments, but got "
                          << args.size();

  // Return MemRefType
  return GetMemRefType();
}

TypePtr DeduceBlockPrintType(const std::vector<ExprPtr>& args,
                             const std::vector<std::pair<std::string, std::any>>& kwargs,
                             const std::string& op_name) {
  CHECK(args.size() == 1 || args.size() == 3)
      << "The operator " << op_name << " requires 1 argument (tile) or 3 arguments "
      << "(tile, offsets, shapes), but got " << args.size();
  auto tile_type = As<TileType>(args[0]->GetType());
  CHECK(tile_type) << "The operator " << op_name << " requires first argument to be a TileType, but got "
                   << args[0]->GetType()->TypeName();

  if (args.size() == 3) {
    auto offsets = As<MakeTuple>(args[1]);
    CHECK(offsets) << "The operator " << op_name << " requires second argument to be a MakeTuple (offsets)";

    auto shapes = As<MakeTuple>(args[2]);
    CHECK(shapes) << "The operator " << op_name << " requires third argument to be a MakeTuple (shapes)";

    const size_t rank = tile_type->shape_.size();
    CHECK(rank == 2) << "The operator " << op_name << " currently only supports 2D tile windows, but got rank "
                     << rank;
    CHECK(offsets->elements_.size() == rank)
        << "The operator " << op_name << " offsets count (" << offsets->elements_.size()
        << ") must match tile rank (" << rank << ")";
    CHECK(shapes->elements_.size() == rank)
        << "The operator " << op_name << " shapes count (" << shapes->elements_.size()
        << ") must match tile rank (" << rank << ")";

    for (size_t i = 0; i < rank; ++i) {
      auto offset_scalar = As<ScalarType>(offsets->elements_[i]->GetType());
      CHECK(offset_scalar) << "The operator " << op_name << " offset element " << i
                           << " must be ScalarType, but got "
                           << offsets->elements_[i]->GetType()->TypeName();
      CHECK(offset_scalar->dtype_.IsInt())
          << "The operator " << op_name << " offset element " << i
          << " must have integer dtype, but got " << offset_scalar->dtype_.ToString();
      CHECK(As<ConstInt>(offsets->elements_[i]))
          << "The operator " << op_name << " currently only supports static offsets; axis " << i
          << " is dynamic";

      auto shape_scalar = As<ScalarType>(shapes->elements_[i]->GetType());
      CHECK(shape_scalar) << "The operator " << op_name << " shape element " << i
                          << " must be ScalarType, but got "
                          << shapes->elements_[i]->GetType()->TypeName();
      CHECK(shape_scalar->dtype_.IsInt())
          << "The operator " << op_name << " shape element " << i
          << " must have integer dtype, but got " << shape_scalar->dtype_.ToString();
      auto shape_const = As<ConstInt>(shapes->elements_[i]);
      CHECK(shape_const) << "The operator " << op_name << " currently only supports static shapes; axis " << i
                         << " is dynamic";
      CHECK(shape_const->value_ > 0) << "The operator " << op_name << " shape element " << i
                                     << " must be positive, got " << shape_const->value_;
    }
  }
  return GetUnknownType();
}

TypePtr DeduceBlockCreateTileType(const std::vector<ExprPtr>& args,
                                  const std::vector<std::pair<std::string, std::any>>& kwargs,
                                  const std::string& op_name) {
  // make_tile signature: (shape)
  // TileType requires static compile-time constant shapes
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 argument, but got "
                          << args.size();

  // Extract dtype attribute
  DataType dtype = GetKwarg<DataType>(kwargs, "dtype");

  // First argument must be MakeTuple with static ConstInt elements
  auto shape_tuple = As<MakeTuple>(args[0]);
  CHECK(shape_tuple)
      << "The operator " << op_name
      << " requires first argument to be a MakeTuple expression with static shape values, but got "
      << args[0]->TypeName();

  // Validate all elements are ConstInt (static compile-time constants)
  std::vector<ExprPtr> tile_shape;
  tile_shape.reserve(shape_tuple->elements_.size());

  for (size_t i = 0; i < shape_tuple->elements_.size(); ++i) {
    auto const_int = As<ConstInt>(shape_tuple->elements_[i]);
    CHECK(const_int) << "The operator " << op_name << " shape element " << i
                     << " must be a compile-time constant (ConstInt), but got "
                     << shape_tuple->elements_[i]->TypeName();
    CHECK(const_int->value_ > 0) << "The operator " << op_name << " shape element " << i
                                 << " must be positive, got " << const_int->value_;
    tile_shape.push_back(shape_tuple->elements_[i]);
  }

  CHECK(!tile_shape.empty()) << "The operator " << op_name << " requires non-empty shape";

  TileView tile_view;

  auto valid_shape_tuple = As<MakeTuple>(args[1]);
  if (valid_shape_tuple) tile_view.valid_shape = valid_shape_tuple->elements_;

  int blayout = GetKwarg<int>(kwargs, "blayout", -1);
  if (blayout >= 0) {
    tile_view.blayout = static_cast<TileLayout>(blayout);
  }

  int slayout = GetKwarg<int>(kwargs, "slayout", -1);
  if (slayout >= 0) {
    tile_view.slayout = static_cast<TileLayout>(slayout);
  }

  int fractal = GetKwarg<int>(kwargs, "fractal", -1);
  if (fractal >= 0) {
    tile_view.fractal = static_cast<uint64_t>(fractal);
  }

  int pad = GetKwarg<int>(kwargs, "pad", -1);
  if (pad >= 0) {
    tile_view.pad = static_cast<TilePad>(pad);
  }
  // If explicit memref kwargs are provided (addr + size + id), attach a MemRef to the TileType.
  // This allows the PTO codegen to emit pto.alloc_tile with base_addr directly from the IR,
  // without requiring the init_memref pass.
  MemorySpace target_memory =
      GetKwarg<MemorySpace>(kwargs, "target_memory", std::optional<MemorySpace>(MemorySpace::Vec));

  bool has_memref = false;
  for (const auto& [k, v] : kwargs) {
    if (k == "memref_id") {
      has_memref = true;
      break;
    }
  }
  if (has_memref) {
    int64_t addr_val = GetKwarg<int>(kwargs, "memref_addr");
    int64_t size_val = GetKwarg<int>(kwargs, "memref_size");
    uint64_t id_val  = static_cast<uint64_t>(GetKwarg<int>(kwargs, "memref_id"));
    auto addr_expr = std::make_shared<ConstInt>(addr_val, DataType::INDEX, Span::unknown());
    MemRefPtr memref = std::make_shared<MemRef>(target_memory, addr_expr,
                                                static_cast<uint64_t>(size_val), id_val);
    return std::make_shared<TileType>(tile_shape, dtype, std::optional<MemRefPtr>(memref), tile_view);
  }

  return std::make_shared<TileType>(tile_shape, dtype);
}

TypePtr DeduceBlockFullType(const std::vector<ExprPtr>& args,
                            const std::vector<std::pair<std::string, std::any>>& kwargs,
                            const std::string& op_name) {
  // block.full signature: (shape, value)
  CHECK(args.size() == 2) << "The operator " << op_name << " requires exactly 2 arguments, but got "
                          << args.size();

  // Extract dtype attribute
  DataType dtype = GetKwarg<DataType>(kwargs, "dtype");

  // First argument must be MakeTuple with static ConstInt elements
  auto make_tuple = As<MakeTuple>(args[0]);
  CHECK(make_tuple)
      << "The operator " << op_name
      << " requires first argument to be a MakeTuple expression with static shape values, but got "
      << args[0]->TypeName();

  // Validate all elements are ConstInt (static compile-time constants)
  std::vector<ExprPtr> tile_shape;
  tile_shape.reserve(make_tuple->elements_.size());

  for (size_t i = 0; i < make_tuple->elements_.size(); ++i) {
    auto const_int = As<ConstInt>(make_tuple->elements_[i]);
    CHECK(const_int) << "The operator " << op_name << " shape element " << i
                     << " must be a compile-time constant (ConstInt), but got "
                     << make_tuple->elements_[i]->TypeName();
    CHECK(const_int->value_ > 0) << "The operator " << op_name << " shape element " << i
                                 << " must be positive, got " << const_int->value_;
    tile_shape.push_back(make_tuple->elements_[i]);
  }

  CHECK(!tile_shape.empty()) << "The operator " << op_name << " requires non-empty shape";

  // Second argument must be ConstInt or ConstFloat
  CHECK(As<ConstInt>(args[1]) || As<ConstFloat>(args[1]))
      << "The operator " << op_name
      << " requires second argument to be a constant value (ConstInt or ConstFloat), but got "
      << args[1]->TypeName();

  // Return TileType with the static shape and dtype
  return std::make_shared<TileType>(tile_shape, dtype);
}

// ============================================================================
// Registration Function for Block Memory Operations
// ============================================================================

REGISTER_OP("block.get_block_idx")
    .set_op_category("BlockOp")
    .set_description("Get the current block index")
    .no_argument()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockGetBlockIdxType(args, kwargs, "block.get_block_idx");
    });

REGISTER_OP("block.get_block_num")
    .set_op_category("BlockOp")
    .set_description("Get the current block number")
    .no_argument()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockGetBlockIdxType(args, kwargs, "block.get_block_num");
    });

REGISTER_OP("block.get_subblock_idx")
    .set_op_category("BlockOp")
    .set_description("Get the current subblock index")
    .no_argument()
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockGetBlockIdxType(args, kwargs, "block.get_subblock_idx");
    });

REGISTER_OP("block.index_cast")
    .set_op_category("BlockOp")
    .set_description("Cast scalar to index type")
    .add_argument("idx", "Input scalar (ScalarType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      CHECK(args.size() == 1) << "block.index_cast requires 1 argument, but got " << args.size();
      auto scalar_type = As<ScalarType>(args[0]->GetType());
      CHECK(scalar_type) << "block.index_cast requires argument to be ScalarType, but got "
                       << args[0]->GetType()->TypeName();
      return std::make_shared<ScalarType>(DataType::INDEX);
    });

REGISTER_OP("block.make_tile")
    .set_op_category("BlockOp")
    .set_description("Create a tile")
    .add_argument("shape", "Shape dimensions (TupleType of ScalarType(INT64))")
    .add_argument("valid_shape", "Valid shape dimensions (optional, TupleType)")
    .set_attr<DataType>("dtype")
    .set_attr<MemorySpace>("target_memory")
    .set_attr<int>("memref_addr")
    .set_attr<int>("memref_size")
    .set_attr<int>("memref_id")
    .set_attr<int>("blayout")
    .set_attr<int>("slayout")
    .set_attr<int>("fractal")
    .set_attr<int>("pad")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockCreateTileType(args, kwargs, "block.make_tile");
    });

REGISTER_OP("block.load")
    .set_op_category("BlockOp")
    .set_description("Copy data from tensor to unified buffer (tile)")
    .add_argument("tensor", "Source tensor (TensorType)")
    .add_argument("offsets", "Offsets in each dimension (TupleType of ScalarType)")
    .add_argument("shapes", "Shape of tile in each dimension (TupleType of ScalarType)")
    .add_argument("valid_shapes", "Valid shape of tile in each dimension (TupleType of ScalarType). ")
    .set_attr<MemorySpace>("target_memory")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockLoadType(args, kwargs, "block.load");
    });

REGISTER_OP("block.store")
    .set_op_category("BlockOp")
    .set_description("Copy data from unified buffer (tile) to tensor")
    .add_argument("tile", "Source tile (TileType)")
    .add_argument("offsets", "Offsets in each dimension (TupleType of ScalarType)")
    .add_argument("shapes", "Shape of tile in each dimension (TupleType of ScalarType)")
    .add_argument("output_tensor", "Output tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockStoreType(args, kwargs, "block.store");
    });

REGISTER_OP("block.l0c_store")
    .set_op_category("BlockOp")
    .set_description("Copy data from Acc tile to GM tensor")
    .add_argument("tile", "Source tile (TileType)")
    .add_argument("offsets", "Offsets in each dimension (TupleType of ScalarType)")
    .add_argument("shapes", "Shape of tile in each dimension (TupleType of ScalarType)")
    .add_argument("output_tensor", "Output tensor (TensorType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockStoreType(args, kwargs, "block.l0c_store");
    });

REGISTER_OP("block.move")
    .set_op_category("BlockOp")
    .set_description("Move tile between memory levels. TMOV variant determined by output tile's memory space.")
    .add_argument("tile", "Input tile (TileType)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockMoveType(args, kwargs, "block.move");
    });

REGISTER_OP("block.vec_move")
    .set_op_category("BlockOp")
    .set_description("Copy tile within Vec memory - Vec to Vec only")
    .add_argument("tile", "Input tile (TileType) in Vec memory")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockUbCopyType(args, kwargs, "block.vec_move");
    });

REGISTER_OP("block.alloc")
    .set_op_category("BlockOp")
    .set_description("Allocate memory for a MemRef object")
    .add_argument("memory_space", "Memory space (int enum value)")
    .add_argument("addr", "Starting address expression")
    .add_argument("size", "Size in bytes (scalar)")
    .add_argument("id", "MemRef ID (scalar)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockAllocType(args, kwargs, "block.alloc");
    });

REGISTER_OP("block.full")
    .set_op_category("BlockOp")
    .set_description("Create a tile of specified shape and filling value in UB")
    .add_argument("shape", "Shape dimensions (TupleType of ScalarType(INT64))")
    .add_argument("value", "Filling value (ConstInt or ConstFloat)")
    .set_attr<DataType>("dtype")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockFullType(args, kwargs, "block.full");
    });

REGISTER_OP("block.print")
    .set_op_category("BlockOp")
    .set_description("Print a tile or tile window for debugging")
    .add_argument("tile", "Input tile (TileType)")
    .add_argument("offsets", "Optional static offsets per dimension (MakeTuple of ConstInt)")
    .add_argument("shapes", "Optional static shape per dimension (MakeTuple of ConstInt)")
    .f_deduce_type([](const std::vector<ExprPtr>& args,
                      const std::vector<std::pair<std::string, std::any>>& kwargs) {
      return DeduceBlockPrintType(args, kwargs, "block.print");
    });
}  // namespace ir
}  // namespace pypto
