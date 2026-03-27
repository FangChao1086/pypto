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
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {

namespace {

// Helper to deduce UnknownType (for ops with no return value)
TypePtr DeduceUnknownType(const std::vector<ExprPtr>& args,
                          const std::vector<std::pair<std::string, std::any>>& kwargs) {
  return GetUnknownType();
}

}  // namespace

// ============================================================================
// Registration Function for Sync Operations
// ============================================================================

// Register system.sync_src (Set Flag)
// Attributes: set_pipe, wait_pipe, event_id
REGISTER_OP("system.sync_src")
    .set_description("Send a synchronization signal (Set Flag)")
    .set_op_category("SyncOp")
    .no_argument()
    .set_attr<int>("set_pipe")
    .set_attr<int>("wait_pipe")
    .set_attr<int>("event_id")
    .f_deduce_type(DeduceUnknownType);

// Register system.sync_dst (Wait Flag)
// Attributes: set_pipe, wait_pipe, event_id
REGISTER_OP("system.sync_dst")
    .set_description("Wait for a synchronization signal (Wait Flag)")
    .set_op_category("SyncOp")
    .no_argument()
    .set_attr<int>("set_pipe")
    .set_attr<int>("wait_pipe")
    .set_attr<int>("event_id")
    .f_deduce_type(DeduceUnknownType);

// Register system.sync_src_dyn (Set Flag, dynamic event_id)
REGISTER_OP("system.sync_src_dyn")
    .set_description("Send a synchronization signal with dynamic event_id (Set Flag)")
    .set_op_category("SyncOp")
    .add_argument("event_id", "Dynamic event ID (ScalarType INDEX)")
    .set_attr<int>("set_pipe")
    .set_attr<int>("wait_pipe")
    .f_deduce_type(DeduceUnknownType);

// Register system.sync_dst_dyn (Wait Flag, dynamic event_id)
REGISTER_OP("system.sync_dst_dyn")
    .set_description("Wait for a synchronization signal with dynamic event_id (Wait Flag)")
    .set_op_category("SyncOp")
    .add_argument("event_id", "Dynamic event ID (ScalarType INDEX)")
    .set_attr<int>("set_pipe")
    .set_attr<int>("wait_pipe")
    .f_deduce_type(DeduceUnknownType);

// Register system.bar_v (Vector Barrier)
// Attributes: None
REGISTER_OP("system.bar_v")
    .set_description("Vector unit barrier")
    .set_op_category("SyncOp")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);

// Register system.bar_m (Matrix Barrier)
// Attributes: None
REGISTER_OP("system.bar_m")
    .set_description("Matrix unit barrier")
    .set_op_category("SyncOp")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);

// Register system.bar_all (Global Barrier)
// Attributes: None
REGISTER_OP("system.bar_all")
    .set_description("Global barrier synchronization")
    .set_op_category("SyncOp")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);

// Register system.set_cross_core (Set Cross Core Flag)
// Attributes: pipe, event_id
REGISTER_OP("system.set_cross_core")
    .set_description("Set for a synchronization signal (Cross core)")
    .set_op_category("SyncOp")
    .no_argument()
    .set_attr<int>("pipe")
    .set_attr<int>("event_id")
    .f_deduce_type(DeduceUnknownType);

// Register system.wait_cross_core (Wait Cross Core Flag)
// Attributes: pipe, event_id
REGISTER_OP("system.wait_cross_core")
    .set_description("Wait for a synchronization signal (Cross core)")
    .set_op_category("SyncOp")
    .no_argument()
    .set_attr<int>("pipe")
    .set_attr<int>("event_id")
    .f_deduce_type(DeduceUnknownType);

// Register system.set_cross_core_dyn (Set Cross Core Flag, dynamic event_id)
REGISTER_OP("system.set_cross_core_dyn")
    .set_description("Cross-core sync set with dynamic event_id")
    .set_op_category("SyncOp")
    .add_argument("event_id", "Dynamic event ID (ScalarType INDEX)")
    .set_attr<int>("pipe")
    .f_deduce_type(DeduceUnknownType);

// Register system.wait_cross_core_dyn (Wait Cross Core Flag, dynamic event_id)
REGISTER_OP("system.wait_cross_core_dyn")
    .set_description("Cross-core sync wait with dynamic event_id")
    .set_op_category("SyncOp")
    .add_argument("event_id", "Dynamic event ID (ScalarType INDEX)")
    .set_attr<int>("pipe")
    .f_deduce_type(DeduceUnknownType);

// Register system.sync_all (Global Core Synchronization)
// Attributes: set_pipe, wait_pipe
REGISTER_OP("system.sync_all")
    .set_description("Global core synchronization")
    .set_op_category("SyncOp")
    .no_argument()
    .set_attr<bool>("aiv_only")
    .set_attr<int>("trigger_pipe")
    .set_attr<int>("wait_pipe")
    .f_deduce_type(DeduceUnknownType);

// Register manual.sync (Manual synchronization for buffer management)
// Attributes: sync_type, event_id, direction (optional), pipeline (optional)
REGISTER_OP("manual.sync")
    .set_description("Manual synchronization operation for buffer management in FlashAttention")
    .set_op_category("ManualOp")
    .no_argument()
    .set_attr<std::string>("sync_type")
    .set_attr<int>("event_id")
    .set_attr<std::string>("direction")
    .set_attr<std::string>("pipeline")
    .f_deduce_type(DeduceUnknownType);

}  // namespace ir
}  // namespace pypto
