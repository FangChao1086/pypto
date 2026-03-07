# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Buffer management types for PyPTO."""

from enum import Enum


class BufferPolicy(Enum):
    """Buffer策略枚举"""
    SINGLE = 0   # 单buffer（无乒乓）
    DOUBLE = 1   # 双buffer（ping-pong轮转）
    TRIPLE = 2   # 3 buffer（轮转）
    QUAD = 3     # 4 buffer（轮转）


class SyncType(Enum):
    """同步类型枚举"""
    NO_SYNC = 0                  # 不同步
    INNER_CORE_SYNC = 1          # 核内同步（需要事件ID）
    CROSS_CORE_SYNC_FORWARD = 2  # 核间同步（单向，需要事件ID）
    CROSS_CORE_SYNC_BOTH = 3     # 核间同步（双向，需要事件ID）


class HardEvent(Enum):
    """硬件事件类型"""
    L1_P2C = 0   # L1：生产者到消费者（MTE2_MTE1）
    L1_C2P = 1   # L1：消费者到生产者（MTE1_MTE2）
    L0_P2C = 2   # L0A/L0B：生产者到消费者（MTE1_M）
    L0_C2P = 3   # L0A/L0B：消费者到生产者（M_MTE1）
    ACC_P2C = 4  # L0C：生产者到消费者（M_FIX）
    ACC_C2P = 5  # L0C：消费者到生产者（FIX_M）
