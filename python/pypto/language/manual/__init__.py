# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not get use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in root of software repository for full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""PyPTO Manual Language Buffer Management and Synchronization.

This module provides buffer management policies and synchronization mechanisms
for FlashAttention kernels on A5 architecture.
"""

from pypto.language.op.manual import (
    TileType,
    make_tile,
    load,
    store,
    add,
    matmul,
)

from .buffer_policy import (
    SyncType,
    PipelineType,
    BuffersPolicyDB,
    BuffersPolicy3buff,
    BuffersPolicy4buff,
    BuffersPolicySingleBuffer,
)

from .sync import (
    SyncEvent,
    inner_core_sync,
    cross_core_sync_forward,
    cross_core_sync_both,
    sync_pipeline,
    allocate_buffer,
    free_buffer,
    record_data_ready,
    wait_data_ready,
)

__all__ = [
    "TileType",
    "make_tile",
    "load",
    "store",
    "add",
    "matmul",
    "SyncType",
    "PipelineType",
    "BuffersPolicyDB",
    "BuffersPolicy3buff",
    "BuffersPolicy4buff",
    "BuffersPolicySingleBuffer",
    "SyncEvent",
    "inner_core_sync",
    "cross_core_sync_forward",
    "cross_core_sync_both",
    "sync_pipeline",
    "allocate_buffer",
    "free_buffer",
    "record_data_ready",
    "wait_data_ready",
]
