# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Automatic intra-core pipeline synchronization for PyPTO.

Tracks per-tile pipeline state during AST parsing and detects cross-pipeline
data dependencies (RAW, WAW, WAR).  When a dependency is found, the caller
emits ``sync_src`` / ``sync_dst`` IR ops via the builder.

Key components:
- ``SyncTracker``: per-tile BufferState tracking with ``record_op()``
- ``_OP_TO_PIPE`` / ``_OP_TILE_ACCESS``: operation metadata tables
- ``prescan_loop_backward_deps()``: lightweight AST pre-scan for loops
- ``emit_*`` helpers: emit sync IR through the builder
- ``EventIdAllocator``: per-pipe-pair event ID assignment
- ``TileRegion``: address overlap detection
"""

from .data_structures import (
    BackwardDep,
    BufferState,
    IfBranchSnapshot,
    LoopContext,
    SyncPair,
    TileAccessPattern,
    TileRegion,
)
from .emission import emit_backward_sync_dst, emit_backward_sync_src, emit_sync_pair
from .event_allocator import EventIdAllocator
from .op_metadata import _OP_TILE_ACCESS, _OP_TO_PIPE, get_move_pipe, get_store_pipe
from .prescan import prescan_loop_backward_deps
from .tracker import SyncTracker

__all__ = [
    "BackwardDep",
    "BufferState",
    "EventIdAllocator",
    "IfBranchSnapshot",
    "LoopContext",
    "SyncPair",
    "SyncTracker",
    "TileAccessPattern",
    "TileRegion",
    "_OP_TILE_ACCESS",
    "_OP_TO_PIPE",
    "emit_backward_sync_dst",
    "emit_backward_sync_src",
    "emit_sync_pair",
    "get_move_pipe",
    "get_store_pipe",
    "prescan_loop_backward_deps",
]
