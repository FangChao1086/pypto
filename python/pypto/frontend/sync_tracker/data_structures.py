# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Data structures for automatic pipeline synchronization."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from pypto.pypto_core.ir import MemorySpace, PipeType


@dataclass
class BufferState:
    """Per-tile pipeline state.

    Tracks which pipeline last wrote to this tile and which pipelines
    have read from it since that write.
    """

    last_write_pipe: PipeType | None = None
    last_read_pipes: set[PipeType] = field(default_factory=set)
    # Pipes to which this tile's last_write_pipe has already been synced.
    # Used to suppress redundant sync: hardware set_flag is per-pipe,
    # so one sync(MTE2→MTE1) covers ALL tiles written by MTE2.
    synced_to: set[PipeType] = field(default_factory=set)


@dataclass
class SyncPair:
    """A ``sync_src`` / ``sync_dst`` pair to emit before an operation.

    Attributes:
        set_pipe: The pipe that produces the data (issues set_flag).
        wait_pipe: The pipe that consumes the data (issues wait_flag).
        dep_type: Dependency kind.
        event_id: Hardware event register index assigned by EventIdAllocator.
    """

    set_pipe: PipeType
    wait_pipe: PipeType
    dep_type: str
    event_id: int = 0


@dataclass
class BackwardDep:
    """A backward (cross-iteration) dependency detected by AST pre-scan.

    Attributes:
        first_pipe: First pipeline to access the tile in the loop body.
        last_pipe: Last pipeline to access the tile in the loop body.
        tile_name: Variable name of the tile (for diagnostics).
        event_id: Hardware event register index (base ID for multi-slot).
        loop_depth: Nesting level of the loop containing this dep.
        n_slots: Number of buffer slots (1=normal, 2=double-buffer).
            When >1, event IDs ``event_id .. event_id + n_slots - 1`` are used,
            and backward sync is emitted per-slot via if-else on the buffer index.
    """

    first_pipe: PipeType
    last_pipe: PipeType
    tile_name: str
    event_id: int = 0
    loop_depth: int = 0
    n_slots: int = 1


@dataclass
class TileAccessPattern:
    """Describes which positional call-args are tile reads / writes.

    Indices reference the DSL call signature as written by the user in the
    kernel body (arg0 is output for generic ops).
    """

    read_indices: list[int]
    write_indices: list[int]


@dataclass(frozen=True)
class TileRegion:
    """Physical memory region descriptor for address overlap detection.

    When addr_offset is None, the address is a runtime expression
    (not a compile-time constant), and overlap with any tile in the
    same memory_space is conservatively assumed.
    """

    memory_space: MemorySpace
    addr_offset: int | None
    byte_size: int

    def overlaps(self, other: TileRegion) -> bool:
        """True when two regions share at least one byte."""
        if self.memory_space != other.memory_space:
            return False
        if self.addr_offset is None or other.addr_offset is None:
            return True
        return (self.addr_offset < other.addr_offset + other.byte_size
                and other.addr_offset < self.addr_offset + self.byte_size)


@dataclass
class LoopContext:
    """Tracks tile access patterns during loop body parsing.

    Used for:
    1. State restoration: pre_loop_snapshot restores buffer states on exit.
    2. Verification: cross-check prescan backward deps against actual access.
    """

    depth: int
    pre_loop_snapshot: dict[str, BufferState] = field(default_factory=dict)
    first_access: dict[str, PipeType] = field(default_factory=dict)
    last_access: dict[str, PipeType] = field(default_factory=dict)


@dataclass
class IfBranchSnapshot:
    """Saved buffer states for conservative if/else merge.

    Lifecycle:
    1. enter_if_branch():  saves pre_if_states (deep copy of _buffer_states)
    2. enter_else_branch(): saves then_states, restores pre_if_states
    3. exit_if():  merges then_states and else_states conservatively
    """

    pre_if_states: dict[str, BufferState] = field(default_factory=dict)
    then_states: dict[str, BufferState] | None = None
