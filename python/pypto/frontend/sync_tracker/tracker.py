# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""SyncTracker: per-tile pipeline state tracker for automatic sync insertion."""

from __future__ import annotations

import copy

from pypto.pypto_core.ir import PipeType

from .data_structures import (
    BufferState,
    IfBranchSnapshot,
    LoopContext,
    SyncPair,
    TileRegion,
)
from .event_allocator import EventIdAllocator


class SyncTracker:
    """Per-tile pipeline state tracker for automatic sync insertion.

    Tile identity is the IR ``Var.name`` string, which is stable throughout
    the kernel because manual ops do not rebind tile variables in scope.

    Args:
        same_pipe_sync: If True, also emit sync for same-pipeline RAW/WAW/WAR
            dependencies (required on dav-2201 / a2 / a3 where the V pipeline
            does not guarantee completion ordering within a single core).
            Set to False for dav-3510 / a5 where hardware handles intra-pipe
            ordering automatically.
    """

    def __init__(self, same_pipe_sync: bool = False) -> None:
        self._same_pipe_sync = same_pipe_sync
        self._buffer_states: dict[str, BufferState] = {}
        self._tile_regions: dict[str, TileRegion] = {}
        self._event_allocator: EventIdAllocator = EventIdAllocator()
        self._loop_context_stack: list[LoopContext] = []
        self._if_branch_stack: list[IfBranchSnapshot] = []
        self._current_loop_depth: int = 0

    # -- tile registration -------------------------------------------------

    def register_tile(self, var_name: str, region: TileRegion) -> None:
        """Register a tile's physical memory region for overlap checking."""
        self._tile_regions[var_name] = region

    # -- accessors ---------------------------------------------------------

    def get_loop_depth(self) -> int:
        """Return the current loop nesting depth."""
        return self._current_loop_depth

    # -- forward sync ------------------------------------------------------

    def _make_sync_pair(self, set_pipe: PipeType, wait_pipe: PipeType, dep_type: str) -> SyncPair:
        """Create a SyncPair with proper event_id.

        V→V pairs on a2/a3 are handled by ``bar_v`` in the emission layer
        (hardware does not support ``set_flag[V, V]``).  They do not
        consume a hardware event register, so event_id is left at 0.
        All other cross-pipe pairs allocate a forward event_id from the
        per-pipe-pair pool.
        """
        if set_pipe == PipeType.V and wait_pipe == PipeType.V:
            return SyncPair(set_pipe, wait_pipe, dep_type, event_id=0)
        eid = self._event_allocator.forward_event_id(set_pipe, wait_pipe)
        return SyncPair(set_pipe, wait_pipe, dep_type, eid)

    def record_op(
        self,
        pipe: PipeType,
        read_tile_names: list[str],
        write_tile_names: list[str],
    ) -> list[SyncPair]:
        """Record an operation and return forward sync pairs to emit.

        Checks read/write tiles against current buffer states for
        cross-pipeline dependencies, deduplicates by ``(set_pipe, wait_pipe)``
        pair, then updates buffer states.

        Args:
            pipe: The pipeline this operation executes on.
            read_tile_names: Variable names of tiles read by this op.
            write_tile_names: Variable names of tiles written by this op.

        Returns:
            List of :class:`SyncPair` to emit **before** this operation.
        """
        emitted: dict[tuple[PipeType, PipeType], SyncPair] = {}

        # Helper: should we emit a sync for this (prev_pipe → current_pipe) dep?
        def _needs_sync(prev_pipe: PipeType, state: BufferState) -> bool:
            if prev_pipe == pipe:
                # Same-pipe sync only needed for PIPE_V on a2/a3
                if not (self._same_pipe_sync and pipe == PipeType.V):
                    return False
            # Already synced (hardware set_flag is per-pipe, covers all tiles)
            if pipe in state.synced_to:
                return False
            return True

        # RAW: reading a tile last written on a (possibly same) pipe
        for name in read_tile_names:
            state = self._buffer_states.get(name)
            if state and state.last_write_pipe is not None and _needs_sync(state.last_write_pipe, state):
                key = (state.last_write_pipe, pipe)
                if key not in emitted:
                    emitted[key] = self._make_sync_pair(*key, "raw")

        # WAW: writing a tile last written on a (possibly same) pipe
        for name in write_tile_names:
            state = self._buffer_states.get(name)
            if state and state.last_write_pipe is not None and _needs_sync(state.last_write_pipe, state):
                key = (state.last_write_pipe, pipe)
                if key not in emitted:
                    emitted[key] = self._make_sync_pair(*key, "waw")

        # WAR: writing a tile last read on a (possibly same) pipe
        for name in write_tile_names:
            state = self._buffer_states.get(name)
            if state:
                for read_pipe in state.last_read_pipes:
                    if read_pipe != pipe or (self._same_pipe_sync and pipe == PipeType.V):
                        key = (read_pipe, pipe)
                        if key not in emitted:
                            emitted[key] = self._make_sync_pair(*key, "war")

        # Address-overlap: check tiles with different names but overlapping memory
        self._check_overlap_deps(pipe, read_tile_names, write_tile_names, emitted)

        # Mark all tiles sharing the same set_pipe as synced to wait_pipe.
        # Hardware set_flag is per-pipe — one sync covers all tiles on that pipe.
        for set_pipe, wait_pipe in emitted:
            for state in self._buffer_states.values():
                if state.last_write_pipe == set_pipe:
                    state.synced_to.add(wait_pipe)

        # Update buffer states
        for name in write_tile_names:
            state = self._buffer_states.setdefault(name, BufferState())
            state.last_write_pipe = pipe
            state.last_read_pipes.clear()
            state.synced_to.clear()  # new write invalidates previous syncs

        for name in read_tile_names:
            state = self._buffer_states.setdefault(name, BufferState())
            state.last_read_pipes.add(pipe)

        # Update LoopContext if inside a loop
        if self._loop_context_stack:
            ctx = self._loop_context_stack[-1]
            for name in read_tile_names + write_tile_names:
                if name not in ctx.first_access:
                    ctx.first_access[name] = pipe
                ctx.last_access[name] = pipe

        return list(emitted.values())

    def _check_overlap_deps(
        self,
        pipe: PipeType,
        read_tile_names: list[str],
        write_tile_names: list[str],
        emitted: dict[tuple[PipeType, PipeType], SyncPair],
    ) -> None:
        """Check for deps via physical address overlap."""
        def _needs_sync(prev_pipe: PipeType) -> bool:
            if prev_pipe == pipe:
                return self._same_pipe_sync and pipe == PipeType.V
            return True

        # RAW overlap
        for read_name in read_tile_names:
            read_region = self._tile_regions.get(read_name)
            if read_region is None:
                continue
            for other_name, other_state in self._buffer_states.items():
                if other_name == read_name:
                    continue
                if other_state.last_write_pipe is None or not _needs_sync(other_state.last_write_pipe):
                    continue
                other_region = self._tile_regions.get(other_name)
                if other_region is None:
                    continue
                if read_region.overlaps(other_region):
                    key = (other_state.last_write_pipe, pipe)
                    if key not in emitted:
                        emitted[key] = self._make_sync_pair(*key, "raw_overlap")

        # WAW overlap
        for write_name in write_tile_names:
            write_region = self._tile_regions.get(write_name)
            if write_region is None:
                continue
            for other_name, other_state in self._buffer_states.items():
                if other_name == write_name:
                    continue
                if other_state.last_write_pipe is None or not _needs_sync(other_state.last_write_pipe):
                    continue
                other_region = self._tile_regions.get(other_name)
                if other_region is None:
                    continue
                if write_region.overlaps(other_region):
                    key = (other_state.last_write_pipe, pipe)
                    if key not in emitted:
                        emitted[key] = self._make_sync_pair(*key, "waw_overlap")

        # WAR overlap
        for write_name in write_tile_names:
            write_region = self._tile_regions.get(write_name)
            if write_region is None:
                continue
            for other_name, other_state in self._buffer_states.items():
                if other_name == write_name:
                    continue
                other_region = self._tile_regions.get(other_name)
                if other_region is None:
                    continue
                if not write_region.overlaps(other_region):
                    continue
                for read_pipe in other_state.last_read_pipes:
                    if _needs_sync(read_pipe):
                        key = (read_pipe, pipe)
                        if key not in emitted:
                            emitted[key] = self._make_sync_pair(*key, "war_overlap")

    # -- pipeline fence (cross-core sync) -----------------------------------

    def pipeline_fence(self) -> None:
        """Clear all buffer states after a pipeline fence.

        Called when the parser encounters ``wait_cross_core``.  While the
        core blocks waiting for the cross-core signal, all previously
        issued pipeline operations complete.  Clearing the states prevents
        the tracker from emitting redundant forward syncs for dependencies
        that were already resolved by the fence.
        """
        self._buffer_states.clear()

    # -- loop state management ---------------------------------------------

    def enter_loop(self) -> None:
        """Push a LoopContext with a snapshot of current buffer states."""
        ctx = LoopContext(
            depth=self._current_loop_depth,
            pre_loop_snapshot=copy.deepcopy(self._buffer_states),
        )
        self._loop_context_stack.append(ctx)
        self._current_loop_depth += 1

    def exit_loop(self) -> LoopContext:
        """Pop and return the LoopContext, restoring pre-loop buffer states.

        After restoring the snapshot, merge back the ``last_write_pipe`` from
        tiles that were written inside the loop body.  This allows post-loop
        code (e.g. ``l0c_store`` after a k-loop) to see the dependency on the
        loop's final write pipe (e.g. ``M``).
        """
        self._current_loop_depth -= 1
        ctx = self._loop_context_stack.pop()
        # Save end-of-loop states before restoring snapshot
        end_of_loop_states = self._buffer_states
        self._buffer_states = ctx.pre_loop_snapshot
        # Merge: tiles written in the loop keep their last_write_pipe
        for name, end_state in end_of_loop_states.items():
            if end_state.last_write_pipe is not None:
                pre_state = self._buffer_states.get(name)
                if pre_state is None or pre_state.last_write_pipe != end_state.last_write_pipe:
                    state = self._buffer_states.setdefault(name, BufferState())
                    state.last_write_pipe = end_state.last_write_pipe
                    state.synced_to.clear()
        return ctx

    # -- if/else branch tracking -------------------------------------------

    def enter_if_branch(self) -> None:
        """Save pre-if buffer states before parsing the then-branch."""
        snapshot = IfBranchSnapshot(
            pre_if_states=copy.deepcopy(self._buffer_states),
        )
        self._if_branch_stack.append(snapshot)

    def enter_else_branch(self) -> None:
        """Save then-states and restore pre-if states for the else-branch."""
        if not self._if_branch_stack:
            return
        snapshot = self._if_branch_stack[-1]
        snapshot.then_states = copy.deepcopy(self._buffer_states)
        self._buffer_states = copy.deepcopy(snapshot.pre_if_states)

    def exit_if(self) -> None:
        """Merge branch states conservatively after an if-statement."""
        if not self._if_branch_stack:
            return
        snapshot = self._if_branch_stack.pop()

        if snapshot.then_states is not None:
            # Has else: merge then_states with current (else-branch final)
            self._buffer_states = self._merge_states(
                snapshot.then_states, self._buffer_states,
            )
        else:
            # No else: merge pre_if_states with current (then-branch final)
            self._buffer_states = self._merge_states(
                snapshot.pre_if_states, self._buffer_states,
            )

    def _merge_states(
        self,
        states_a: dict[str, BufferState],
        states_b: dict[str, BufferState],
    ) -> dict[str, BufferState]:
        """Conservative merge of two possible buffer state worlds.

        For each tile:
        - last_write_pipe: If both agree, keep it. Otherwise None (ambiguous).
        - last_read_pipes: Union of both branches' read pipes.
        """
        all_names = set(states_a) | set(states_b)
        merged: dict[str, BufferState] = {}
        for name in all_names:
            a = states_a.get(name, BufferState())
            b = states_b.get(name, BufferState())

            if a.last_write_pipe == b.last_write_pipe:
                write_pipe = a.last_write_pipe
            else:
                write_pipe = None  # Ambiguous → forces sync on next use

            read_pipes = a.last_read_pipes | b.last_read_pipes
            merged[name] = BufferState(
                last_write_pipe=write_pipe,
                last_read_pipes=set(read_pipes),
            )
        return merged
