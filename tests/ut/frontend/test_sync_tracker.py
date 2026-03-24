# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Unit tests for SyncTracker and sync_tracker module."""

import pytest
from pypto.pypto_core.ir import MemorySpace, PipeType

from pypto.frontend.sync_tracker import (
    BufferState,
    EventIdAllocator,
    LoopContext,
    SyncPair,
    SyncTracker,
    TileRegion,
    _OP_TILE_ACCESS,
    _OP_TO_PIPE,
    get_store_pipe,
)


class TestBufferState:
    """Test BufferState defaults."""

    def test_default_state(self):
        state = BufferState()
        assert state.last_write_pipe is None
        assert state.last_read_pipes == set()


class TestSyncTracker:
    """Test SyncTracker.record_op for forward dependency detection."""

    def test_no_sync_first_op(self):
        tracker = SyncTracker()
        pairs = tracker.record_op(PipeType.MTE2, [], ["tile_a"])
        assert pairs == []

    def test_no_sync_same_pipe(self):
        """Same-pipe ops: no sync by default (same_pipe_sync=False)."""
        tracker = SyncTracker()  # default: same_pipe_sync=False
        tracker.record_op(PipeType.V, [], ["tile_a"])
        pairs = tracker.record_op(PipeType.V, ["tile_a"], [])
        assert pairs == []

    def test_same_pipe_sync_enabled(self):
        """same_pipe_sync=True: V→V RAW emits a V→V sync (required on a2/a3)."""
        tracker = SyncTracker(same_pipe_sync=True)
        tracker.record_op(PipeType.V, [], ["tile_a"])   # write on V
        pairs = tracker.record_op(PipeType.V, ["tile_a"], [])  # read on V
        assert len(pairs) == 1
        assert pairs[0].set_pipe == PipeType.V
        assert pairs[0].wait_pipe == PipeType.V
        assert pairs[0].dep_type == "raw"

    def test_same_pipe_sync_cross_pipe_unaffected(self):
        """same_pipe_sync=True does not affect cross-pipe sync detection."""
        tracker = SyncTracker(same_pipe_sync=True)
        tracker.record_op(PipeType.MTE2, [], ["tile_a"])
        pairs = tracker.record_op(PipeType.V, ["tile_a"], [])
        assert len(pairs) == 1
        assert pairs[0].set_pipe == PipeType.MTE2
        assert pairs[0].wait_pipe == PipeType.V

    def test_raw_detection(self):
        """Write on MTE2, then read on V → RAW sync."""
        tracker = SyncTracker()
        tracker.record_op(PipeType.MTE2, [], ["tile_a"])  # write on MTE2
        pairs = tracker.record_op(PipeType.V, ["tile_a"], [])  # read on V
        assert len(pairs) == 1
        assert pairs[0].set_pipe == PipeType.MTE2
        assert pairs[0].wait_pipe == PipeType.V
        assert pairs[0].dep_type == "raw"

    def test_waw_detection(self):
        """Write on MTE2, then write on V → WAW sync."""
        tracker = SyncTracker()
        tracker.record_op(PipeType.MTE2, [], ["tile_a"])  # write on MTE2
        pairs = tracker.record_op(PipeType.V, [], ["tile_a"])  # write on V
        assert len(pairs) == 1
        assert pairs[0].set_pipe == PipeType.MTE2
        assert pairs[0].wait_pipe == PipeType.V
        assert pairs[0].dep_type == "waw"

    def test_war_detection(self):
        """Read on V, then write on MTE2 → WAR sync."""
        tracker = SyncTracker()
        tracker.record_op(PipeType.V, ["tile_a"], [])  # read on V
        pairs = tracker.record_op(PipeType.MTE2, [], ["tile_a"])  # write on MTE2
        assert len(pairs) == 1
        assert pairs[0].set_pipe == PipeType.V
        assert pairs[0].wait_pipe == PipeType.MTE2
        assert pairs[0].dep_type == "war"

    def test_no_dep_different_tiles(self):
        """Different tiles → no dependency."""
        tracker = SyncTracker()
        tracker.record_op(PipeType.MTE2, [], ["tile_a"])  # write tile_a on MTE2
        pairs = tracker.record_op(PipeType.V, ["tile_b"], [])  # read tile_b on V
        assert pairs == []

    def test_deduplication(self):
        """Multiple tiles with same pipe pair → single SyncPair."""
        tracker = SyncTracker()
        tracker.record_op(PipeType.MTE2, [], ["tile_a", "tile_b"])
        pairs = tracker.record_op(PipeType.V, ["tile_a", "tile_b"], [])
        # Both tiles produce MTE2→V RAW, but deduped to one pair
        assert len(pairs) == 1

    def test_multiple_pipe_pairs(self):
        """Different tiles on different pipes → multiple SyncPairs."""
        tracker = SyncTracker()
        tracker.record_op(PipeType.MTE2, [], ["tile_a"])  # write on MTE2
        tracker.record_op(PipeType.M, [], ["tile_b"])  # write on M
        pairs = tracker.record_op(PipeType.V, ["tile_a", "tile_b"], [])
        # tile_a: MTE2→V, tile_b: M→V → two distinct pairs
        assert len(pairs) == 2
        pipe_pairs = {(p.set_pipe, p.wait_pipe) for p in pairs}
        assert (PipeType.MTE2, PipeType.V) in pipe_pairs
        assert (PipeType.M, PipeType.V) in pipe_pairs

    def test_write_clears_readers(self):
        """A write clears previous read_pipes so stale reads don't produce WAR."""
        tracker = SyncTracker()
        tracker.record_op(PipeType.V, ["tile_a"], [])  # read on V
        tracker.record_op(PipeType.V, [], ["tile_a"])  # write on V → clears reads
        # Now a write from MTE2 should NOT produce WAR for the old V read
        pairs = tracker.record_op(PipeType.MTE2, [], ["tile_a"])
        # Should only have WAW (V→MTE2), NOT WAR
        assert all(p.dep_type != "war" for p in pairs)

    def test_loop_enter_exit_restores_state(self):
        """enter_loop / exit_loop restores pre-loop buffer state."""
        tracker = SyncTracker()
        tracker.record_op(PipeType.MTE2, [], ["tile_a"])  # write on MTE2

        tracker.enter_loop()
        tracker.record_op(PipeType.V, [], ["tile_a"])  # write on V inside loop
        loop_ctx = tracker.exit_loop()

        # After exit_loop, state is restored: tile_a was last written by MTE2
        pairs = tracker.record_op(PipeType.V, ["tile_a"], [])
        assert len(pairs) == 1
        assert pairs[0].set_pipe == PipeType.MTE2
        # LoopContext should have captured access info
        assert isinstance(loop_ctx, LoopContext)

    def test_event_id_assignment(self):
        """Different pipe pairs get different event IDs."""
        tracker = SyncTracker()
        tracker.record_op(PipeType.MTE2, [], ["tile_a"])
        tracker.record_op(PipeType.M, [], ["tile_b"])
        pairs = tracker.record_op(PipeType.V, ["tile_a", "tile_b"], [])
        assert len(pairs) == 2
        event_ids = {p.event_id for p in pairs}
        # Two different pipe pairs should get different event IDs
        assert len(event_ids) == 2

    def test_same_pipe_pair_same_event_id(self):
        """Same pipe pair reuses the same event ID."""
        tracker = SyncTracker()
        tracker.record_op(PipeType.MTE2, [], ["tile_a"])
        pairs1 = tracker.record_op(PipeType.V, ["tile_a"], [])
        tracker.record_op(PipeType.MTE2, [], ["tile_b"])
        pairs2 = tracker.record_op(PipeType.V, ["tile_b"], [])
        assert pairs1[0].event_id == pairs2[0].event_id


class TestEventIdAllocator:
    """Test EventIdAllocator."""

    def test_same_pair_returns_same_id(self):
        alloc = EventIdAllocator()
        id1 = alloc.forward_event_id(PipeType.MTE2, PipeType.V)
        id2 = alloc.forward_event_id(PipeType.MTE2, PipeType.V)
        assert id1 == id2

    def test_different_pairs_return_different_ids(self):
        alloc = EventIdAllocator()
        id1 = alloc.forward_event_id(PipeType.MTE2, PipeType.V)
        id2 = alloc.forward_event_id(PipeType.M, PipeType.V)
        assert id1 != id2

    def test_wrap_around_at_max_events(self):
        alloc = EventIdAllocator()
        ids = []
        for i in range(10):
            # Create unique pipe pairs using available PipeType values
            pipes = [PipeType.MTE2, PipeType.V, PipeType.M, PipeType.FIX, PipeType.MTE1]
            set_pipe = pipes[i % len(pipes)]
            wait_pipe = pipes[(i + 1) % len(pipes)]
            ids.append(alloc.forward_event_id(set_pipe, wait_pipe))
        # All IDs should be in range [0, MAX_EVENTS)
        assert all(0 <= eid < EventIdAllocator.MAX_EVENTS for eid in ids)

    def test_backward_uses_separate_pool(self):
        alloc = EventIdAllocator()
        fwd = alloc.forward_event_id(PipeType.MTE2, PipeType.V)
        bwd = alloc.backward_event_id(PipeType.MTE2, PipeType.V, 0)
        # Both pools start at 0, so the first allocation in each returns 0
        assert fwd == 0
        assert bwd == 0

    def test_different_depths_get_different_ids(self):
        alloc = EventIdAllocator()
        id0 = alloc.backward_event_id(PipeType.V, PipeType.MTE2, 0)
        id1 = alloc.backward_event_id(PipeType.V, PipeType.MTE2, 1)
        assert id0 != id1

    def test_reset_clears_all(self):
        alloc = EventIdAllocator()
        alloc.forward_event_id(PipeType.MTE2, PipeType.V)
        alloc.backward_event_id(PipeType.V, PipeType.MTE2, 0)
        alloc.reset()
        # After reset, first allocation starts from 0 again
        assert alloc.forward_event_id(PipeType.M, PipeType.V) == 0
        assert alloc.backward_event_id(PipeType.M, PipeType.V, 0) == 0


class TestTileRegion:
    """Test TileRegion overlap detection."""

    def test_same_space_overlapping(self):
        a = TileRegion(MemorySpace.Vec, 0, 128)
        b = TileRegion(MemorySpace.Vec, 64, 128)
        assert a.overlaps(b)
        assert b.overlaps(a)

    def test_same_space_non_overlapping(self):
        a = TileRegion(MemorySpace.Vec, 0, 128)
        b = TileRegion(MemorySpace.Vec, 128, 128)
        assert not a.overlaps(b)
        assert not b.overlaps(a)

    def test_different_space_never_overlaps(self):
        a = TileRegion(MemorySpace.Vec, 0, 128)
        b = TileRegion(MemorySpace.Mat, 0, 128)
        assert not a.overlaps(b)

    def test_none_addr_conservative_overlap(self):
        a = TileRegion(MemorySpace.Vec, None, 128)
        b = TileRegion(MemorySpace.Vec, 64, 128)
        assert a.overlaps(b)
        assert b.overlaps(a)

    def test_adjacent_ranges_no_overlap(self):
        a = TileRegion(MemorySpace.Vec, 0, 64)
        b = TileRegion(MemorySpace.Vec, 64, 64)
        assert not a.overlaps(b)


class TestOverlapDetection:
    """Test SyncTracker overlap detection in record_op."""

    def test_overlapping_tiles_trigger_sync(self):
        tracker = SyncTracker()
        tracker.register_tile("tile_a", TileRegion(MemorySpace.Vec, 0, 128))
        tracker.register_tile("tile_b", TileRegion(MemorySpace.Vec, 64, 128))
        tracker.record_op(PipeType.MTE2, [], ["tile_a"])  # write tile_a
        # read tile_b (overlaps tile_a) on different pipe
        pairs = tracker.record_op(PipeType.V, ["tile_b"], [])
        # Should detect RAW overlap dep: MTE2→V
        overlap_pairs = [p for p in pairs if "overlap" in p.dep_type]
        assert len(overlap_pairs) >= 1

    def test_non_overlapping_tiles_no_sync(self):
        tracker = SyncTracker()
        tracker.register_tile("tile_a", TileRegion(MemorySpace.Vec, 0, 128))
        tracker.register_tile("tile_b", TileRegion(MemorySpace.Vec, 128, 128))
        tracker.record_op(PipeType.MTE2, [], ["tile_a"])
        pairs = tracker.record_op(PipeType.V, ["tile_b"], [])
        overlap_pairs = [p for p in pairs if "overlap" in p.dep_type]
        assert len(overlap_pairs) == 0

    def test_dynamic_addr_conservative(self):
        tracker = SyncTracker()
        tracker.register_tile("tile_a", TileRegion(MemorySpace.Vec, None, 128))
        tracker.register_tile("tile_b", TileRegion(MemorySpace.Vec, 64, 128))
        tracker.record_op(PipeType.MTE2, [], ["tile_a"])
        pairs = tracker.record_op(PipeType.V, ["tile_b"], [])
        overlap_pairs = [p for p in pairs if "overlap" in p.dep_type]
        assert len(overlap_pairs) >= 1


class TestIfBranchTracking:
    """Test if/else branch state tracking and merging."""

    def test_then_only_no_else_merge(self):
        """If without else: merge pre-if with then-final."""
        tracker = SyncTracker()
        tracker.record_op(PipeType.MTE2, [], ["tile_a"])  # write MTE2

        tracker.enter_if_branch()
        tracker.record_op(PipeType.V, [], ["tile_a"])  # then: write V
        tracker.exit_if()  # merge: MTE2 vs V → None (ambiguous)

        # After merge, write_pipe is ambiguous (None)
        state = tracker._buffer_states.get("tile_a")
        assert state is not None
        assert state.last_write_pipe is None

    def test_then_else_same_pipe_preserves(self):
        """If both branches write on same pipe, keep it."""
        tracker = SyncTracker()

        tracker.enter_if_branch()
        tracker.record_op(PipeType.V, [], ["tile_a"])  # then: write V
        tracker.enter_else_branch()
        tracker.record_op(PipeType.V, [], ["tile_a"])  # else: write V
        tracker.exit_if()  # merge: V vs V → V

        state = tracker._buffer_states.get("tile_a")
        assert state is not None
        assert state.last_write_pipe == PipeType.V

    def test_then_else_different_pipe_clears(self):
        """If branches write on different pipes, result is ambiguous."""
        tracker = SyncTracker()

        tracker.enter_if_branch()
        tracker.record_op(PipeType.MTE2, [], ["tile_a"])  # then: write MTE2
        tracker.enter_else_branch()
        tracker.record_op(PipeType.V, [], ["tile_a"])  # else: write V
        tracker.exit_if()  # merge: MTE2 vs V → None

        state = tracker._buffer_states.get("tile_a")
        assert state is not None
        assert state.last_write_pipe is None

    def test_nested_if_merge(self):
        """Nested if-statements should merge correctly."""
        tracker = SyncTracker()
        tracker.record_op(PipeType.MTE2, [], ["tile_a"])

        tracker.enter_if_branch()  # outer if
        tracker.enter_if_branch()  # inner if
        tracker.record_op(PipeType.V, [], ["tile_a"])
        tracker.exit_if()  # inner exit: MTE2 vs V → None
        tracker.exit_if()  # outer exit: MTE2 vs None → None

        state = tracker._buffer_states.get("tile_a")
        assert state.last_write_pipe is None


class TestLoopContext:
    """Test LoopContext first/last access tracking."""

    def test_first_last_access_tracking(self):
        tracker = SyncTracker()
        tracker.enter_loop()
        tracker.record_op(PipeType.MTE2, [], ["tile_a"])  # first: MTE2
        tracker.record_op(PipeType.V, ["tile_a"], [])  # last: V
        loop_ctx = tracker.exit_loop()

        assert loop_ctx.first_access["tile_a"] == PipeType.MTE2
        assert loop_ctx.last_access["tile_a"] == PipeType.V

    def test_nested_loop_independent_contexts(self):
        tracker = SyncTracker()
        tracker.enter_loop()  # outer
        tracker.record_op(PipeType.MTE2, [], ["tile_a"])
        tracker.enter_loop()  # inner
        tracker.record_op(PipeType.V, [], ["tile_b"])
        inner_ctx = tracker.exit_loop()
        outer_ctx = tracker.exit_loop()

        assert "tile_b" in inner_ctx.first_access
        assert "tile_a" in outer_ctx.first_access
        # tile_b is in inner context but also in outer context
        # (inner ops contribute to outer)

    def test_loop_depth_tracking(self):
        tracker = SyncTracker()
        assert tracker.get_loop_depth() == 0
        tracker.enter_loop()
        assert tracker.get_loop_depth() == 1
        tracker.enter_loop()
        assert tracker.get_loop_depth() == 2
        tracker.exit_loop()
        assert tracker.get_loop_depth() == 1
        tracker.exit_loop()
        assert tracker.get_loop_depth() == 0


class TestOpMappingConsistency:
    """Verify _OP_TO_PIPE and _OP_TILE_ACCESS are consistent."""

    def test_every_pipe_op_has_access_pattern(self):
        """Every op in _OP_TO_PIPE should have a corresponding _OP_TILE_ACCESS entry."""
        for op_name in _OP_TO_PIPE:
            assert op_name in _OP_TILE_ACCESS, f"Missing _OP_TILE_ACCESS for {op_name}"

    def test_move_not_in_pipe_map(self):
        """'move' uses get_move_pipe(), so it's NOT in _OP_TO_PIPE."""
        assert "move" not in _OP_TO_PIPE

    def test_store_not_in_pipe_map(self):
        """'store'/'store_tile' use get_store_pipe(), so not in _OP_TO_PIPE."""
        assert "store" not in _OP_TO_PIPE
        assert "store_tile" not in _OP_TO_PIPE

    def test_l0c_store_is_fix(self):
        """l0c_store always stores from ACC → PIPE_FIX."""
        assert _OP_TO_PIPE["l0c_store"] == PipeType.FIX

    def test_move_has_access_pattern(self):
        """'move' should still have an access pattern."""
        assert "move" in _OP_TILE_ACCESS

    def test_store_has_access_pattern(self):
        """'store'/'store_tile' should still have access patterns."""
        assert "store" in _OP_TILE_ACCESS
        assert "store_tile" in _OP_TILE_ACCESS

    def test_dsl_convention_binary(self):
        """Binary tile×tile ops: DSL arg0=out(write), arg1/2=inputs(read)."""
        for op in ("add", "sub", "mul", "div"):
            pattern = _OP_TILE_ACCESS[op]
            assert 0 in pattern.write_indices, f"{op}: arg0 should be write (out)"
            assert 1 in pattern.read_indices, f"{op}: arg1 should be read (lhs)"
            assert 2 in pattern.read_indices, f"{op}: arg2 should be read (rhs)"

    def test_dsl_convention_unary(self):
        """Unary ops: DSL arg0=out(write), arg1=input(read)."""
        for op in ("neg", "exp", "sqrt"):
            pattern = _OP_TILE_ACCESS[op]
            assert 0 in pattern.write_indices, f"{op}: arg0 should be write (out)"
            assert 1 in pattern.read_indices, f"{op}: arg1 should be read (tile)"

    def test_dsl_convention_load(self):
        """load: arg0=out tile (write), no tile reads."""
        pattern = _OP_TILE_ACCESS["load"]
        assert 0 in pattern.write_indices
        assert pattern.read_indices == []

    def test_dsl_convention_store(self):
        """store: arg1=source tile (read), no tile writes."""
        pattern = _OP_TILE_ACCESS["store"]
        assert 1 in pattern.read_indices
        assert pattern.write_indices == []


class TestGetStorePipe:
    """Test get_store_pipe() dynamic pipe resolution."""

    def test_vec_store_is_mte3(self):
        """Store from Vec (UB) → PIPE_MTE3."""
        assert get_store_pipe(MemorySpace.Vec) == PipeType.MTE3

    def test_acc_store_is_fix(self):
        """Store from Acc (L0C) → PIPE_FIX."""
        assert get_store_pipe(MemorySpace.Acc) == PipeType.FIX

    def test_none_defaults_to_mte3(self):
        """Unknown memory space → PIPE_MTE3 (most common store)."""
        assert get_store_pipe(None) == PipeType.MTE3

    def test_mat_defaults_to_mte3(self):
        """Store from Mat → PIPE_MTE3 (conservative)."""
        assert get_store_pipe(MemorySpace.Mat) == PipeType.MTE3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
