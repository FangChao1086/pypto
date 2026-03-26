# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Event ID allocator for pipeline synchronization.

Assigns hardware event register indices to sync operations so that
independent dependency chains do not share the same event ID.

Hardware model: each ``(set_pipe, wait_pipe)`` pair has its own independent
namespace of 8 event IDs (0..7).  The allocator tracks usage per pipe pair
so that different pipe pairs can freely reuse the same ID values.
"""

from __future__ import annotations

import warnings

from pypto.pypto_core.ir import PipeType


class EventIdAllocator:
    """Assigns event IDs to sync operations, per hardware pipe pair.

    Hardware constraint: 8 event IDs (0..7) per ``(set_pipe, wait_pipe)`` pair.

    Strategy: each unique ``(set_pipe, wait_pipe)`` pair has its own counter
    starting from 0.  Forward and backward syncs on the same pipe pair share
    the same counter to avoid collisions within the hardware namespace.
    Different pipe pairs have independent counters.
    """

    MAX_EVENTS: int = 8

    def __init__(self) -> None:
        # Per-pipe-pair counter: (set_pipe, wait_pipe) → next available ID
        self._pipe_pair_next: dict[tuple[PipeType, PipeType], int] = {}
        # Cache: (set_pipe, wait_pipe, context_key) → allocated base ID
        self._alloc_cache: dict[tuple, int] = {}

    def forward_event_id(
        self, set_pipe: PipeType, wait_pipe: PipeType, n_slots: int = 1,
    ) -> int:
        """Get or allocate forward sync event ID(s) for a pipe pair.

        Args:
            set_pipe: The pipeline that produces data.
            wait_pipe: The pipeline that consumes data.
            n_slots: Number of consecutive event IDs to allocate.

        Returns:
            Base event ID.  For slot *i*, use ``(base + i) % MAX_EVENTS``.
        """
        cache_key = ("fwd", set_pipe, wait_pipe)
        if cache_key in self._alloc_cache:
            return self._alloc_cache[cache_key]
        base = self._allocate(set_pipe, wait_pipe, n_slots)
        self._alloc_cache[cache_key] = base
        return base

    def backward_event_id(
        self, set_pipe: PipeType, wait_pipe: PipeType, loop_depth: int,
        n_slots: int = 1,
    ) -> int:
        """Get or allocate backward sync event ID(s) for a pipe pair + depth.

        Returns the base event ID.  For slot *i*, use
        ``(base + i) % MAX_EVENTS``.
        """
        cache_key = ("bwd", set_pipe, wait_pipe, loop_depth)
        if cache_key in self._alloc_cache:
            return self._alloc_cache[cache_key]
        base = self._allocate(set_pipe, wait_pipe, n_slots)
        self._alloc_cache[cache_key] = base
        return base

    def _allocate(self, set_pipe: PipeType, wait_pipe: PipeType, n_slots: int) -> int:
        """Allocate *n_slots* consecutive event IDs on a pipe pair."""
        pipe_key = (set_pipe, wait_pipe)
        next_id = self._pipe_pair_next.get(pipe_key, 0)
        base = next_id % self.MAX_EVENTS
        if next_id + n_slots > self.MAX_EVENTS:
            warnings.warn(
                f"Auto-sync: event ID pool for ({set_pipe.name}, {wait_pipe.name}) exhausted "
                f"({next_id + n_slots} IDs > {self.MAX_EVENTS} events). "
                f"Wrapping around — may cause false synchronization.",
                stacklevel=3,
            )
        self._pipe_pair_next[pipe_key] = next_id + n_slots
        return base

    def reset(self) -> None:
        """Reset all allocations. Called per-kernel."""
        self._pipe_pair_next.clear()
        self._alloc_cache.clear()
