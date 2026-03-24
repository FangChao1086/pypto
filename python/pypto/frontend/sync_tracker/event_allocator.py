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
"""

from __future__ import annotations

import warnings

from pypto.pypto_core.ir import PipeType


class EventIdAllocator:
    """Assigns event IDs to sync operations.

    Hardware constraint: 8 event IDs (0..7).

    Strategy:
    - Forward syncs: each unique (set_pipe, wait_pipe) pair gets a stable ID.
    - Backward syncs: each unique (set_pipe, wait_pipe, loop_depth) triple
      gets a stable ID from a separate pool.
    - Both pools wrap around modulo MAX_EVENTS if exhausted.
    """

    MAX_EVENTS: int = 8

    def __init__(self) -> None:
        self._forward_map: dict[tuple[PipeType, PipeType], int] = {}
        self._backward_map: dict[tuple[PipeType, PipeType, int], int] = {}
        self._forward_next: int = 0
        self._backward_next: int = 0

    def forward_event_id(self, set_pipe: PipeType, wait_pipe: PipeType) -> int:
        """Get or allocate a forward sync event ID for a pipe pair."""
        key = (set_pipe, wait_pipe)
        if key not in self._forward_map:
            eid = self._forward_next % self.MAX_EVENTS
            if self._forward_next >= self.MAX_EVENTS:
                warnings.warn(
                    f"Auto-sync: forward event ID pool exhausted "
                    f"({self._forward_next + 1} pipe pairs > {self.MAX_EVENTS} events). "
                    f"Wrapping around — may cause false synchronization.",
                    stacklevel=2,
                )
            self._forward_map[key] = eid
            self._forward_next += 1
        return self._forward_map[key]

    def backward_event_id(
        self, set_pipe: PipeType, wait_pipe: PipeType, loop_depth: int,
    ) -> int:
        """Get or allocate a backward sync event ID for a pipe pair + depth."""
        key = (set_pipe, wait_pipe, loop_depth)
        if key not in self._backward_map:
            eid = self._backward_next % self.MAX_EVENTS
            if self._backward_next >= self.MAX_EVENTS:
                warnings.warn(
                    f"Auto-sync: backward event ID pool exhausted "
                    f"({self._backward_next + 1} triples > {self.MAX_EVENTS} events). "
                    f"Wrapping around — may cause false synchronization.",
                    stacklevel=2,
                )
            self._backward_map[key] = eid
            self._backward_next += 1
        return self._backward_map[key]

    def reset(self) -> None:
        """Reset all allocations. Called per-kernel."""
        self._forward_map.clear()
        self._backward_map.clear()
        self._forward_next = 0
        self._backward_next = 0
