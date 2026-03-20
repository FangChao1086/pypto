# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not get use this file except in compliance with License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Synchronization management for FlashAttention.

This module provides synchronization mechanisms similar to ops-transformer's
sync management, supporting inner core sync and cross-core sync with
forward and bidirectional modes.
"""

from typing import Optional, Union

from pypto.pypto_core import ir as _ir_core
from pypto.pypto_core.ir import Span

from .buffer_policy import SyncType, PipelineType


class SyncEvent:
    """Synchronization event for cross-core synchronization.
    
    This class manages cross-core synchronization events with event IDs
    and synchronization types. It provides methods to record (producer
    notifies consumer) and allocate (consumer waits for producer) events.
    
    Example::
    
        event = SyncEvent(event_id=0, sync_type=SyncType.CROSS_CORE_SYNC_BOTH)
        
        # Producer side
        event.record()
        
        # Consumer side
        event.allocate()
    """
    
    def __init__(self, event_id: int, sync_type: SyncType):
        """Initialize a sync event.
        
        Args:
            event_id: Event identifier for synchronization
            sync_type: Type of synchronization
        """
        self.event_id = event_id
        self.sync_type = sync_type
    
    def record(self):
        """Record sync event (producer notifies consumer).
        
        This is called by the producer after completing computation
        to notify the consumer that data is ready.
        """
        _sync_op(self.sync_type, self.event_id, direction="record")
    
    def allocate(self):
        """Allocate sync event (consumer waits for producer).
        
        This is called by the consumer to wait for the producer
        to complete computation and make data available.
        """
        _sync_op(self.sync_type, self.event_id, direction="allocate")
    
    def wait(self):
        """Wait for sync event.
        
        This is an alias for allocate() for convenience.
        """
        self.allocate()


def inner_core_sync():
    """Inner core synchronization.
    
    Ensures all operations within the same core are completed before
    proceeding. This is useful for ordering operations within a core.
    
    Example::
    
        plm.matmul(a, b, out=c)
        inner_core_sync()
        plm.store(c, [0, 0], [64, 64], output)
    """
    _sync_op(SyncType.INNER_CORE_SYNC, event_id=0)


def cross_core_sync_forward(event_id: int):
    """Cross-core forward synchronization.
    
    Producer notifies consumer that computation is complete. This is a
    one-way synchronization from producer to consumer.
    
    Args:
        event_id: Event identifier for synchronization
    
    Example::
    
        # Producer (Cube core)
        plm.matmul(q, k, out=qk)
        cross_core_sync_forward(event_id=0)
        
        # Consumer (Vec core)
        plm.exp(qk, out=p)
    """
    _sync_op(SyncType.CROSS_CORE_SYNC_FORWARD, event_id, direction="record")


def cross_core_sync_both(event_id: int, direction: str):
    """Cross-core bidirectional synchronization.
    
    Supports both allocation (consumer waits) and recording (producer notifies)
    for bidirectional synchronization between producer and consumer.
    
    Args:
        event_id: Event identifier for synchronization
        direction: "allocate" (consumer waits) or "record" (producer notifies)
    
    Example::
    
        # Producer (Cube core)
        plm.matmul(q, k, out=qk)
        cross_core_sync_both(event_id=0, direction="record")
        
        # Consumer (Vec core)
        cross_core_sync_both(event_id=0, direction="allocate")
        plm.exp(qk, out=p)
    """
    _sync_op(SyncType.CROSS_CORE_SYNC_BOTH, event_id, direction=direction)


def allocate_buffer(event_id: int):
    """Allocate buffer - wait for consumer to release buffer space.
    
    This is producer-side operation that waits for buffer space to be available
    before writing. Corresponds to pto-isa's ubBufSync.allocate().
    
    Args:
        event_id: Event identifier for synchronization
    """
    _sync_op(SyncType.CROSS_CORE_SYNC_BOTH, event_id, direction="allocate")


def free_buffer(event_id: int):
    """Free buffer - notify producer that buffer space is available.
    
    This is consumer-side operation that signals buffer space is available.
    Corresponds to pto-isa's ubBufSync.free().
    
    Args:
        event_id: Event identifier for synchronization
    """
    _sync_op(SyncType.CROSS_CORE_SYNC_BOTH, event_id, direction="record")


def record_data_ready(event_id: int):
    """Record data ready - producer notifies consumer that data is ready.
    
    This is producer-side operation that signals data is ready for consumption.
    Corresponds to pto-isa's TSync_Custom::record().
    
    Args:
        event_id: Event identifier for synchronization
    """
    _sync_op(SyncType.CROSS_CORE_SYNC_BOTH, event_id, direction="record")


def wait_data_ready(event_id: int):
    """Wait for data ready - consumer waits for producer to make data ready.
    
    This is consumer-side operation that waits for data to be ready.
    Corresponds to pto-isa's TSync_Custom::wait().
    
    Args:
        event_id: Event identifier for synchronization
    """
    _sync_op(SyncType.CROSS_CORE_SYNC_BOTH, event_id, direction="allocate")


def _sync_op(sync_type: SyncType, event_id: int = 0, 
            direction: Optional[str] = None, pipeline: Optional[str] = None):
    """Internal sync operation that generates pto.sync MLIR.
    
    Args:
        sync_type: Type of synchronization
        event_id: Event identifier
        direction: Optional direction ("allocate" or "record")
        pipeline: Optional pipeline type for fine-grained sync
    """
    kwargs = {
        "sync_type": sync_type.value,
        "event_id": event_id,
    }
    if direction is not None:
        kwargs["direction"] = direction
    if pipeline is not None:
        kwargs["pipeline"] = pipeline
    
    _ir_core.create_op_call("manual.sync", [], kwargs, Span.unknown())


def sync_pipeline(pipeline: Union[str, "PipelineType"]):
    """Fine-grained pipeline synchronization.
    
    Args:
        pipeline: Pipeline type to synchronize
                  Supports both string ("PIPE_V") and enum (PipelineType.PIPE_V)
    
    Examples::
        # Synchronize vector pipeline
        sync_pipeline("PIPE_V")
        sync_pipeline(PipelineType.PIPE_V)
        
        # Synchronize MTE3 pipeline (for store operations)
        sync_pipeline("PIPE_MTE3")
    """
    if hasattr(pipeline, 'value'):
        pipeline = pipeline.value
    
    _sync_op(SyncType.INNER_CORE_SYNC, pipeline=pipeline)


def sync_vector():
    """Synchronize vector pipeline (PIPE_V)."""
    sync_pipeline("PIPE_V")


def sync_matrix():
    """Synchronize matrix pipeline (PIPE_M)."""
    sync_pipeline("PIPE_M")


def sync_mte2():
    """Synchronize MTE2 pipeline (PIPE_MTE2).
    
    Commonly used after load operations.
    """
    sync_pipeline("PIPE_MTE2")


def sync_mte3():
    """Synchronize MTE3 pipeline (PIPE_MTE3).
    
    Commonly used after store operations.
    """
    sync_pipeline("PIPE_MTE3")


__all__ = [
    "SyncEvent",
    "inner_core_sync",
    "cross_core_sync_forward",
    "cross_core_sync_both",
    "allocate_buffer",
    "free_buffer",
    "record_data_ready",
    "wait_data_ready",
    "PipelineType",
    "sync_pipeline",
    "sync_vector",
    "sync_matrix",
    "sync_mte2",
    "sync_mte3",
]
