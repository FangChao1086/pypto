# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not get use this file except in compliance with License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in root of software repository for full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Buffer management policies for FlashAttention.

This module provides buffer management strategies similar to ops-transformer's
BuffersPolicyDB, BuffersPolicy3buff, BuffersPolicy4buff, etc. These policies
are designed for efficient memory management in FlashAttention kernels on A5 architecture.
"""

from enum import Enum
from typing import Optional

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import MemorySpace

from . import TileType, make_tile


class SyncType(Enum):
    """Synchronization type enumeration matching ops-transformer SyncType."""
    NO_SYNC = "no_sync"
    INNER_CORE_SYNC = "inner_core_sync"
    CROSS_CORE_SYNC_FORWARD = "cross_core_sync_forward"
    CROSS_CORE_SYNC_BOTH = "cross_core_sync_both"


class PipelineType(Enum):
    """Pipeline type enumeration for fine-grained synchronization."""
    PIPE_V = "PIPE_V"
    PIPE_M = "PIPE_M"
    PIPE_FIX = "PIPE_FIX"
    PIPE_MTE1 = "PIPE_MTE1"
    PIPE_MTE2 = "PIPE_MTE2"
    PIPE_MTE3 = "PIPE_MTE3"
    PIPE_ALL = "PIPE_ALL"


def _get_dtype_size(dtype: DataType) -> int:
    """Get in bytes."""
    return dtype.get_bit() // 8


def _calc_total_elems(shape) -> int:
    """Calculate total number of elements from shape."""
    total_elems = 1
    if hasattr(shape, '__iter__'):
        for dim in shape:
            total_elems *= dim
    return total_elems


class EventIdGenerator:
    """Global event ID generator following ops-transformer's MAKE_ID pattern.

    Each call to make_id() returns a unique event ID in range [0, 10],
    following to pattern: (++idCounterNum) % 11
    """
    _counter = 0

    @classmethod
    def make_id(cls) -> int:
        """Generate next event ID.

        Returns:
            Event ID in range [0, 10]
        """
        event_id = cls._counter % 11
        cls._counter += 1
        return event_id

    @classmethod
    def reset(cls) -> None:
        """Reset counter (for testing purposes)."""
        cls._counter = 0


class BuffersPolicyDB:
    """Double buffer policy with ping-pong rotation.

    This policy allocates 2 buffers and rotates between them. It's suitable for
    scenarios like Q/K data buffering in FlashAttention.

    Example::

        tile_type = TileType(shape=[64, 128], dtype=FP16, target_memory=MemorySpace.Vec)
        policy = BuffersPolicyDB(MemorySpace.Vec, SyncType.CROSS_CORE_SYNC_BOTH)

        buf0 = policy.get()
        buf1 = policy.get()

        # Synchronization methods based on sync_type
        policy.sync_inner()                    # INNER_CORE_SYNC
        policy.sync_forward(event_id=0)         # CROSS_CORE_SYNC_FORWARD
        policy.sync_both(event_id=0, "record") # CROSS_CORE_SYNC_BOTH
    """

    def __init__(
        self,
        tile_type: TileType,
        memory_space: MemorySpace,
        sync_type: SyncType = SyncType.INNER_CORE_SYNC,
        base_addr: int = 0,
    ):
        self.tile_type = tile_type
        self.memory_space = memory_space
        self.sync_type = sync_type
        self.base_addr = base_addr
        self._flag = 0
        self._ping = None
        self._pong = None
        self._buffer_size = self._calc_size()
        
        self._id0 = EventIdGenerator.make_id()
        self._id1 = EventIdGenerator.make_id()

    def get(self):
        """Get current buffer and auto-swap."""
        if self._flag == 0:
            if self._ping is None:
                self._ping = make_tile(self.tile_type, addr=self.base_addr, size=self._buffer_size)
            self._flag = 1
            return self._ping
        else:
            if self._pong is None:
                self._pong = make_tile(
                    self.tile_type, addr=self.base_addr + self._buffer_size, size=self._buffer_size
                )
            self._flag = 0
            return self._pong

    def get_pre(self):
        """Get previous buffer (for Q reuse)."""
        if self._flag == 0:
            return self.get()
        else:
            return self._pong if self._pong else self.get()

    def get_reused(self):
        """Get reusable buffer (for KV reuse)."""
        return self.get()

    def swap(self):
        """Swap to other buffer."""
        self._flag = 1 - self._flag

    def sync_inner(self, pipeline: Optional[str] = None):
        """Perform inner core synchronization.
        
        Args:
            pipeline: Optional pipeline type for fine-grained sync (e.g., "PIPE_V", "PIPE_MTE2")
                      If None, performs generic inner core synchronization
        """
        from . import sync as _sync
        if pipeline is not None:
            _sync.sync_pipeline(pipeline)
        else:
            _sync.inner_core_sync()

    def sync_forward(self, event_id: int):
        """Perform cross-core forward synchronization.

        Args:
            event_id: Event identifier for synchronization
        """
        from . import sync as _sync
        _sync.cross_core_sync_forward(event_id)

    def sync_both(self, event_id: int, direction: str):
        """Perform cross-core bidirectional synchronization.

        Args:
            event_id: Event identifier for synchronization
            direction: "allocate" (consumer waits) or "record" (producer notifies)
        """
        from . import sync as _sync
        _sync.cross_core_sync_both(event_id, direction)

    def allocate_buffer(self, event_id: int):
        """Allocate buffer - wait for consumer to release buffer space.
        
        This is producer-side operation that waits for buffer space to be available
        before writing. Corresponds to pto-isa's ubBufSync.allocate().
        
        Args:
            event_id: Event identifier for synchronization
        """
        from . import sync as _sync
        _sync.allocate_buffer(event_id)

    def free_buffer(self, event_id: int):
        """Free buffer - notify producer that buffer space is available.
        
        This is consumer-side operation that signals buffer space is available.
        Corresponds to pto-isa's ubBufSync.free().
        
        Args:
            event_id: Event identifier for synchronization
        """
        from . import sync as _sync
        _sync.free_buffer(event_id)

    def _calc_size(self) -> int:
        """Calculate buffer size in bytes."""
        elem_size = _get_dtype_size(self.tile_type.dtype)
        total_elems = _calc_total_elems(self.tile_type.shape)
        return total_elems * elem_size

    def init_consumer(self):
        """Initialize consumer side (AIV) - release all buffers.
        
        For double buffer policy with 2 Vec subblocks:
        - First call: releases flag_id+1 (for Vec subblock 0)
        - Second call: releases flag_id+1+16 (for Vec subblock 1)
        
        This allows Cube to wait for both Vec subblocks to release buffers.
        
        Raises:
            RuntimeError: If sync_type is not CROSS_CORE_SYNC_BOTH
        """
        if self.sync_type != SyncType.CROSS_CORE_SYNC_BOTH:
            raise RuntimeError(
                "init_consumer() only works with SyncType.CROSS_CORE_SYNC_BOTH, "
                f"got {self.sync_type}"
            )
        
        from . import sync as _sync
        
        # Release buffer 0 for both Vec subblocks
        _sync.free_buffer(self._id0)
        _sync.free_buffer(self._id0 + 16)
        
        # Release buffer 1 for both Vec subblocks
        _sync.free_buffer(self._id1)
        _sync.free_buffer(self._id1 + 16)

    def init_producer(self):
        """Initialize producer side (AIC).

        AIC side doesn't need initialization.

        Raises:
            RuntimeError: If sync_type is not CROSS_CORE_SYNC_BOTH
        """
        if self.sync_type != SyncType.CROSS_CORE_SYNC_BOTH:
            raise RuntimeError(
                "init_producer() only works with SyncType.CROSS_CORE_SYNC_BOTH, "
                f"got {self.sync_type}"
            )
        pass

    def get_event_ids(self):
        """Get current policy's event_id pair."""
        return (self._id0, self._id1)
    
    def record_data_ready(self, event_id: int):
        """Record data ready - producer notifies consumer that data is ready.
        
        This is producer-side operation that signals data is ready for consumption.
        Corresponds to pto-isa's TSync_Custom::record().
        
        Args:
            event_id: Event identifier for synchronization
        """
        from . import sync as _sync
        _sync.record_data_ready(event_id)
    
    def wait_data_ready(self, event_id: int):
        """Wait for data ready - consumer waits for producer to make data ready.
        
        This is consumer-side operation that waits for data to be ready.
        Corresponds to pto-isa's TSync_Custom::wait().
        
        Args:
            event_id: Event identifier for synchronization
        """
        from . import sync as _sync
        _sync.wait_data_ready(event_id)


class BuffersPolicy3buff:
    """Three buffer policy with rotation.
    
    This policy allocates 3 buffers and rotates between them. It's suitable for
    scenarios like L1 buffering in FlashAttention with mixcore architecture.
    
    Features:
    - Get(): Rotates through buffers (a -> b) -> c -> a)
    - GetVec(): Independent rotation for Vec subblocks (mixcore architecture)
    - GetCube(): Independent rotation for Cube subblocks (mixcore architecture)
    - GetPre(): Q reuse - returns previous buffer
    - GetReused(): KV reuse - independent rotation
    
    Example::
    
        tile_type = TileType(shape=[64, 128], dtype=FP16, target_memory=MemorySpace.Mat)
        policy = BuffersPolicy3buff(tile_type, MemorySpace.Mat, SyncType.INNER_CORE_SYNC)
        
        buf_a = policy.get()
        buf_b = policy.get()
        buf_c = policy.get()
        
        # Mixcore architecture: separate Vec and Cube access
        buf_vec = policy.get_vec()
        buf_cube = policy.get_cube()
        
        # Q reuse pattern
        buf_prev = policy.get_pre()
        
        # KV reuse pattern
        buf_reused = policy.get_reused()
    """
    
    def __init__(
        self,
        tile_type: TileType,
        memory_space: MemorySpace,
        sync_type: SyncType = SyncType.INNER_CORE_SYNC,
        base_addr: int = 0,
    ):
        self.tile_type = tile_type
        self.memory_space = memory_space
        self.sync_type = sync_type
        self.base_addr = base_addr
        
        # Initialize all 3 buffers as None (lazy allocation)
        self._buffers = [None, None, None]  # [a_, b_, c_]
        self._buffer_size = self._calc_size()
        
        # Initialize event IDs
        self._id0 = EventIdGenerator.make_id()
        self._id1 = EventIdGenerator.make_id()
        
        # Initialize flags for different access patterns
        self._flag1 = 0          # For Get() rotation
        self._flag1_vec1_ = 0     # For GetVec() independent rotation
        self._flag1_bmm2_ = 0     # For GetCube() independent rotation
        self._flag2 = 0          # For GetReused() KV reuse
    
    def get(self):
        """Get current buffer and rotate (a -> b -> c -> a).
        
        Returns:
            Current buffer tile
        """
        if self._flag1 == 0:
            self._flag1 = 1
            return self._get_or_create_buffer(0)  # a_
        elif self._flag1 == 1:
            self._flag1 = 2
            return self._get_or_create_buffer(1)  # b_
        else:
            self._flag1 = 0
            return self._get_or_create_buffer(2)  # c_
    
    def get_vec(self):
        """Get buffer for Vec subblock with independent rotation.
        
        This is used in mixcore architecture where Vec and Cube subblocks
        have independent buffer access patterns.
        
        Returns:
            Current Vec buffer tile
        """
        if self._flag1_vec1_ == 0:
            self._flag1_vec1_ = 1
            return self._get_or_create_buffer(0)  # a_
        elif self._flag1_vec1_ == 1:
            self._flag1_vec1_ = 2
            return self._get_or_create_buffer(1)  # b_
        else:
            self._flag1_vec1_ = 0
            return self._get_or_create_buffer(2)  # c_
    
    def get_cube(self):
        """Get buffer for Cube subblock with independent rotation.
        
        This is used in mixcore architecture where Vec and Cube subblocks
        have independent buffer access patterns.
        
        Returns:
            Current Cube buffer tile
        """
        if self._flag1_bmm2_ == 0:
            self._flag1_bmm2_ = 1
            return self._get_or_create_buffer(0)  # a_
        elif self._flag1_bmm2_ == 1:
            self._flag1_bmm2_ = 2
            return self._get_or_create_buffer(1)  # b_
        else:
            self._flag1_bmm2_ = 0
            return self._get_or_create_buffer(2)  # c_
    
    def get_pre(self):
        """Get previous buffer (for Q reuse).
        
        Returns:
            Previous buffer tile
        """
        if self._flag1 == 0:
            return self._get_or_create_buffer(2)  # c_
        elif self._flag1 == 1:
            return self._get_or_create_buffer(0)  # a_
        else:
            return self._get_or_create_buffer(1)  # b_
    
    def get_reused(self):
        """Get reusable buffer (for KV reuse).
        
        This maintains an independent rotation state for KV reuse scenarios.
        Rotation: a -> b -> c -> a
        
        Returns:
            Reusable buffer tile
        """
        if self._flag2 == 0:
            self._flag2 = 1
            return self._get_or_create_buffer(0)  # a_
        elif self._flag2 == 1:
            self._flag2 = 2
            return self._get_or_create_buffer(1)  # b_
        else:
            self._flag2 = 0
            return self._get_or_create_buffer(2)  # c_
    
    def _get_or_create_buffer(self, idx: int):
        """Get buffer at index, create if not exists.
        
        Args:
            idx: Buffer index (0=a_, 1=b_, 2=c_)
            
        Returns:
            Buffer tile at specified index
        """
        if self._buffers[idx] is None:
            addr = self.base_addr + idx * self._buffer_size
            self._buffers[idx] = make_tile(self.tile_type, addr=addr, size=self._buffer_size)
        return self._buffers[idx]
    
    # Synchronization methods (same as other policies)
    def sync_inner(self, pipeline: Optional[str] = None):
        """Perform inner core synchronization.
        
        Args:
            pipeline: Optional pipeline type for fine-grained sync (e.g., "PIPE_V", "PIPE_MTE2")
                      If None, performs generic inner core synchronization
        """
        from . import sync as _sync
        if pipeline is not None:
            _sync.sync_pipeline(pipeline)
        else:
            _sync.inner_core_sync()

    def sync_forward(self, event_id: int):
        """Perform cross-core forward synchronization.
        
        Args:
            event_id: Event identifier for synchronization
        """
        from . import sync as _sync
        _sync.cross_core_sync_forward(event_id)
    
    def sync_both(self, event_id: int, direction: str):
        """Perform cross-core bidirectional synchronization.
        
        Args:
            event_id: Event identifier for synchronization
            direction: "allocate" (consumer waits) or "record" (producer notifies)
        """
        from . import sync as _sync
        _sync.cross_core_sync_both(event_id, direction)
    
    def allocate_buffer(self, event_id: int):
        """Allocate buffer - wait for consumer to release buffer space.
        
        Args:
            event_id: Event identifier for synchronization
        """
        from . import sync as _sync
        _sync.allocate_buffer(event_id)
    
    def free_buffer(self, event_id: int):
        """Free buffer - notify producer that buffer space is available.
        
        Args:
            event_id: Event identifier for synchronization
        """
        from . import sync as _sync
        _sync.free_buffer(event_id)
    
    def _calc_size(self) -> int:
        """Calculate buffer size in bytes."""
        elem_size = _get_dtype_size(self.tile_type.dtype)
        total_elems = _calc_total_elems(self.tile_type.shape)
        return total_elems * elem_size
    
    def init_consumer(self):
        """Initialize consumer side (AIV).
        
        Raises:
            RuntimeError: If sync_type is not CROSS_CORE_SYNC_BOTH
        """
        if self.sync_type != SyncType.CROSS_CORE_SYNC_BOTH:
            raise RuntimeError(
                "init_consumer() only works with SyncType.CROSS_CORE_SYNC_BOTH, "
                f"got {self.sync_type}"
            )
        pass
    
    def init_producer(self):
        """Initialize producer side (AIC).
        
        Raises:
            RuntimeError: If sync_type is not CROSS_CORE_SYNC_BOTH
        """
        if self.sync_type != SyncType.CROSS_CORE_SYNC_BOTH:
            raise RuntimeError(
                "init_producer() only works with SyncType.CROSS_CORE_SYNC_BOTH, "
                f"got {self.sync_type}"
            )
        pass
    
    def get_event_ids(self):
        """Get current policy's event_id pair."""
        return (self._id0, self._id1)
    
    def record_data_ready(self, event_id: int):
        """Record data ready - producer notifies consumer that data is ready.
        
        Args:
            event_id: Event identifier for synchronization
        """
        from . import sync as _sync
        _sync.record_data_ready(event_id)
    
    def wait_data_ready(self, event_id: int):
        """Wait for data ready - consumer waits for producer to make data ready.
        
        Args:
            event_id: Event identifier for synchronization
        """
        from . import sync as _sync
        _sync.wait_data_ready(event_id)


class BuffersPolicy4buff:
    """Four buffer policy with FIFO management.

    This policy allocates 4 buffers and manages them as a FIFO. It's suitable for
    scenarios like L0C buffering in FlashAttention.

    Example::

        tile_type = TileType(shape=[64, 64], dtype=FP32, target_memory=MemorySpace.Acc)
        policy = BuffersPolicy4buff(tile_type, MemorySpace.Acc, SyncType.INNER_CORE_SYNC)

        buf0 = policy.get()
        buf1 = policy.get()
        policy.release()
        buf2 = policy.get()
        buf3 = policy.get()

        # Synchronization methods based on sync_type
        policy.sync_inner()                    # INNER_CORE_SYNC
        policy.sync_forward(event_id=0)         # CROSS_CORE_SYNC_FORWARD
        policy.sync_both(event_id=0, "record") # CROSS_CORE_SYNC_BOTH
    """

    def __init__(
        self,
        tile_type: TileType,
        memory_space: MemorySpace,
        sync_type: SyncType = SyncType.INNER_CORE_SYNC,
        base_addr: int = 0,
    ):
        self.tile_type = tile_type
        self.memory_space = memory_space
        self.sync_type = sync_type
        self.base_addr = base_addr
        self._tail = 0
        self._head = 0
        self._used = 0
        self._buffers = [None, None, None, None]
        self._buffer_size = self._calc_size()
        self._id0 = 0
        self._id1 = 1

    def get(self):
        """Get current buffer (FIFO)."""
        if self._used < 4:
            idx = self._head
            self._head = (self._head + 1) % 4
            self._used += 1

            if self._buffers[idx] is None:
                addr = self.base_addr + idx * self._buffer_size
                self._buffers[idx] = make_tile(self.tile_type, addr=addr, size=self._buffer_size)
                return self._buffers[idx]
            else:
                raise RuntimeError("FIFO buffer overflow")

    def release(self):
        """Release buffer (FIFO)."""
        if self._used > 0:
            self._tail = (self._tail + 1) % 4
            self._used -= 1

    def sync_inner(self, pipeline: Optional[str] = None):
        """Perform inner core synchronization.
        
        Args:
            pipeline: Optional pipeline type for fine-grained sync (e.g., "PIPE_V", "PIPE_MTE2")
                      If None, performs generic inner core synchronization
        """
        from . import sync as _sync
        if pipeline is not None:
            _sync.sync_pipeline(pipeline)
        else:
            _sync.inner_core_sync()

    def sync_forward(self, event_id: int):
        """Perform cross-core forward synchronization.

        Args:
            event_id: Event identifier for synchronization
        """
        from . import sync as _sync
        _sync.cross_core_sync_forward(event_id)

    def sync_both(self, event_id: int, direction: str):
        """Perform cross-core bidirectional synchronization.

        Args:
            event_id: Event identifier for synchronization
            direction: "allocate" (consumer waits) or "record" (producer notifies)
        """
        from . import sync as _sync
        _sync.cross_core_sync_both(event_id, direction)

    def allocate_buffer(self, event_id: int):
        """Allocate buffer - wait for consumer to release buffer space.
        
        This is producer-side operation that waits for buffer space to be available
        before writing. Corresponds to pto-isa's ubBufSync.allocate().
        
        Args:
            event_id: Event identifier for synchronization
        """
        from . import sync as _sync
        _sync.allocate_buffer(event_id)

    def free_buffer(self, event_id: int):
        """Free buffer - notify producer that buffer space is available.
        
        This is consumer-side operation that signals buffer space is available.
        Corresponds to pto-isa's ubBufSync.free().
        
        Args:
            event_id: Event identifier for synchronization
        """
        from . import sync as _sync
        _sync.free_buffer(event_id)

    def _calc_size(self) -> int:
        """Calculate buffer size in bytes."""
        elem_size = _get_dtype_size(self.tile_type.dtype)
        total_elems = _calc_total_elems(self.tile_type.shape)
        return total_elems * elem_size

    def init_consumer(self):
        """Initialize consumer side (AIV) - set buffer as consumed.

        For four buffer policy with 2 Vec subblocks:
        - First 4 calls: set flag_id+1 (for Vec subblock 0, 4 buffers)

        Raises:
            RuntimeError: If sync_type is not CROSS_CORE_SYNC_BOTH
        """
        if self.sync_type != SyncType.CROSS_CORE_SYNC_BOTH:
            raise RuntimeError(
                "init_consumer() only works with SyncType.CROSS_CORE_SYNC_BOTH, "
                f"got {self.sync_type}"
            )

        from . import sync as _sync
        _sync.cross_core_sync_both(self._id1, direction="record")
        _sync.cross_core_sync_both(self._id1, direction="record")
        _sync.cross_core_sync_both(self._id1, direction="record")
        _sync.cross_core_sync_both(self._id1, direction="record")

    def init_producer(self):
        """Initialize producer side (AIC).

        AIC side doesn't need initialization.

        Raises:
            RuntimeError: If sync_type is not CROSS_CORE_SYNC_BOTH
        """
        if self.sync_type != SyncType.CROSS_CORE_SYNC_BOTH:
            raise RuntimeError(
                "init_producer() only works with SyncType.CROSS_CORE_SYNC_BOTH, "
                f"got {self.sync_type}"
            )
        pass

    def get_event_ids(self):
        """Get current policy's event_id pair."""
        return (self._id0, self._id1)
    
    def record_data_ready(self, event_id: int):
        """Record data ready - producer notifies consumer that data is ready.
        
        This is producer-side operation that signals data is ready for consumption.
        Corresponds to pto-isa's TSync_Custom::record().
        
        Args:
            event_id: Event identifier for synchronization
        """
        from . import sync as _sync
        _sync.record_data_ready(event_id)
    
    def wait_data_ready(self, event_id: int):
        """Wait for data ready - consumer waits for producer to make data ready.
        
        This is consumer-side operation that waits for data to be ready.
        Corresponds to pto-isa's TSync_Custom::wait().
        
        Args:
            event_id: Event identifier for synchronization
        """
        from . import sync as _sync
        _sync.wait_data_ready(event_id)


class BuffersPolicySingleBuffer:
    """Single buffer policy.

    This policy allocates a single buffer. It's suitable for scenarios that
    don't require double buffering.

    Example::

        tile_type = TileType(shape=[64, 128], dtype=FP16, target_memory=MemorySpace.Vec)
        policy = BuffersPolicySingleBuffer(tile_type, MemorySpace.Vec, SyncType.INNER_CORE_SYNC)

        buf = policy.get()

        # Synchronization methods based on sync_type
        policy.sync_inner()                    # INNER_CORE_SYNC
        policy.sync_forward(event_id=0)         # CROSS_CORE_SYNC_FORWARD
        policy.sync_both(event_id=0, "record") # CROSS_CORE_SYNC_BOTH
    """

    def __init__(
        self,
        tile_type: TileType,
        memory_space: MemorySpace,
        sync_type: SyncType = SyncType.INNER_CORE_SYNC,
        base_addr: int = 0,
    ):
        self.tile_type = tile_type
        self.memory_space = memory_space
        self.sync_type = sync_type
        self.base_addr = base_addr
        self._buffer = None
        self._buffer_size = self._calc_size()
        self._id0 = 0
        self._id1 = 1

    def get(self):
        """Get single buffer."""
        if self._buffer is None:
            self._buffer = make_tile(self.tile_type, addr=self.base_addr, size=self._buffer_size)
        return self._buffer

    def sync_inner(self, pipeline: Optional[str] = None):
        """Perform inner core synchronization.
        
        Args:
            pipeline: Optional pipeline type for fine-grained sync (e.g., "PIPE_V", "PIPE_MTE2")
                      If None, performs generic inner core synchronization
        """
        from . import sync as _sync
        if pipeline is not None:
            _sync.sync_pipeline(pipeline)
        else:
            _sync.inner_core_sync()

    def sync_forward(self, event_id: int):
        """Perform cross-core forward synchronization.

        Args:
            event_id: Event identifier for synchronization
        """
        from . import sync as _sync
        _sync.cross_core_sync_forward(event_id)

    def sync_both(self, event_id: int, direction: str):
        """Perform cross-core bidirectional synchronization.

        Args:
            event_id: Event identifier for synchronization
            direction: "allocate" (consumer waits) or "record" (producer notifies)
        """
        from . import sync as _sync
        _sync.cross_core_sync_both(event_id, direction)

    def allocate_buffer(self, event_id: int):
        """Allocate buffer - wait for consumer to release buffer space.
        
        This is producer-side operation that waits for buffer space to be available
        before writing. Corresponds to pto-isa's ubBufSync.allocate().
        
        Args:
            event_id: Event identifier for synchronization
        """
        from . import sync as _sync
        _sync.allocate_buffer(event_id)

    def free() -> None:
        """Free buffer (consumer releases for producer).

        This is consumer-side operation that signals buffer space is available.
        Corresponds to pto-isa's ubBufSync.free().
        """
        from . import sync as _sync
        _sync.free_buffer(self._id1)

    def _calc_size(self) -> int:
        """Calculate buffer size in bytes."""
        elem_size = _get_dtype_size(self.tile_type.dtype)
        total_elems = _calc_total_elems(self.tile_type.shape)
        return total_elems * elem_size

    def init_consumer(self):
        """Initialize consumer side (AIV) - set buffer as consumed.

        For single buffer policy, calls SetCrossCore(id1_) once.

        Raises:
            RuntimeError: If sync_type is not CROSS_CORE_SYNC_BOTH
        """
        if self.sync_type != SyncType.CROSS_CORE_SYNC_BOTH:
            raise RuntimeError(
                "init_consumer() only works with SyncType.CROSS_CORE_SYNC_BOTH, "
                f"got {self.sync_type}"
            )

        from . import sync as _sync
        _sync.cross_core_sync_both(self._id1, direction="record")

    def init_producer(self):
        """Initialize producer side (AIC).

        AIC side doesn't need initialization.

        Raises:
            RuntimeError: If sync_type is not CROSS_CORE_SYNC_BOTH
        """
        if self.sync_type != SyncType.CROSS_CORE_SYNC_BOTH:
            raise RuntimeError(
                "init_producer() only works with SyncType.CROSS_CORE_SYNC_BOTH, "
                f"got {self.sync_type}"
            )
        pass

    def get_event_ids(self):
        """Get current policy's event_id pair."""
        return (self._id0, self._id1)
    
    def record_data_ready(self, event_id: int):
        """Record data ready - producer notifies consumer that data is ready.
        
        This is producer-side operation that signals data is ready for consumption.
        Corresponds to pto-isa's TSync_Custom::record().
        
        Args:
            event_id: Event identifier for synchronization
        """
        from . import sync as _sync
        _sync.record_data_ready(event_id)
    
    def wait_data_ready(self, event_id: int):
        """Wait for data ready - consumer waits for producer to make data ready.
        
        This is consumer-side operation that waits for data to be ready.
        Corresponds to pto-isa's TSync_Custom::wait().
        
        Args:
            event_id: Event identifier for synchronization
        """
        from . import sync as _sync
        _sync.wait_data_ready(event_id)


__all__ = [
    "SyncType",
    "BuffersPolicyDB",
    "BuffersPolicy3buff",
    "BuffersPolicy4buff",
    "BuffersPolicySingleBuffer",
]
