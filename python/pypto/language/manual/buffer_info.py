# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Buffer information utilities for PyPTO."""

from pypto.pypto_core.ir import MemorySpace
from .buffer_type import HardEvent


class BufferInfo:
    """Buffer information class for mapping MemorySpace to HardEvent."""

    @staticmethod
    def get_event_p2c(memory_space: MemorySpace) -> HardEvent:
        """Get producer-to-consumer event type for given memory space.

        Args:
            memory_space: Memory space type

        Returns:
            HardEvent: Producer-to-consumer event type

        Raises:
            ValueError: If memory space is not supported
        """
        event_map = {
            MemorySpace.Mat: HardEvent.L1_P2C,
            MemorySpace.Left: HardEvent.L0_P2C,
            MemorySpace.Right: HardEvent.L0_P2C,
            MemorySpace.Acc: HardEvent.ACC_P2C,
        }
        
        if memory_space not in event_map:
            raise ValueError(f"Unsupported memory space for P2C event: {memory_space}")
        
        return event_map[memory_space]

    @staticmethod
    def get_event_c2p(memory_space: MemorySpace) -> HardEvent:
        """Get consumer-to-producer event type for given memory space.

        Args:
            memory_space: Memory space type

        Returns:
            HardEvent: Consumer-to-producer event type

        Raises:
            ValueError: If memory space is not supported
        """
        event_map = {
            MemorySpace.Mat: HardEvent.L1_C2P,
            MemorySpace.Left: HardEvent.L0_C2P,
            MemorySpace.Right: HardEvent.L0_C2P,
            MemorySpace.Acc: HardEvent.ACC_C2P,
        }
        
        if memory_space not in event_map:
            raise ValueError(f"Unsupported memory space for C2P event: {memory_space}")
        
        return event_map[memory_space]
