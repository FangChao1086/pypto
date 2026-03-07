# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Buffer class for PyPTO (PTO MLIR backend)."""

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import MemorySpace, PipeType
from pypto.ir.op import block_ops as _ir_block_ops
from pypto.ir.op import system_ops as _ir_system_ops
from .buffer_type import SyncType
from pypto.language.typing import Tile


_event_id_counter = 0
_max_event_id = 7


def _alloc_event_id() -> int:
    """分配事件ID"""
    global _event_id_counter
    event_id = _event_id_counter
    _event_id_counter = (_event_id_counter + 1) % (_max_event_id + 1)
    return event_id


def _release_event_id(event_id: int) -> None:
    """释放事件ID（暂时不做任何操作）"""
    pass


class Buffer:
    """Buffer类（同步管理）"""

    def __init__(
        self,
        shape: list[int],
        dtype: DataType,
        buffer_type: MemorySpace,
        sync_type: SyncType = SyncType.INNER_CORE_SYNC
    ):
        self.shape = shape
        self.dtype = dtype
        self.buffer_type = buffer_type
        self.sync_type = sync_type
        
        # 事件ID
        self._p2c_event_id = None
        self._c2p_event_id = None
        self._id0 = None  # 跨核同步ID（正向）
        self._id1 = None  # 跨核同步ID（反向）
        
        # 创建tensor
        self._tensor = self._create_tensor()

    def _create_tensor(self) -> Tile:
        """创建tensor根据shape和dtype"""
        return Tile(expr=_ir_block_ops.make_tile(
            self.shape, self.dtype, self.buffer_type
        ))

    def init(self):
        """初始化事件ID，设置初始标志"""
        if self.sync_type == SyncType.INNER_CORE_SYNC:
            self._p2c_event_id = _alloc_event_id()
            self._c2p_event_id = _alloc_event_id()
        elif (self.sync_type == SyncType.CROSS_CORE_SYNC_FORWARD or
              self.sync_type == SyncType.CROSS_CORE_SYNC_BOTH):
            self._id0 = _alloc_event_id()
            self._id1 = _alloc_event_id()

    def uninit(self):
        """释放事件ID"""
        if self.sync_type == SyncType.INNER_CORE_SYNC:
            if self._p2c_event_id is not None:
                _release_event_id(self._p2c_event_id)
            if self._c2p_event_id is not None:
                _release_event_id(self._c2p_event_id)
        elif (self.sync_type == SyncType.CROSS_CORE_SYNC_FORWARD or
              self.sync_type == SyncType.CROSS_CORE_SYNC_BOTH):
            if self._id0 is not None:
                _release_event_id(self._id0)
            if self._id1 is not None:
                _release_event_id(self._id1)

    def wait(self, event_type: int):
        """等待事件（PTO MLIR：使用pto.wait_event）

        Args:
            event_type: 事件类型（HardEvent枚举值）
        """
        if self.sync_type == SyncType.INNER_CORE_SYNC:
            if event_type == 2 or event_type == 4:  # L0_P2C or ACC_P2C
                if self._p2c_event_id is not None:
                    _ir_system_ops.wait_event(src_op=event_type, dst_op=event_type, event_id=self._p2c_event_id)
            elif event_type == 3 or event_type == 5:  # L0_C2P or ACC_C2P
                if self._c2p_event_id is not None:
                    _ir_system_ops.wait_event(src_op=event_type, dst_op=event_type, event_id=self._c2p_event_id)

    def set(self, event_type: int):
        """设置事件（PTO MLIR：使用pto.record_event）

        Args:
            event_type: 事件类型（HardEvent枚举值）
        """
        if self.sync_type == SyncType.INNER_CORE_SYNC:
            if event_type == 2 or event_type == 4:  # L0_P2C or ACC_P2C
                if self._p2c_event_id is not None:
                    _ir_system_ops.record_event(src_op=event_type, dst_op=event_type, event_id=self._p2c_event_id)
            elif event_type == 3 or event_type == 5:  # L0_C2P or ACC_C2P
                if self._c2p_event_id is not None:
                    _ir_system_ops.record_event(src_op=event_type, dst_op=event_type, event_id=self._c2p_event_id)

    def set_cross_core(self):
        """设置跨核同步（PTO MLIR：使用pto.sync.set）"""
        if self.sync_type in [SyncType.CROSS_CORE_SYNC_FORWARD,
                            SyncType.CROSS_CORE_SYNC_BOTH]:
            if self._id0 is not None:
                if self.buffer_type in [MemorySpace.Vec, MemorySpace.DDR]:
                    self._set_cross_core_ub_gm()
                elif self.buffer_type == MemorySpace.Mat:
                    self._set_cross_core_l1()

    def wait_cross_core(self):
        """等待跨核同步（PTO MLIR：使用pto.sync.wait）"""
        if self.sync_type in [SyncType.CROSS_CORE_SYNC_FORWARD,
                            SyncType.CROSS_CORE_SYNC_BOTH]:
            if self._id0 is not None:
                if self.buffer_type in [MemorySpace.Vec, MemorySpace.DDR]:
                    self._wait_cross_core_ub_gm()
                elif self.buffer_type == MemorySpace.Mat:
                    self._wait_cross_core_l1()

    def _set_cross_core_ub_gm(self):
        """UB/GM的跨核同步设置（PTO MLIR：使用pto.sync.set）

        参考ops-transformer实现：
        - AIC是生产者，AIV是消费者
        - 一个AIC对应两个AIV（AIV0和AIV1）
        - 使用PIPE_FIX和PIPE_V进行同步
        """
        if self._id0 is not None and self._id1 is not None:
            _ir_system_ops.sync_set(pipe=PipeType.FIX, event_id=self._id0)
            _ir_system_ops.sync_set(pipe=PipeType.FIX, event_id=self._id0 + 16)
            _ir_system_ops.sync_set(pipe=PipeType.V, event_id=self._id1)

    def _wait_cross_core_ub_gm(self):
        """UB/GM的跨核同步等待（PTO MLIR：使用pto.sync.wait）

        参考ops-transformer实现：
        - AIC是生产者，AIV是消费者
        - 一个AIC对应两个AIV（AIV0和AIV1）
        - 使用PIPE_FIX和PIPE_V进行同步
        """
        if self._id0 is not None and self._id1 is not None:
            _ir_system_ops.sync_wait(pipe=PipeType.FIX, event_id=self._id1)
            _ir_system_ops.sync_wait(pipe=PipeType.FIX, event_id=self._id1 + 16)
            _ir_system_ops.sync_wait(pipe=PipeType.V, event_id=self._id0)

    def _set_cross_core_l1(self):
        """L1的跨核同步设置（PTO MLIR：使用pto.sync.set）

        参考ops-transformer实现：
        - AIC是消费者，AIV是生产者
        - 一个AIC对应两个AIV（AIV0和AIV1）
        - 使用PIPE_MTE1和PIPE_MTE3进行同步
        - CROSS_CORE_SYNC_BOTH需要双向同步
        """
        if self._id0 is not None and self._id1 is not None:
            if self.sync_type == SyncType.CROSS_CORE_SYNC_BOTH:
                _ir_system_ops.sync_set(pipe=PipeType.MTE1, event_id=self._id1)
                _ir_system_ops.sync_set(pipe=PipeType.MTE1, event_id=self._id1 + 16)
            _ir_system_ops.sync_set(pipe=PipeType.MTE3, event_id=self._id0)

    def _wait_cross_core_l1(self):
        """L1的跨核同步等待（PTO MLIR：使用pto.sync.wait）

        参考ops-transformer实现：
        - AIC是消费者，AIV是生产者
        - 一个AIC对应两个AIV（AIV0和AIV1）
        - 使用PIPE_MTE1和PIPE_MTE3进行同步
        - CROSS_CORE_SYNC_BOTH需要双向同步
        """
        if self._id0 is not None and self._id1 is not None:
            _ir_system_ops.sync_wait(pipe=PipeType.MTE1, event_id=self._id0)
            if self.sync_type == SyncType.CROSS_CORE_SYNC_BOTH:
                _ir_system_ops.sync_wait(pipe=PipeType.MTE1, event_id=self._id0 + 16)
            if self.sync_type == SyncType.CROSS_CORE_SYNC_BOTH:
                _ir_system_ops.sync_wait(pipe=PipeType.MTE3, event_id=self._id1)

    def get_tensor(self) -> Tile:
        """获取tensor"""
        return self._tensor
