# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Buffer policy classes for PyPTO."""

from pypto.pypto_core import DataType
from pypto.pypto_core.ir import MemorySpace
from .buffer_type import SyncType
from .buffer import Buffer


class SingleBufferPolicy:
    """单buffer策略（无乒乓）"""

    def __init__(
        self,
        shape: list[int],
        dtype: DataType,
        buffer_type: MemorySpace,
        sync_type: SyncType = SyncType.INNER_CORE_SYNC
    ):
        self.buffer = Buffer(shape, dtype, buffer_type, sync_type)

    def init(self):
        """初始化"""
        self.buffer.init()

    def uninit(self):
        """释放"""
        self.buffer.uninit()

    def get(self) -> Buffer:
        """获取当前buffer"""
        return self.buffer

    def get_pre(self) -> Buffer:
        """获取前一个buffer（Q复用）"""
        return self.buffer

    def get_reused(self) -> Buffer:
        """获取复用buffer（KV复用）"""
        return self.buffer


class DoubleBufferPolicy:
    """双buffer策略（ping-pong轮转）"""

    def __init__(
        self,
        shape: list[int],
        dtype: DataType,
        buffer_type: MemorySpace,
        sync_type: SyncType = SyncType.INNER_CORE_SYNC
    ):
        self.ping = Buffer(shape, dtype, buffer_type, sync_type)
        self.pong = Buffer(shape, dtype, buffer_type, sync_type)
        self._flag1 = 0  # 轮转标记
        self._flag2 = 0  # 复用标记

    def init(self):
        """初始化"""
        self.ping.init()
        self.pong.init()

    def uninit(self):
        """释放"""
        self.ping.uninit()
        self.pong.uninit()

    def get(self) -> Buffer:
        """获取当前buffer（轮转）"""
        if self._flag1:
            self._flag1 = 0
            return self.ping
        else:
            self._flag1 = 1
            return self.pong

    def get_pre(self) -> Buffer:
        """获取前一个buffer（Q复用）"""
        if self._flag1:
            return self.pong
        else:
            return self.ping

    def get_reused(self) -> Buffer:
        """获取复用buffer（KV复用）"""
        if self._flag2 == 0:
            self._flag2 = 1
            return self.pong
        else:
            self._flag2 = 0
            return self.ping


class TripleBufferPolicy:
    """三buffer策略（轮转）"""

    def __init__(
        self,
        shape: list[int],
        dtype: DataType,
        buffer_type: MemorySpace,
        sync_type: SyncType = SyncType.INNER_CORE_SYNC
    ):
        self.a = Buffer(shape, dtype, buffer_type, sync_type)
        self.b = Buffer(shape, dtype, buffer_type, sync_type)
        self.c = Buffer(shape, dtype, buffer_type, sync_type)
        self._flag1 = 0  # 轮转标记
        self._flag2 = 0  # 复用标记
        self._flag1_vec = 0  # vec轮转标记
        self._flag1_cube = 0  # cube轮转标记

    def init(self):
        """初始化"""
        self.a.init()
        self.b.init()
        self.c.init()

    def uninit(self):
        """释放"""
        self.a.uninit()
        self.b.uninit()
        self.c.uninit()

    def get(self) -> Buffer:
        """获取当前buffer（轮转）"""
        if self._flag1 == 0:
            self._flag1 = 1
            return self.a
        elif self._flag1 == 1:
            self._flag1 = 2
            return self.b
        else:
            self._flag1 = 0
            return self.c

    def get_vec(self) -> Buffer:
        """获取buffer（用于vec计算）"""
        if self._flag1_vec == 0:
            self._flag1_vec = 1
            return self.a
        elif self._flag1_vec == 1:
            self._flag1_vec = 2
            return self.b
        else:
            self._flag1_vec = 0
            return self.c

    def get_cube(self) -> Buffer:
        """获取buffer（用于cube计算）"""
        if self._flag1_cube == 0:
            self._flag1_cube = 1
            return self.a
        elif self._flag1_cube == 1:
            self._flag1_cube = 2
            return self.b
        else:
            self._flag1_cube = 0
            return self.c

    def get_pre(self) -> Buffer:
        """获取前一个buffer（Q复用）"""
        if self._flag1 == 0:
            return self.c
        elif self._flag1 == 1:
            return self.a
        else:
            return self.b

    def get_reused(self) -> Buffer:
        """获取复用buffer（KV复用）"""
        if self._flag2 == 0:
            self._flag2 = 1
            return self.a
        elif self._flag2 == 1:
            self._flag2 = 2
            return self.b
        else:
            self._flag2 = 0
            return self.c


class QuadBufferPolicy:
    """四buffer策略（队列模式）"""

    def __init__(
        self,
        shape: list[int],
        dtype: DataType,
        buffer_type: MemorySpace,
        sync_type: SyncType = SyncType.INNER_CORE_SYNC
    ):
        self.a = Buffer(shape, dtype, buffer_type, sync_type)
        self.b = Buffer(shape, dtype, buffer_type, sync_type)
        self.c = Buffer(shape, dtype, buffer_type, sync_type)
        self.d = Buffer(shape, dtype, buffer_type, sync_type)
        self._tail = 0  # 队列队尾
        self._head = 0  # 队列队首+1
        self._used = 0  # 已使用的buffer

    def init(self):
        """初始化"""
        self.a.init()
        self.b.init()
        self.c.init()
        self.d.init()

    def uninit(self):
        """释放"""
        self.a.uninit()
        self.b.uninit()
        self.c.uninit()
        self.d.uninit()

    def get(self, buffer_id: int | None = None) -> Buffer:
        """获取buffer（按ID或队列模式）"""
        if buffer_id is not None:
            return self._get_by_id(buffer_id)
        else:
            buffer = self._get_by_id(self._head)
            self._head += 1
            return buffer

    def get_reused(self) -> Buffer:
        """获取复用buffer（队列复用）"""
        # 暂不实现，等待IR层支持
        return self.a

    def get_free(self) -> Buffer:
        """获取空闲buffer"""
        # 暂不实现，等待IR层支持
        return self.a

    def _get_by_id(self, buffer_id: int) -> Buffer:
        """根据ID获取buffer"""
        flag = buffer_id % 4
        if flag == 0:
            return self.a
        elif flag == 1:
            return self.b
        elif flag == 2:
            return self.c
        else:
            return self.d
