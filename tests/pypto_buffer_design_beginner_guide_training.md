# PyPTO Buffer管理设计 - 小白入门指南（训练算子版）

## 目录
1. [为什么需要Buffer管理？](#1-为什么需要buffer管理)
2. [核心概念详解](#2-核心概念详解)
3. [设计理念](#3-设计理念)
4. [API设计](#4-api设计)
5. [实现步骤](#5-实现步骤)
6. [实践示例](#6-实践示例)
7. [同步机制详解](#7-同步机制详解)
8. [常见问题](#8-常见问题)
9. [总结](#9-总结)

---

## 1. 为什么需要Buffer管理？

### 1.1 生活比喻：厨房做饭

想象你在厨房做饭：

**没有buffer管理（单缓冲）**：
```
1. 切菜（5分钟）
2. 炒菜（10分钟）
3. 炒菜（5分钟）
总时间：20分钟
```

**有buffer管理（双缓冲）**：
```
1. 切菜到锅A（5分钟）
2. 炒菜到锅B（10分钟）【同时切菜到锅A】
3. 炒菜到锅A（5分钟）【同时切菜到锅B】
总时间：10分钟（节省50%！）
```

### 1.2 在NPU中的应用

NPU（神经网络处理器）就像一个高效的厨房：

- **DDR（全局内存）**：大仓库，存储所有数据
- **Vec（统一缓冲区）**：工作台，放当前要处理的数据
- **Mat（L1缓存）**：大托盘，放常用数据
- **Left/Right/Acc（L0缓冲区）**：小托盘，放矩阵操作数

**问题**：从DDR加载数据很慢，但计算很快。如果等加载完再计算，计算单元就闲置了。

**解决**：使用双缓冲，让加载和计算同时进行！

```
时间轴：
t0: [加载数据到buffer1]  [计算buffer0]
t1: [加载数据到buffer0]  [计算buffer1]
t2: [加载数据到buffer1]  [计算buffer0]
...
```

### 1.3 Flash Attention的场景

Flash Attention是NPU上的一个重要算法：

```
输入：Q, K, V 三个矩阵
输出：Attention结果

计算过程：
1. 计算 Q × K^T  → 得到Score
2. 对Score做Softmax
3. 计算 Score × V → 得到结果
```

**为什么需要buffer管理？**

- Q、K、V很大，不能一次性加载到NPU
- 需要分块加载和处理
- 需要流水线并行，提高效率
- 需要复用机制（Q复用、KV复用）

---

## 2. 核心概念详解

### 2.1 Buffer（缓冲区）

**什么是Buffer？**

Buffer就是一块存储数据的内存空间。

**比喻**：
- Buffer就像一个"托盘"
- 你把数据放到托盘上，然后交给计算单元处理

**PyPTO中的Buffer类型**：

| Buffer类型 | PyPTO名称 | 作用 | 比喻 |
|-----------|-----------|------|------|
| L1 | `plm.MemorySpace.Mat` | L1缓存 | 大托盘，放常用数据 |
| L0A | `plm.MemorySpace.Left` | 矩阵左操作数托盘 | 左手托盘 |
| L0B | `plm.MemorySpace.Right` | 矩阵右操作数托盘 | 右手托盘 |
| L0C | `plm.MemorySpace.Acc` | 矩阵累加器托盘 | 结果托盘 |
| UB | `plm.MemorySpace.Vec` | 统一缓冲区 | 工作台 |
| DDR | `plm.MemorySpace.DDR` | DDR内存（片外） | 大仓库 |

### 2.2 Buffer Policy（缓冲策略）

**什么是Buffer Policy？**

Buffer Policy就是管理多个buffer的策略。

**比喻**：
- Buffer Policy就像"托盘管理员"
- 他负责分配托盘，告诉你用哪个托盘

**常见的Buffer策略**：

| 策略 | Buffer数量 | 作用 | 比喻 |
|-------|-----------|------|------|
| SingleBuffer | 1个 | 只有一个托盘 | 只有一个托盘，用完再装 |
| DoubleBuffer | 2个 | 两个托盘轮转 | 两个托盘，交替使用 |
| TripleBuffer | 3个 | 三个托盘轮转 | 三个托盘，循环使用 |
| QuadBuffer | 4个 | 四个托盘轮转 | 四个托盘，循环使用 |

### 2.3 Sync Type（同步类型）

**什么是Sync Type？**

Sync Type就是控制buffer之间同步的方式。

**比喻**：
- Sync Type就像"托盘使用规则" + "同步信号"
- 规则规定了什么时候可以用托盘，什么时候要等待
- 同步信号用于协调多个核心之间的数据访问

**常见的Sync类型**：

| Sync类型 | 作用 | 比喻 | 需要事件ID |
|----------|------|------|------------|
| NO_SYNC | 不同步 | 免费用托盘，不用等 | ❌ 不需要 |
| INNER_CORE_SYNC | 核内同步 | 核内用托盘，核内要等 | ✅ 需要 |
| CROSS_CORE_SYNC_FORWARD | 核间同步（单向） | 核间用托盘，单向等待 | ✅ 需要 |
| CROSS_CORE_SYNC_BOTH | 核间同步（双向） | 核间用托盘，双向等待 | ✅ 需要 |

### 2.4 HardEvent（硬件事件）

**什么是HardEvent？**

HardEvent是NPU硬件提供的事件类型，用于精确控制同步。

**比喻**：
- HardEvent就像"门铃"
- 生产者按门铃通知消费者数据已准备好
- 消费者按门铃通知生产者数据已消费

**常见的HardEvent类型**：

| HardEvent | 作用 | 对应Buffer类型 |
|-----------|------|---------------|
| L1_P2C | L1：生产者到消费者（MTE2_MTE1） | L1 |
| L1_C2P | L1：消费者到生产者（MTE1_MTE2） | L1 |
| L0_P2C | L0A/L0B：生产者到消费者（MTE1_M） | L0A/L0B |
| L0_C2P | L0A/L0B：消费者到生产者（M_MTE1） | L0A/L0B |
| ACC_P2C | L0C：生产者到消费者（M_FIX） | L0C |
| ACC_C2P | L0C：消费者到生产者（FIX_M） | L0C |

### 2.5 BufferInfo（Buffer硬件信息）

**什么是BufferInfo？**

BufferInfo提供Buffer的硬件抽象信息，包括HardEvent。

**比喻**：
- BufferInfo就像"托盘说明书"
- 告诉你这个托盘应该用什么门铃
- 告诉你这个托盘应该放在什么位置

**BufferInfo的作用**：

```python
# 获取生产者到消费者的HardEvent
event_p2c = plm.BufferInfo.get_event_p2c(plm.MemorySpace.Mat)

# 获取消费者到生产者的HardEvent
event_c2p = plm.BufferInfo.get_event_c2p(plm.MemorySpace.Mat)
```

---

## 3. 设计理念

### 3.1 核心思想

**ops-transformer的buffer管理设计理念**：
- Buffer管理的是**计算结果**的中间存储
- 输入数据（Q/K/V）直接从DDR加载到L0/L1，不需要buffer管理
- 只有只有计算结果需要buffer管理，因为需要在多次计算中复用

**PyPTO的优化设计理念**：
- 保持与ops-transformer完全一致的设计理念
- 添加HardEvent抽象，支持精确的同步控制
- 添加GetPre和GetReused方法，支持Q复用和KV复用
- 添加BufferInfo，提供硬件抽象
- **专注于训练算子，简化推理算子特有的API**

### 3.2 为什么这样设计？

**优势1：与ops-transformer完全一致**
- 用户可以无缝切换到PyPTO
- 生成的代码与ops-transformer生成的代码结构相同
- 可以直接参考ops-transformer的实现

**优势2：更精确的同步控制**
- 支持指定具体的HardEvent
- 区分生产者等待消费者和消费者等待生产者
- 更符合硬件特性

**优势3：更完整的复用支持**
- 支持Q复用（GetPre）
- 支持KV复用（GetReused）
- 支持独立轮转（GetVec/GetCube）

**优势4：专注于训练算子**
- 简化API，去掉推理算子特有的功能（如GetEVENTID）
- 更符合训练算子的实际需求
- 降低学习成本

### 3.3 与原设计的对比

| 对比项 | 原设计 | 优化设计（适配ops-transformer） |
|-------|-------|------------------------|
| HardEvent抽象 | ❌ 无 | ✅ 有 |
| BufferInfo | ❌ 无 | ✅ 有 |
| GetPre方法 | ❌ 无 | ✅ 有 |
| GetReused方法 | ❌ 无 | ✅ 有 |
| 跨核同步处理 | ⚠️ 简化 | ✅ 区分UB/GM和L1 |
| 一个AIC两个AIV | ❌ 未处理 | ✅ 已处理 |
| 与ops-transformer一致性 | ⚠️ 部分 | ✅ 完全一致 |
| 推理算子特有API | ⚠️ 混合 | ❌（专注于训练算子） |

---

## 4. API设计

### 4.1 设计原则

1. **简单易用**：小白也能快速上手
2. **循序渐进**：从简单到复杂，逐步学习
3. **一致性**：与ops-transformer保持一致
4. **灵活性**：支持多种使用场景
5. **精确同步**：支持HardEvent和跨核同步
6. **专注训练**：简化API，专注于训练算子需求

### 4.2 API层次

```
层次1：简单API（适合新手）
    ├─ create_buffer()：创建计算结果的buffer
    └─ create_tile()：创建输入数据的tile

层次2：高级API（适合专家）
    ├─ SingleBufferPolicy：单buffer策略
    ├─ DoubleBufferPolicy：双buffer策略
    ├─ TripleBufferPolicy：三buffer策略
    └─ QuadBufferPolicy：四buffer策略

层次3：同步API（高级同步）
    ├─ BufferInfo：Buffer硬件信息
    ├─ wait(HardEvent)：等待指定事件
    ├─ set(HardEvent)：设置指定事件
    ├─ set_cross_core()：设置跨核同步
    └─ wait_cross_core()：等待跨核同步
```

### 4.3 层次1：简单API

**用途**：适合简单的计算场景

```python
import pypto.frontend as fe
import pypto.language.manual as plm

@fe.kernel
def simple_kernel(
    a: plm.Tensor[[64, 128], plm.FP16],
    b: plm.Tensor[[64, 128], plm.FP16],
) -> plm.Tensor[[64, 128], plm.FP16]:
    # 第1步：创建计算结果的buffer
    # 就像准备一个托盘，放计算结果
    result_buffer = plm.create_buffer(
        [64, 128], plm.FP16,
        buffer_policy=plm.BufferPolicy.DOUBLE,  # 双缓冲，流水线并行
        buffer_type=plm.MemorySpace.Vec,     # 放到工作台
        sync_type=plm.SyncType.CROSS_CORE_SYNC_BOTH  # 跨核同步（双向，需要事件ID）
    )

    # 第2步：创建输入数据的tile
    # 就像准备两个小托盘，放输入数据
    a_tile = plm.create_tile([64, 128], plm.FP16, target_memory=plm.MemorySpace.Left)
    b_tile = plm.create_tile([64, 128], plm.FP16, target_memory=plm.MemorySpace.Right)

    # 第3步：加载输入数据
    # 就像把数据放到小托盘上
    plm.load(a, [0, 0], [64, 128], out=a_tile)
    plm.load(b, [0, 0], [64, 128], out=b_tile)

    # 第4步：计算
    # 就像把小托盘的数据放到大托盘上
    plm.matmul(a_tile, b_tile, out=result_buffer)

    # 第5步：返回结果
    return result_buffer
```

**关键点**：
- `create_buffer()`创建计算结果的buffer
- `create_tile()`创建输入数据的tile
- 计算结果使用buffer管理，支持流水线并行
- `sync_type=plm.SyncType.INNER_CORE_SYNC`需要事件ID

### 4.4 层次2：高级API

**用途**：适合复杂的计算场景（如Flash Attention）

```python
@fe.kernel
def flash_attention_kernel(
    q: plm.Tensor[[128, 128], plm.FP16],
    k: plm.Tensor[[128, 128], plm.FP16],
    v: plm.Tensor[[128, 128], plm.FP16],
) -> plm.Tensor[[128, 128], plm.FP16]:
    # 第1步：创建计算结果的buffer策略
    # 就像准备两个大托盘，放计算结果
    score_buffer = plm.DoubleBufferPolicy(
        [64, 128], plm.FP16,
        buffer_type=plm.MemorySpace.Vec,
        sync_type=plm.SyncType.CROSS_CORE_SYNC_BOTH
    )

    attention_buffer = plm.DoubleBufferPolicy(
        [64, 128], plm.FP16,
        buffer_type=plm.MemorySpace.Vec,
        sync_type=plm.SyncType.CROSS_CORE_SYNC_BOTH
    )

    # 第2步：创建输入数据的tile
    # 就像准备三个小托盘，放输入数据
    q_tile = plm.create_tile([64, 128], plm.FP16, target_memory=plm.MemorySpace.Left)
    k_tile = plm.create_tile([64, 128], plm.FP16, target_memory=plm.MemorySpace.Right)
    v_tile = plm.create_tile([64, 128], plm.FP16, target_memory=plm.MemorySpace.Right)

    # 第3步：加载和计算
    # 第1次计算
    plm.load(q, [0, 0], [64, 128], out=q_tile)
    plm.load(k, [0, 0], [64, 128], out=k_tile)
    plm.load(v, [0, 0], [64, 128], out=v_tile)

    score_tile = score_buffer.get()
    plm.matmul(q_tile, k_tile, out=score_tile)

    attention_tile = attention_buffer.get()
    plm.matmul(score_tile, v_tile, out=attention_tile)

    # 第2次计算（流水线并行）
    plm.load(q, [64, 0], [64, 128], out=q_tile)
    plm.load(k, [0, 0], [64, 128], out=k_tile)
    plm.load(v, [0, 0], [64, 128], out=v_tile)

    score_tile_next = score_buffer.get()
    plm.matmul(q_tile, k_tile, out=score_tile_next)

    attention_tile_next = attention_buffer.get()
    plm.matmul(score_tile_next, v_tile, out=attention_tile_next)

    return attention_tile_next
```

**关键点**：
- `DoubleBufferPolicy`管理计算结果
- `get()`方法轮转使用buffer
- 支持流水线并行
- `sync_type=plm.SyncType.CROSS_CORE_SYNC_BOTH`需要事件ID和跨核同步

### 4.5 层次3：同步API

**用途**：高级同步控制

```python
# 创建buffer策略
score_buffer = plm.DoubleBufferPolicy(
    [64, 128], plm.FP16,
    buffer_type=plm.MemorySpace.Vec,
    sync_type=plm.SyncType.INNER_CORE_SYNC
)

# 初始化buffer
score_buffer.init()

# 获取硬件事件
event_p2c = plm.BufferInfo.get_event_p2c(plm.MemorySpace.Vec)
event_c2p = plm.BufferInfo.get_event_c2p(plm.MemorySpace.Vec)

# 等待生产者完成生产
score_buffer.wait(event_p2c)

# 设置事件（生产者通知消费者）
score_buffer.set(event_p2c)

# 等待消费者完成消费
score_buffer.wait(event_c2p)

# 设置事件（消费者通知生产者）
score_buffer.set(event_c2p)

# 跨核同步
score_buffer.set_cross_core()
score_buffer.wait_cross_core()

# 释放buffer
score_buffer.uninit()
```

**关键点**：
- `init()`：初始化事件ID，设置初始标志
- `uninit()`：释放事件ID
- `wait(HardEvent)`：等待指定事件
- `set(HardEvent)`：设置指定事件
- `set_cross_core()`和`wait_cross_core()`：跨核同步

---

## 5. 实现步骤

### 5.1 第一步：理解需求

**需求**：
- 实现buffer管理功能
- 支持流水线并行
- 支持HardEvent和跨核同步
- 与ops-transformer保持一致
- **专注于训练算子**

**思考**：
- 需要哪些类型？（BufferPolicy, SyncType, HardEvent）
- 需要哪些类？（BufferInfo, Buffer, SingleBufferPolicy, DoubleBufferPolicy等）
- 需要哪些函数？（create_buffer, create_tile）
- **不需要推理算子特有的API（如GetEVENTID）**

### 5.2 第二步：设计类型系统

**文件**：`pypto/python/pypto/language/manual/buffer_type.py`

```python
from enum import Enum

class BufferPolicy(Enum):
    """Buffer策略枚举"""
    SINGLE = 0   # 单buffer（无乒乓）
    DOUBLE = 1   # 双buffer（ping-pong轮转）
    TRIPLE = 2   # 3 buffer（轮转）
    QUAD = 3     # 4 buffer（轮转）


class SyncType(Enum):
    """同步类型枚举"""
    NO_SYNC = 0                  # 不同步
    INNER_CORE_SYNC = 1          # 核内同步（需要事件ID）
    CROSS_CORE_SYNC_FORWARD = 2  # 核间同步（单向，需要事件ID）
    CROSS_CORE_SYNC_BOTH = 3     # 核间同步（双向，需要事件ID）


class HardEvent(Enum):
    """硬件事件类型"""
    L1_P2C = 0   # L1：生产者到消费者（MTE2_MTE1）
    L1_C2P = 1   # L1：消费者到生产者（MTE1_MTE2）
    L0_P2C = 2   # L0A/L0B：生产者到消费者（MTE1_M）
    L0_C2P = 3   # L0A/L0B：消费者到生产者（M_MTE1）
    ACC_P2C = 4  # L0C：生产者到消费者（M_FIX）
    ACC_C2P = 5  # L0C：消费者到生产者（FIX_M）
```

### 5.3 第三步：实现BufferInfo类

**文件**：`pypto/python/pypto/language/manual/buffer_info.py`

```python
from pypto.pypto_core.ir import MemorySpace

class BufferInfo:
    """Buffer硬件信息"""

    @staticmethod
    def get_event_p2c(buffer_type: MemorySpace) -> HardEvent:
        """获取生产者到消费者的HardEvent"""
        mapping = {
            MemorySpace.Mat: HardEvent.L1_P2C,
            MemorySpace.Left: HardEvent.L0_P2C,
            MemorySpace.Right: HardEvent.L0_P2C,
            MemorySpace.Acc: HardEvent.ACC_P2C,
        }
        return mapping.get(buffer_type)

    @staticmethod
    def get_event_c2p(buffer_type: MemorySpace) -> HardEvent:
        """获取消费者到生产者的HardEvent"""
        mapping = {
            MemorySpace.Mat: HardEvent.L1_C2P,
            MemorySpace.Left: HardEvent.L0_C2P,
            MemorySpace.Right: HardEvent.L0_C2P,
            MemorySpace.Acc: HardEvent.ACC_C2P,
        }
        return mapping.get(buffer_type)
```

### 5.4 第四步：实现Buffer类

**文件**：`pypto/python/pypto/language/manual/buffer.py`

```python
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import MemorySpace
from pypto.ir.op import block_ops as _ir_block_ops

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
        
        # 硬件信息
        self._event_p2c = BufferInfo.get_event_p2c(buffer_type)
        self._event_c2p = BufferInfo.get_event_c2p(buffer_type)
        
        # 内部映射MemorySpace到硬件位置（不暴露给用户）
        _position_map = {
            MemorySpace.Mat: 0,    # A1
            MemorySpace.Left: 1,   # A2
            MemorySpace.Right: 2,  # B2
            MemorySpace.Acc: 3,    # CO1
            MemorySpace.Vec: 4,    # VECIN
            MemorySpace.DDR: 5,    # GM
        }
        self._position = _position_map.get(buffer_type)

        # 事件ID
        self._p2c_event_id = None
        self._c2p_event_id = None
        self._id0 = None  # 跨核同步ID（正向）
        self._id1 = None  # 跨核同步ID（反向）

        # 创建tensor
        self._tensor = self._create_tensor()

    def init(self):
        """初始化事件ID，设置初始标志"""
        if self.sync_type == SyncType.INNER_CORE_SYNC:
            self._p2c_event_id = _ir_block_ops.alloc_event_id()
            self._c2p_event_id = _ir_block_ops.alloc_event_id()
            # 设置初始标志（消费者到生产者）
            _ir_block_ops.set_flag(self._event_c2p, self._c2p_event_id)
        elif (self.sync_type == SyncType.CROSS_CORE_SYNC_FORWARD or
              self.sync_type == SyncType.CROSS_CORE_SYNC_BOTH):
            # 初始化跨核同步事件ID
            self._id0 = _ir_block_ops.alloc_event_id()
            self._id1 = _ir_block_ops.alloc_event_id()

    def uninit(self):
        """释放事件ID"""
        if self.sync_type == SyncType.INNER_CORE_SYNC:
            if self._c2p_event_id is not None:
                _ir_block_ops.wait_flag(self._event_c2p, self._c2p_event_id)
                _ir_block_ops.release_event_id(self._event_p2c, self._p2c_event_id)
                _ir_block_ops.release_event_id(self._event_c2p, self._c2p_event_id)
        elif (self.sync_type == SyncType.CROSS_CORE_SYNC_FORWARD or
              self.sync_type == SyncType.CROSS_CORE_SYNC_BOTH):
            # 释放跨核同步事件ID
            if self._id0 is not None:
                _ir_block_ops.release_event_id(self._id0)
            if self._id1 is not None:
                _ir_block_ops.release_event_id(self._id1)

    def wait(self, event_type: int):
        """等待事件（指定事件类型）

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
        """设置事件（指定事件类型）

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
        """设置跨核同步"""
        if self.sync_type in [SyncType.CROSS_CORE_SYNC_FORWARD,
                            SyncType.CROSS_CORE_SYNC_BOTH]:
            if self.buffer_type in [MemorySpace.Vec, MemorySpace.DDR]:
                self._set_cross_core_ub_gm()
            elif self.buffer_type == MemorySpace.Mat:
                self._set_cross_core_l1()

    def wait_cross_core(self):
        """等待跨核同步"""
        if self.sync_type in [SyncType.CROSS_CORE_SYNC_FORWARD,
                            SyncType.CROSS_CORE_SYNC_BOTH]:
            if self.buffer_type in [MemorySpace.Vec, MemorySpace.DDR]:
                self._wait_cross_core_ub_gm()
            elif self.buffer_type == MemorySpace.Mat:
                self._wait_cross_core_l1()

    def _set_cross_core_ub_gm(self):
        """UB/GM的跨核同步设置
        
        参考ops-transformer实现：
        - AIC是生产者，AIV是消费者
        - 一个AIC对应两个AIV（AIV0和AIV1）
        - 使用PIPE_FIX和PIPE_V进行同步
        - 使用mode=4进行跨核同步
        """
        if self._id0 is not None and self._id1 is not None:
            is_aic = _ir_system_ops.is_aic()
            is_aiv = _ir_system_ops.is_aiv()
            
            if is_aic:
                _ir_system_ops.cross_core_set_flag(mode=4, pipe=PipeType.FIX, event_id=self._id0)
                _ir_system_ops.cross_core_set_flag(mode=4, pipe=PipeType.FIX, event_id=self._id0 + 16)
            if is_aiv:
                _ir_system_ops.cross_core_set_flag(mode=4, pipe=PipeType.V, event_id=self._id1)

    def _wait_cross_core_ub_gm(self):
        """UB/GM的跨核同步等待
        
        参考ops-transformer实现：
        - AIC是生产者，AIV是消费者
        - 一个AIC对应两个AIV（AIV0和AIV1）
        - 使用PIPE_FIX和PIPE_V进行同步
        - 使用mode=4进行跨核同步
        """
        if self._id0 is not None and self._id1 is not None:
            is_aic = _ir_system_ops.is_aic()
            is_aiv = _ir_system_ops.is_aiv()
            
            if is_aic:
                _ir_system_ops.cross_core_wait_flag(mode=4, pipe=PipeType.FIX, event_id=self._id1)
                _ir_system_ops.cross_core_wait_flag(mode=4, pipe=PipeType.FIX, event_id=self._id1 + 16)
            if is_aiv:
                _ir_system_ops.cross_core_wait_flag(mode=4, pipe=PipeType.V, event_id=self._id0)

    def _set_cross_core_l1(self):
        """L1的跨核同步设置
        
        参考ops-transformer实现：
        - AIC是消费者，AIV是生产者
        - 一个AIC对应两个AIV（AIV0和AIV1）
        - 使用PIPE_MTE1和PIPE_MTE3进行同步
        - 使用mode=4进行跨核同步
        - CROSS_CORE_SYNC_BOTH需要双向同步
        """
        if self._id0 is not None and self._id1 is not None:
            is_aic = _ir_system_ops.is_aic()
            is_aiv = _ir_system_ops.is_aiv()
            
            if is_aic and self.sync_type == SyncType.CROSS_CORE_SYNC_BOTH:
                _ir_system_ops.cross_core_set_flag(mode=4, pipe=PipeType.MTE1, event_id=self._id1)
                _ir_system_ops.cross_core_set_flag(mode=4, pipe=PipeType.MTE1, event_id=self._id1 + 16)
            if is_aiv:
                _ir_system_ops.cross_core_set_flag(mode=4, pipe=PipeType.MTE3, event_id=self._id0)

    def _wait_cross_core_l1(self):
        """L1的跨核同步等待
        
        参考ops-transformer实现：
        - AIC是消费者，AIV是生产者
        - 一个AIC对应两个AIV（AIV0和AIV1）
        - 使用PIPE_MTE1和PIPE_MTE3进行同步
        - 使用mode=4进行跨核同步
        - CROSS_CORE_SYNC_BOTH需要双向同步
        """
        if self._id0 is not None and self._id1 is not None:
            is_aic = _ir_system_ops.is_aic()
            is_aiv = _ir_system_ops.is_aiv()
            
            if is_aic:
                _ir_system_ops.cross_core_wait_flag(mode=4, pipe=PipeType.MTE1, event_id=self._id0)
                if self.sync_type == SyncType.CROSS_CORE_SYNC_BOTH:
                    _ir_system_ops.cross_core_wait_flag(mode=4, pipe=PipeType.MTE1, event_id=self._id0 + 16)
            if is_aiv and self.sync_type == SyncType.CROSS_CORE_SYNC_BOTH:
                _ir_system_ops.cross_core_wait_flag(mode=4, pipe=PipeType.MTE3, event_id=self._id1)

    def get_tensor(self) -> Tensor:
        """获取tensor"""
        return self._tensor
```

**注意**：
- 同步相关的方法（init、uninit、wait、set、set_cross_core、wait_cross_core等）已实现
- `wait()` 和 `set()` 方法调用 `system_ops.sync_dst/sync_src` 实现核内同步
- `set_cross_core()` 和 `wait_cross_core()` 方法调用 `system_ops.cross_core_set_flag/cross_core_wait_flag` 实现跨核同步
- 跨核同步的详细实现（区分UB/GM和L1）已根据ops-transformer完整实现
- 跨核同步使用mode=4进行同步，区分AIC和AIV的角色
- 一个AIC对应两个AIV（AIV0和AIV1），使用event_id和event_id+16进行同步
- 使用`is_aic()`和`is_aiv()`函数区分核心类型，根据核心类型选择正确的同步操作

**完整实现说明**：

1. **核心类型判断**：
   - 添加了`is_aic()`函数判断当前核心是否为AIC
   - 添加了`is_aiv()`函数判断当前核心是否为AIV
   - 在跨核同步时根据核心类型选择正确的同步操作

2. **UB/GM跨核同步的完整实现**（参考ops-transformer）：
   - AIC（生产者）：`SetCrossCore()` 执行 `PIPE_FIX(id0_)` + `PIPE_FIX(id0_ + 16)`
   - AIV（消费者）：`SetCrossCore()` 执行 `PIPE_V(id1_)`
   - AIC（生产者）：`WaitCrossCore()` 执行 `PIPE_FIX(id1_)` + `PIPE_FIX(id1_ + 16)`
   - AIV（消费者）：`WaitCrossCore()` 执行 `PIPE_V(id0_)`

3. **L1跨核同步的完整实现**（参考ops-transformer）：
   - AIC（消费者）：`SetCrossCore()` 执行 `PIPE_MTE1(id1_)` + `PIPE_MTE1(id1_ + 16)`（仅CROSS_CORE_SYNC_BOTH）
   - AIV（生产者）：`SetCrossCore()` 执行 `PIPE_MTE3(id0_)`
   - AIC（消费者）：`WaitCrossCore()` 执行 `PIPE_MTE1(id0_)` + `PIPE_MTE1(id0_ + 16)`（仅CROSS_CORE_SYNC_BOTH）
   - AIV（生产者）：`WaitCrossCore()` 执行 `PIPE_MTE3(id1_)`（仅CROSS_CORE_SYNC_BOTH）

### 5.5 第五步：实现Buffer策略类

**文件**：`python/pypto/language/manual/buffer_policy.py`

```python
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

    def init(self, buffer_manager: BufferManager):
        """初始化"""
        self.buffer.init()

    def uninit(self, buffer_manager: BufferManager):
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

    def init(self, buffer_manager: BufferManager):
        """初始化"""
        self.ping.init()
        self.pong.init()

    def uninit(self, buffer_manager: BufferManager):
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

    def init(self, buffer_manager: BufferManager):
        """初始化"""
        self.a.init()
        self.b.init()
        self.c.init()

    def uninit(self, buffer_manager: BufferManager):
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

    def init(self, buffer_manager: BufferManager):
        """初始化"""
        self.a.init()
        self.b.init()
        self.c.init()
        self.d.init()

    def uninit(self, buffer_manager: BufferManager):
        """释放"""
        self.a.uninit()
        self.b.uninit()
        self.c.uninit()
        self.d.uninit()

    def get(self, id: int = None) -> Buffer:
        """获取buffer（按ID或队列模式）"""
        if id is not None:
            return self._get_by_id(id)
        else:
            buffer = self._get_by_id(self._head)
            self._head += 1
            return buffer

    def get_reused(self) -> Buffer:
        """获取复用buffer（队列复用）"""
        buffer = self._get_by_id(self._used)
        self._used = (self._used - self._tail + 1) % (self._head - self._tail) + self._tail
        return buffer

    def get_free(self) -> Buffer:
        """获取空闲buffer"""
        if self._tail == self._used:
            self._used += 1
        buffer = self._get_by_id(self._tail)
        self._tail += 1
        return buffer

    def _get_by_id(self, id: int) -> Buffer:
        """根据ID获取buffer"""
        flag = id % 4
        if flag == 0:
            return self.a
        elif flag == 1:
            return self.b
        elif flag == 2:
            return self.c
        else:
            return self.d
```

### 5.6 第六步：实现create_buffer函数

**文件**：`pypto/python/pypto/language/manual/op/manual_ops.py`

```python
from pypto.pypto_core import DataType
from pypto.pypto_core.ir import MemorySpace
from pypto.ir.op import block_ops as _ir_block_ops
from ..typing import Tile

def create_buffer(
    shape: list[int],
    dtype: DataType,
    buffer_policy: BufferPolicy = BufferPolicy.SINGLE,
    buffer_type: MemorySpace = MemorySpace.Vec,
    sync_type: SyncType = SyncType.INNER_CORE_SYNC,
) -> Tile:
    """Create a buffer for computation results.

    Args:
        shape: Buffer shape.
        dtype: Element data type.
        buffer_policy: Buffer policy (default SINGLE).
        buffer_type: Memory space (default UB).
        sync_type: Sync type (default INNER_CORE_SYNC).

    Returns:
        Tile wrapping of buffer allocation expression.
    """
    return Tile(expr=_ir_block_ops.create_tile(
        shape, dtype, buffer_type=buffer_type,
        buffer_policy=buffer_policy, sync_type=sync_type
    ))
```

### 5.7 第七步：导出API

**文件**：`pypto/python/pypto/language/manual/__init__.py`

```python
from .op.buffer_type import (
    BufferPolicy,
    SyncType,
    HardEvent,
)
from .op.buffer_info import BufferInfo
from .op.buffer_policy import (
    SingleBufferPolicy,
    DoubleBufferPolicy,
    TripleBufferPolicy,
    QuadBufferPolicy,
)
from .op.manual_ops import create_buffer

__all__ = [
    # Buffer types
    "BufferPolicy",
    "SyncType",
    "HardEvent",

    # Buffer info
    "BufferInfo",

    # Buffer policies
    "SingleBufferPolicy",
    "DoubleBufferPolicy",
    "TripleBufferPolicy",
    "QuadBufferPolicy",

    # Buffer functions
    "create_buffer",
]
```

---

## 6. 实践示例

### 6.1 示例1：简单的矩阵加法

**场景**：计算两个矩阵的和

```python
@fe.kernel
def add_kernel(
    a: plm.Tensor[[64, 128], plm.FP16],
    b: plm.Tensor[[64, 128], plm.FP16],
) -> plm.Tensor[[64, 128], plm.FP16]:
    # 创建计算结果的buffer
    result_buffer = plm.create_buffer(
        [64, 128], plm.FP16,
        buffer_policy=plm.BufferPolicy.SINGLE,
        buffer_type=plm.MemorySpace.Vec,
        sync_type=plm.SyncType.INNER_CORE_SYNC
    )

    # 创建输入数据的tile
    a_tile = plm.create_tile([64, 128], plm.FP16, target_memory=plm.MemorySpace.Left)
    b_tile = plm.create_tile([64, 128], plm.FP16, target_memory=plm.MemorySpace.Right)

    # 加载输入数据
    plm.load(a, [0, 0], [64, 128], out=a_tile)
    plm.load(b, [0, 0], [64, 128], out=b_tile)

    # 计算
    plm.add(a_tile, b_tile, out=result_buffer)

    return result_buffer
```

**步骤说明**：
1. 创建计算结果的buffer（单缓冲）
2. 创建输入数据的tile
3. 加载输入数据
4. 计算矩阵加法
5. 返回结果

### 6.2 示例2：分块矩阵乘法

**场景**：计算两个大矩阵的乘法（分块计算）

```python
@fe.kernel
def matmul_kernel(
    a: plm.Tensor[[128, 128], plm.FP16],
    b: plm.Tensor[[128, 128], plm.FP16],
) -> plm.Tensor[[128, 128], plm.FP16]:
    # 创建计算结果的buffer（双缓冲，流水线并行）
    result_buffer = plm.DoubleBufferPolicy(
        [64, 128], plm.FP16,
        buffer_type=plm.MemorySpace.Vec,
        sync_type=plm.SyncType.CROSS_CORE_SYNC_BOTH
    )

    # 创建输入数据的tile
    a_tile = plm.create_tile([64, 128], plm.FP16, target_memory=plm.MemorySpace.Left)
    b_tile = plm.create_tile([64, 128], plm.FP16, target_memory=plm.MemorySpace.Right)

    # 分块计算
    result_tile = None
    for i in range(0, 2):
        for j in range(0, 2):
            # 加载输入数据
            plm.load(a, [i*64, 0], [64, 128], out=a_tile)
            plm.load(b, [0, j*64], [128, 64], out=b_tile)

            # 计算结果使用buffer管理
            result_tile = result_buffer.get()
            plm.matmul(a_tile, b_tile, out=result_tile)

    return result_tile
```

**步骤说明**：
1. 创建计算结果的buffer（双缓冲）
2. 创建输入数据的tile
3. 分块加载和计算
4. 使用buffer管理计算结果
5. 返回结果

### 6.3 示例3：Flash Attention（简化版）

**场景**：实现Flash Attention的核心计算

```python
@fe.kernel
def flash_attention_kernel(
    q: plm.Tensor[[128, 128], plm.FP16],
    k: plm.Tensor[[128, 128], plm.FP16],
    v: plm.Tensor[[128, 128], plm.FP16],
) -> plm.Tensor[[128, 128], plm.FP16]:
    # 创建计算结果的buffer策略
    score_buffer = plm.DoubleBufferPolicy(
        [64, 128], plm.FP16,
        buffer_type=plm.MemorySpace.Vec,
        sync_type=plm.SyncType.CROSS_CORE_SYNC_BOTH
    )

    attention_buffer = plm.DoubleBufferPolicy(
        [64, 128], plm.FP16,
        buffer_type=plm.MemorySpace.Vec,
        sync_type=plm.SyncType.CROSS_CORE_SYNC_BOTH
    )

    # K/V在L1上的复用buffer（3个buffer）
    kv_reuse_buffer = plm.TripleBufferPolicy(
        [64, 128], plm.FP16,
        buffer_type=plm.MemorySpace.Mat,
        sync_type=plm.SyncType.CROSS_CORE_SYNC_FORWARD
    )

    # 初始化buffer
    score_buffer.init()
    attention_buffer.init()
    kv_reuse_buffer.init()

    # 创建输入数据的tile
    q_tile = plm.create_tile([64, 128], plm.FP16, target_memory=plm.MemorySpace.Left)
    k_tile = plm.create_tile([64, 128], plm.FP16, target_memory=plm.MemorySpace.Right)
    v_tile = plm.create_tile([64, 128], plm.FP16, target_memory=plm.MemorySpace.Right)

    # 最终结果
    final_result = plm.create_tile([128, 128], plm.FP16, target_memory=plm.MemorySpace.DDR)

    # Flash Attention循环
    attention_tile = None
    for s2_idx in range(0, 2):
        # 加载Q
        plm.load(q, [s2_idx*64, 0], [64, 128], out=q_tile)

        # 加载K/V（第一次加载，后续复用）
        if s2_idx == 0:
            # 第一次加载K/V到L1 buffer
            k_tile_l1 = kv_reuse_buffer.get()
            v_tile_l1 = kv_reuse_buffer.get()
            plm.load(k, [0, 0], [64, 128], out=k_tile_l1)
            plm.load(v, [0, 0], [64, 128], out=v_tile_l1)
        else:
            # 后续计算，复用之前加载的K/V buffer
            k_tile_l1 = kv_reuse_buffer.get_reused()
            v_tile_l1 = kv_reuse_buffer.get_reused()
            # 不需要重新加载，直接使用复用的buffer

        # 计算Q × K^T
        score_tile = score_buffer.get()
        plm.matmul(q_tile, k_tile_l1, out=score_tile)

        # Softmax（简化）
        # plm.softmax(score_tile, out=score_tile)

        # 计算Score × V
        attention_tile = attention_buffer.get()
        plm.matmul(score_tile, v_tile_l1, out=attention_tile)

        # 累加到最终结果
        plm.add(final_result, attention_tile, out=final_result)

    # 释放buffer
    score_buffer.uninit()
    attention_buffer.uninit()
    kv_reuse_buffer.uninit()

    return final_result
```

**步骤说明**：
1. 创建计算结果的buffer策略
2. 创建输入数据的tile
3. 初始化buffer
4. 循环加载和计算
5. 使用buffer管理计算结果
6. 使用GetReused实现KV复用（第一次加载K/V，后续复用）
7. 累加到最终结果
8. 释放buffer

### 6.4 示例4：使用HardEvent同步

**场景**：高级同步控制

```python
@fe.kernel
def sync_kernel(
    a: plm.Tensor[[64, 128], plm.FP16],
    b: plm.Tensor[[64, 128], plm.FP16],
) -> plm.Tensor[[64, 128], plm.FP16]:
    # 创建计算结果的buffer
    result_buffer = plm.DoubleBufferPolicy(
        [64, 128], plm.FP16,
        buffer_type=plm.MemorySpace.Vec,
        sync_type=plm.SyncType.INNER_CORE_SYNC
    )

    # 初始化buffer
    result_buffer.init()

    # 获取硬件事件
    event_p2c = plm.BufferInfo.get_event_p2c(plm.MemorySpace.Vec)
    event_c2p = plm.BufferInfo.get_event_c2p(plm.MemorySpace.Vec)

    # 创建输入数据的tile
    a_tile = plm.create_tile([64, 128], plm.FP16, target_memory=plm.MemorySpace.Left)
    b_tile = plm.create_tile([64, 128], plm.FP16, target_memory=plm.MemorySpace.Right)

    # 加载输入数据
    plm.load(a, [0, 0], [64, 128], out=a_tile)
    plm.load(b, [0, 0], [64, 128], out=b_tile)

    # 等待生产者完成生产
    result_buffer.wait(event_p2c)

    # 计算
    result_tile = result_buffer.get()
    plm.matmul(a_tile, b_tile, out=result_tile)

    # 设置事件（生产者通知消费者）
    result_buffer.set(event_p2c)

    # 等待消费者完成消费
    result_buffer.wait(event_c2p)

    # 设置事件（消费者通知生产者）
    result_buffer.set(event_c2p)

    # 释放buffer
    result_buffer.uninit()

    return result_tile
```

**步骤说明**：
1. 创建计算结果的buffer
2. 初始化buffer
3. 获取硬件事件
4. 加载输入数据
5. 等待生产者完成生产
6. 计算
7. 设置事件（生产者通知消费者）
8. 等待消费者完成消费
9. 设置事件（消费者通知生产者）
10. 释放buffer
11. 返回结果

---

## 7. 同步机制详解

### 7.1 为什么需要同步？

**比喻**：多个厨师共享一个厨房

**场景**：
- 厨师A负责切菜（生产者）
- 厨师B负责炒菜（消费者）
- 两个厨师共享一个厨房（NPU）

**问题**：
- 厨师A切好菜后，如何通知厨师B可以开始炒菜？
- 厨师B炒完菜后，如何通知厨师A可以切下一道菜？

**解决**：使用事件ID和同步信号

### 7.2 HardEvent的作用

**什么是HardEvent？**

HardEvent是NPU硬件提供的事件类型，用于精确控制同步。

**比喻**：
- 每个Buffer有两个门铃：生产者到消费者、消费者到生产者
- 生产者按门铃通知消费者数据已准备好
- 消费者按门铃通知生产者数据已消费

**HardEvent的类型**：

| Buffer类型 | 生产者到消费者 | 消费者到生产者 |
|-----------|---------------|---------------|
| L1 | L1_P2C (MTE2_MTE1) | L1_C2P (MTE1_MTE2) |
| L0A | L0_P2C (MTE1_M) | L0_C2P (M_MTE1) |
| L0B | L0_P2C (MTE1_M) | L0_C2P (M_MTE1) |
| L0C | ACC_P2C (M_FIX) | ACC_C2P (FIX_M) |

**关键点**：
- ✅ 不同的Buffer类型使用不同的HardEvent
- ✅ 生产者到消费者和消费者到生产者是不同的HardEvent
- ✅ BufferInfo自动根据BufferType选择正确的HardEvent

### 7.3 跨核同步的作用

**什么是跨核同步？**

跨核同步就是协调多个核心之间的数据访问。

**比喻**：
- 多个厨师（核心）共享一个厨房（NPU）
- 每个厨师有一个门牌号（事件ID）
- 厨师A完成切菜后，用门牌号通知所有厨师可以开始炒菜

**跨核同步的特殊处理**：

| Buffer类型 | AIC角色 | AIV角色 |
|-----------|---------|---------|
| UB/GM | 生产者 | 消费者 |
| L1 | 消费者 | 生产者 |

**关键点**：
- ✅ UB/GM：AIC是生产者，AIV是消费者
- ✅ L1：AIC是消费者，AIV是生产者
- ✅ 一个AIC对应两个AIV（AIV0和AIV1）
- ✅ 需要分别同步AIV0和AIV1

### 7.4 同步流程

**核内同步流程**：

```
1. 初始化buffer（Init）
2. 分配事件ID
3. 设置初始标志（SetFlag）
4. 等待事件（Wait<HardEvent>）
5. 执行计算
6. 设置事件（Set<HardEvent>）
7. 重复步骤4-6
8. 释放buffer（UnInit）
```

**跨核同步流程**：

```
1. AIC（生产者）：
    - 设置跨核同步（SetCrossCore）
    - 执行计算
    - 通知AIV可以开始（SetCrossCore）

2. AIV（消费者）：
    - 等待跨核同步（WaitCrossCore）
    - 执行计算
    - 通知AIC可以继续（SetCrossCore）
```

### 7.5 PyPTO中的同步API

**核内同步**：

```python
# 创建buffer策略
buffer = plm.DoubleBufferPolicy(
    [64, 128], plm.FP16,
    buffer_type=plm.MemorySpace.Vec,
    sync_type=plm.SyncType.INNER_CORE_SYNC
)

# 初始化buffer
buffer.init()

# 获取硬件事件
event_p2c = plm.BufferInfo.get_event_p2c(plm.MemorySpace.Vec)
event_c2p = plm.BufferInfo.get_event_c2p(plm.MemorySpace.Vec)

# 等待生产者完成生产
buffer.wait(event_p2c)

# 设置事件（生产者通知消费者）
buffer.set(event_p2c)

# 等待消费者完成消费

buffer.wait(event_c2p)

# 设置事件（消费者通知生产者）
buffer.set(event_c2p)

# 释放buffer
buffer.uninit()
```

**跨核同步**：

```python
# 创建buffer策略
buffer = plm.DoubleBufferPolicy(
    [64, 128], plm.FP16,
    buffer_type=plm.MemorySpace.Vec,
    sync_type=plm.SyncType.CROSS_CORE_SYNC_BOTH
)

# 初始化buffer
buffer.init()

# 设置跨核同步
buffer.set_cross_core()

# 等待跨核同步
buffer.wait_cross_core()

# 释放buffer
buffer.uninit()
```

---

## 8. 常见问题

### 8.1 Q1：什么时候用SingleBuffer？

**A**：当计算结果不需要流水线并行时。

**示例**：
```python
# 简单的矩阵加法，不需要流水线并行
result_buffer = plm.create_buffer(
    [64, 128], plm.FP16,
    buffer_policy=plm.BufferPolicy.SINGLE
)
```

### 8.2 Q2：什么时候用DoubleBuffer？

**A**：当计算结果需要流水线并行时。

**示例**：
```python
# 分块矩阵乘法，需要流水线并行
result_buffer = plm.DoubleBufferPolicy(
    [64, 128], plm.FP16,
    buffer_policy=plm.BufferPolicy.DOUBLE
)
```

### 8.3 Q3：create_buffer和create_tile有什么区别？

**A**：
- `create_buffer`：创建计算结果的buffer，支持buffer策略
- `create_tile`：创建输入数据的tile，不支持buffer策略

**示例**：
```python
# 创建计算结果的buffer
result_buffer = plm.create_buffer(
    [64, 128], plm.FP16,
    buffer_policy=plm.BufferPolicy.DOUBLE
)

# 创建输入数据的tile
a_tile = plm.create_tile([64, 128], plm.FP16)
```

### 8.4 Q4：什么时候用get()，什么时候用get_pre()？

**A**：
- `get()`：获取当前buffer（轮转）
- `get_pre()`：获取前一个buffer（Q复用）

**示例**：
```python
# 获取当前buffer
score_tile = score_buffer.get()

# 获取前一个buffer（Q复用）
score_tile_prev = score_buffer.get_pre()
```

### 8.5 Q5：如何选择SyncType？

**A**：
- `NO_SYNC`：不需要同步
- `INNER_CORE_SYNC`：核内同步（需要事件ID）
- `CROSS_CORE_SYNC_FORWARD`：核间同步（单向，需要事件ID）
- `CROSS_CORE_SYNC_BOTH`：核间同步（双向，需要事件ID）

**示例**：
```python
# 核内同步
result_buffer = plm.create_buffer(
    [64, 128], plm.FP16,
    sync_type=plm.SyncType.INNER_CORE_SYNC
)

# 核间同步（双向）
result_buffer = plm.create_buffer(
    [64, 128], plm.FP16,
    sync_type=plm.SyncType.CROSS_CORE_SYNC_BOTH
)
```

### 8.6 Q6：什么时候需要调用init()和uninit()？

**A**：当使用BuffersPolicy类时。

**示例**：
```python
# 创建buffer策略
buffer = plm.DoubleBufferPolicy(
    [64, 128], plm.FP16,
    sync_type=plm.SyncType.INNER_CORE_SYNC
)

# 初始化buffer
buffer.init()

# 使用buffer
tile = buffer.get()

# 释放buffer
buffer.uninit()
```

### 8.7 Q7：如何使用HardEvent？

**A**：通过BufferInfo获取HardEvent，然后用于wait和set。

**示例**：
```python
# 获取硬件事件
event_p2c = plm.BufferInfo.get_event_p2c(plm.MemorySpace.Vec)
event_c2p = plm.BufferInfo.get_event_c2p(plm.MemorySpace.Vec)

# 等待生产者完成生产
buffer.wait(event_p2c)

# 设置事件（生产者通知消费者）
buffer.set(event_p2c)
```

### 8.8 Q8：如何选择MemorySpace？

**A**：
- `plm.MemorySpace.Vec`：统一缓冲区（工作台）
- `plm.MemorySpace.Mat`：L1缓存（大托盘）
- `plm.MemorySpace.Left/Right/Acc`：L0缓冲区（小托盘）
- `plm.MemorySpace.DDR`：DDR内存（片外，大仓库）

**示例**：
```python
# 计算结果放到工作台
result_buffer = plm.create_buffer(
    [64, 128], plm.FP16,
    buffer_type=plm.MemorySpace.Vec
)

# 输入数据放到小托盘
a_tile = plm.create_tile([64, 128], plm.FP16, target_memory=plm.MemorySpace.Left)
```

### 8.9 Q9：GetReused有什么用？

**A**：GetReused用于KV复用，在Flash Attention中复用K/V buffer，避免重复加载数据。

**示例**：
```python
# 第1次计算：加载K/V到L1 buffer
k_tile = kv_reuse_buffer.get()
v_tile = kv_reuse_buffer.get()
plm.load(k, [0, 0], [64, 128], out=k_tile)
plm.load(v, [0, 0], [64, 128], out=v_tile)

# 第2次计算：复用之前加载的K/V buffer
k_tile_reused = kv_reuse_buffer.get_reused()
v_tile_reused = kv_reuse_buffer.get_reused()
# 不需要重新加载K/V，直接使用复用的buffer进行计算
plm.matmul(q_tile, k_tile_reused, out=score_tile)
plm.matmul(score_tile, v_tile_reused, out=attention_tile)
```

### 8.10 Q10：GetVec和GetCube有什么用？

**A**：GetVec和GetCube用于独立轮转，在TripleBufferPolicy中分别用于vec计算和cube计算。

**示例**：
```python
# vec计算
buffer_vec = kv_reuse_buffer.get_vec()

# cube计算
buffer_cube = kv_reuse_buffer.get_cube()
```

---

## 9. 总结

### 9.1 核心要点

1. **Buffer管理的是计算结果**，不是输入数据
2. **输入数据直接加载到L0/L1，不需要buffer管理**
3. **计算结果使用buffer管理，支持流水线并行**
4. **与ops-transformer保持一致的设计理念**
5. **支持HardEvent和跨核同步**
6. **支持GetPre和GetReused复用机制**
7. **专注于训练算子，简化API**

### 9.2 学习路径

1. **第一步**：理解核心概念（Buffer, BufferPolicy, SyncType, HardEvent）
2. **第二步**：学习简单API（create_buffer, create_tile）
3. **第三步**：实践简单示例（矩阵加法、矩阵乘法）
4. **第四步**：学习高级API（DoubleBufferPolicy等）
5. **第五步**：实践复杂示例（Flash Attention）
6. **第六步**：学习同步API（BufferInfo, wait, set, set_cross_core, wait_cross_core）
7. **第七步**：深入理解实现细节

### 9.3 下一步

1. 实现BufferPolicy和SyncType枚举
2. 实现HardEvent枚举
3. 实现BufferInfo类
4. 实现Buffer类
5. 实现Buffer策略类（Single/Double/Triple/Quad）
6. 实现create_buffer和create_tile函数
7. 扩展C++ IR和Codegen
8. 编写测试用例
9. 验证生成的MLIR正确性

---

## 实现注意事项

### 1. 函数名称一致性

**重要**：在PyPTO中，创建tile的函数是`make_tile()`，而不是`create_tile()`。

- ✅ 正确：`_ir_block_ops.make_tile()`
- ❌ 错误：`_ir_block_ops.create_tile()`

### 2. 参数名称一致性

**重要**：`make_tile()`函数的参数名称是`target_memory`，而不是`buffer_type`。

- `make_tile(shape, dtype, target_memory=...)` ✅
- `make_tile(shape, dtype, buffer_type=...)` ❌

### 3. 枚举值转换

**重要**：当传递枚举类型给IR层时，需要使用`.value`属性转换为int。

```python
# 正确
buffer_policy=buffer_policy.value,  # BufferPolicy -> int
sync_type=sync_type.value           # SyncType -> int

# 错误
buffer_policy=buffer_policy,  # BufferPolicy类型
sync_type=sync_type           # SyncType类型
```

### 4. 全局函数vs实例方法

**重要**：在`buffer.py`中，`_alloc_event_id()`和`_release_event_id()`是全局函数，不是实例方法。

```python
# 正确
self._p2c_event_id = _alloc_event_id()
_release_event_id(self._p2c_event_id)

# 错误
self._p2c_event_id = self._alloc_event_id()
self._release_event_id(self._p2c_event_id)
```

---

## 实现注意事项

### 1. 函数名称一致性

**重要**：在PyPTO中，创建tile的函数是`make_tile()`，而不是`create_tile()`。

- ✅ 正确：`_ir_block_ops.make_tile()`
- ❌ 错误：`_ir_block_ops.create_tile()`

### 2. 参数名称一致性

**重要**：`make_tile()`函数的参数名称是`target_memory`，而不是`buffer_type`。

- `make_tile(shape, dtype, target_memory=...)` ✅
- `make_tile(shape, dtype, buffer_type=...)` ❌

### 3. 枚举值转换

**重要**：当传递枚举类型给IR层时，需要使用`.value`属性转换为int。

```python
# 正确
buffer_policy=buffer_policy.value,  # BufferPolicy -> int
sync_type=sync_type.value           # SyncType -> int

# 错误
buffer_policy=buffer_policy,  # BufferPolicy类型
sync_type=sync_type           # SyncType类型
```

### 4. 全局函数vs实例方法

**重要**：在`buffer.py`中，`_alloc_event_id()`和`_release_event_id()`是全局函数，不是实例方法。

```python
# 正确
self._p2c_event_id = _alloc_event_id()
_release_event_id(self._p2c_event_id)

# 错误
self._p2c_event_id = self._alloc_event_id()
self._release_event_id(self._p2c_event_id)
```

### 5. IR层操作注册

**重要**：所有IR层操作都需要在C++文件中注册，包括同步操作。

```cpp
// 正确
REGISTER_OP("system.is_aic")
    .set_description("Check if current core is AIC (returns bool)")
    .set_op_category("SyncOp")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);
```

### 6. TileView结构体更新

**重要**：当添加新字段到TileView时，需要同时更新构造函数。

```cpp
// 正确
struct TileView {
  // ... 现有字段
  int buffer_policy = -1;
  int sync_type = -1;

  // 更新构造函数
  TileView(..., int buffer_policy = -1, int sync_type = -1)
      : ...,
        buffer_policy(buffer_policy),
        sync_type(sync_type) {}
};
```

### 7. 用户API一致性

**重要**：用户使用`import pypto.language.manual as plm`的方式，所以所有buffer相关的类型和函数都需要在`__init__.py`中导出。

```python
# 正确
from .buffer_type import BufferPolicy, SyncType, HardEvent
from .buffer_info import BufferInfo
from .buffer_policy import SingleBufferPolicy, DoubleBufferPolicy, TripleBufferPolicy, QuadBufferPolicy
from .op.manual_ops import create_buffer

__all__ = [
    # Buffer types
    "BufferPolicy", "SyncType", "HardEvent",
    # Buffer info
    "BufferInfo",
    # Buffer policies
    "SingleBufferPolicy", "DoubleBufferPolicy", "TripleBufferPolicy", "QuadBufferPolicy",
    # Buffer functions
    "create_buffer",
    # ... 其他导出
]
```

---

## 实现注意事项

### 1. 函数名称一致性

**重要**：在PyPTO中，创建tile的函数是`make_tile()`，而不是`create_tile()`。

- ✅ 正确：`_ir_block_ops.make_tile()`
- ❌ 错误：`_ir_block_ops.create_tile()`

### 2. 参数名称一致性

**重要**：`make_tile()`函数的参数名称是`target_memory`，而不是`buffer_type`。

- `make_tile(shape, dtype, target_memory=...)` ✅
- `make_tile(shape, dtype, buffer_type=...)` ❌

### 3. 枚举值转换

**重要**：当传递枚举类型给IR层时，需要使用`.value`属性转换为int。

```python
# 正确
buffer_policy=buffer_policy.value,  # BufferPolicy -> int
sync_type=sync_type.value           # SyncType -> int

# 错误
buffer_policy=buffer_policy,  # BufferPolicy类型
sync_type=sync_type           # SyncType类型
```

### 4. 全局函数vs实例方法

**重要**：在`buffer.py`中，`_alloc_event_id()`和`_release_event_id()`是全局函数，不是实例方法。

```python
# 正确
self._p2c_event_id = _alloc_event_id()
_release_event_id(self._p2c_event_id)

# 错误
self._p2c_event_id = self._alloc_event_id()
self._release_event_id(self._p2c_event_id)
```

### 5. IR层操作注册

**重要**：所有IR层操作都需要在C++文件中注册，包括同步操作。

```cpp
// 正确
REGISTER_OP("system.is_aic")
    .set_description("Check if current core is AIC (returns bool)")
    .set_op_category("SyncOp")
    .no_argument()
    .f_deduce_type(DeduceUnknownType);
```

### 6. TileView结构体更新

**重要**：当添加新字段到TileView时，需要同时更新构造函数。

```cpp
// 正确
struct TileView {
  // ... 现有字段
  int buffer_policy = -1;
  int sync_type = -1;

  // 更新构造函数
  TileView(..., int buffer_policy = -1, int sync_type = -1)
      : ...,
        buffer_policy(buffer_policy),
        sync_type(sync_type) {}
};
```

### 7. 用户API一致性

**重要**：用户使用`import pypto.language.manual as plm`的方式，所以所有buffer相关的类型和函数都需要在`__init__.py`中导出。

```python
# 正确
from .buffer_type import BufferPolicy, SyncType, HardEvent
from .buffer_info import BufferInfo
from .buffer_policy import SingleBufferPolicy, DoubleBufferPolicy, TripleBufferPolicy, QuadBufferPolicy
from .op.manual_ops import create_buffer

__all__ = [
    # Buffer types
    "BufferPolicy", "SyncType", "HardEvent",
    # Buffer info
    "BufferInfo",
    # Buffer policies
    "SingleBufferPolicy", "DoubleBufferPolicy", "TripleBufferPolicy", "QuadBufferPolicy",
    # Buffer functions
    "create_buffer",
    # ... 其他导出
]
```

### 8. 代码生成层实现

**重要**：代码生成层需要将buffer_policy和sync_type转换为PTO字符串。

```cpp
// 正确
static std::string ConvertBufferPolicy(int buffer_policy) {
  switch (buffer_policy) {
    case 0: return "SINGLE";
    case 1: return "DOUBLE";
    case 2: return "TRIPLE";
    case 3: return "QUAD";
    default: return "SINGLE";
  }
}

static std::string ConvertSyncType(int sync_type) {
  switch (sync_type) {
    case 0: return "NO_SYNC";
    case 1: return "INNER_CORE_SYNC";
    case 2: return "CROSS_CORE_SYNC_FORWARD";
    case 3: return "CROSS_CORE_SYNC_BOTH";
    default: return "NO_SYNC";
  }
}
```

### 9. 测试用例实现

**重要**：测试用例需要覆盖所有Buffer管理功能。

```python
# 正确
class TestBufferPolicyRotation:
    """Test Buffer policy rotation."""

    def test_double_buffer_rotation(self):
        """Test DoubleBufferPolicy rotation."""
        policy = DoubleBufferPolicy(
            [64, 128], DataType.FP16, MemorySpace.Vec, SyncType.NO_SYNC
        )

        # First get() should return pong (flag1=0 -> flag1=1 -> return pong)
        buffer1 = policy.get()
        assert buffer1 == policy.pong

        # Second get() should return ping (flag1=1 -> flag1=0 -> return ping)
        buffer2 = policy.get()
        assert buffer2 == policy.ping
```

---

**文档版本**：5.0（训练算子版）
**最后更新**：2026-03-06
**作者**：PyPTO团队

---

## 10. PTO MLIR后端实现

### 10.1 为什么选择PTO MLIR后端？

**硬件无关性**：
- PTO MLIR是硬件无关的中间表示
- 可以支持多种后端（CCE、PTO等）
- 不需要绑定到特定的硬件API

**高层抽象**：
- PTO MLIR提供了更高层级的抽象
- 使用`pto.record_event`、`pto.wait_event`等操作
- 隐藏了硬件细节（如`set_flag`、`wait_flag`）

**更好的可移植性**：
- PTO MLIR代码可以在不同的硬件平台上运行
- 减少了对特定硬件的依赖
- 便于后续扩展到其他硬件平台

### 10.2 PTO MLIR同步操作

**核内同步**：
- `pto.record_event`：记录事件用于同步
- `pto.wait_event`：等待记录的事件

```mlir
pto.record_event [#pto.pipe_event_type<EVENT_LOAD_FROM_GM>, #pto.pipe_event_type<EVENT_COMPUTE_VEC>, #pto.event<EVENT_ID0>]
pto.wait_event [#pto.pipe_event_type<EVENT_LOAD_FROM_GM>, #pto.pipe_event_type<EVENT_COMPUTE_VEC>, #pto.event<EVENT_ID0>]
```

**跨核同步**：
- `pto.sync.set`：设置跨核同步标志
- `pto.sync.wait`：等待跨核同步标志

```mlir
pto.sync.set #pto.pipe<PIPE_M>, 0
pto.sync.wait #pto.pipe<PIPE_V>, 0
```

### 10.3 与CCE后端的对比

| 特性 | CCE后端 | PTO MLIR后端 |
|------|----------|---------------|
| 同步操作 | `set_flag`、`wait_flag` | `pto.record_event`、`pto.wait_event` |
| 跨核同步 | `cross_core_set_flag`、`cross_core_wait_flag` | `pto.sync.set`、`pto.sync.wait` |
|后端硬件依赖 | 强依赖CCE API | 硬件无关 |
| 可移植性 | 仅支持CCE硬件 | 支持多种硬件平台 |
| 抽象层级 | 底层硬件API | 高层中间表示 |

### 10.4 PTO MLIR后端的实现要点

**1. 移除CCE特定的同步调用**：
- 删除`buffer_info.py`文件（不再需要HardEvent和PipeType映射）
- 修改`buffer.pykt`，使用PTO MLIR同步操作

**2. 添加PTO MLIR同步操作**：
- 在`system_ops.py`中添加`record_event`、`wait_event`、`sync_set`、`sync_wait`函数
- 在`sync.cpp`中注册这些操作
- 在`backend_910b_pto_ops.cpp`中添加代码生成

**3. 更新Buffer类**：
- `wait(src_op, dst_op)`：使用`pto.wait_event`
- `set(src_op, dst_op)`：使用`pto.record_event`
- `set_cross_core()`：使用`pto.sync.set`
- `wait_cross_core()`：使用`pto.sync.wait`

### 10.5 PipeEventKind的使用

**PipeEventKind定义**（参考`ptoas_ir.md`）：
```python
EVENT_LOAD_FROM_GM = 0   # Load from GM
EVENT_STORE_FROM_ACC = 1  # Store from accumulator
EVENT_STORE_FROM_VEC = 2  # Store from vector/UB
EVENT_MOVE_MAT_TO_LEFT = 3  # Move: MAT -> LEFT
EVENT_MOVE_MAT_TO_SCALAR = 4  # Move: MAT -> scalar
EVENT_MOVE_MAT_TO_BIAS = 5  # Move: MAT -> BIAS
EVENT_MOVE_MAT_TO_VEC = 6  # Move: MAT -> VEC
EVENT_MOVE_VEC_TO_MAT = 7  # Move: VEC -> MAT
EVENT_COMPUTE_MATMUL = 8  # Matrix multiplication
EVENT_COMPUTE_VEC_VEC = 9  # Vector operation
EVENT_VEC_WAITPOINT = 10  # Vector wait event
```

**使用示例**：
```python
# 核内同步：从GM加载到Vec计算
buffer.set(src_op=EVENT_LOAD_FROM_GM, dst_op=EVENT_COMPUTE_VEC)
buffer.wait(src_op=EVENT_LOAD_FROM_GM, dst_op=EVENT_COMPUTE_VEC)
```

### 10.6 代码生成示例

**Python代码**：
```python
import pypto.language.manual as plm

# 创建buffer
buffer = plm.Buffer([64, 128], plm.DataType.FP16, plm.MemorySpace.Vec, plm.SyncType.INNER_CORE_SYNC)
buffer.init()

# 核内同步
buffer.set(src_op=0, dst_op=9)  # EVENT_LOAD_FROM_GM -> EVENT_COMPUTE_VEC
buffer.wait(src_op=0, dst_op=9)

buffer.uninit()
```

**生成的PTO MLIR代码**：
```mlir
pto.record_event [#pto.pipe_event_type<0>, #pto.pipe_event_type<9>, #pto.event<EVENT_ID0>]
pto.wait_event [#pto.pipe_event_type<0>, #pto.pipe_event_type<9>, #pto.event<EVENT_ID0>]
```

### 10.7 跨核同步示例

**Python代码**：
```python
import pypto.language.manual as plm

# 创建跨核同步buffer
buffer = plm.Buffer([64, 128], plm.DataType.FP16, plm.MemorySpace.Vec, plm.SyncType.CROSS_CORE_SYNC_BOTH)
buffer.init()

# 跨核同步
buffer.set_cross_core()
buffer.wait_cross_core()

buffer.uninit()
```

**生成的PTO MLIR代码**：
```mlir
pto.sync.set #pto.pipe<PIPE_FIX>, 0
pto.sync.set #pto.pipe<PIPE_FIX>, 16
pto.sync.wait #pto.pipe<PIPE_V>, 0
pto.sync.wait #pto.pipe<PIPE_V>, 16
```

---

**文档版本**：6.0（PTO MLIR后端）
**最后更新**：2026-03-06
**作者**：PyPTO团队
EOF`


---

## 10. PTO MLIR后端实现

### 10.1 为什么选择PTO MLIR后端？

**硬件无关性**：
- PTO MLIR是硬件无关的中间表示
- 可以支持多种后端（CCE、PTO等）
- 不需要绑定到特定的硬件API

**高层抽象**：
- PTO MLIR提供了更高更高级别的抽象
- 使用`pto.record_event`、`pto.wait_event`等操作
- 隐藏了硬件细节（如`set_flag`、`wait_flag`）

**更好的可移植性**：
- PTO MLIR代码可以在不同的硬件平台上运行
- 减少了对特定硬件的依赖
- 便于后续扩展到其他硬件平台

### 10.2 PTO MLIR同步操作

**核内同步**：
- `pto.record_event`：记录事件用于同步
- `pto.wait_event`：等待记录的事件

**跨核同步**：
- `pto.sync.set`：设置跨核同步标志
- `pto.sync.wait`：等待跨核同步标志

### 10.3 与CCE后端的对比

| 特性 | CCE后端 | PTO MLIR后端 |
|------|----------|---------------|
| 同步操作 | `set_flag`、`wait_flag` | `pto.record_event`、`pto.wait_event` |
| 跨核同步 | `cross_core_set_flag`、`cross_core_wait_flag` | `pto.sync.set`、`pto.sync.wait` |
| 硬件依赖 | 强依赖CCE API | 硬件无关 |
| 可移植性 | 仅支持CCE硬件 | 支持多种硬件平台 |
| 抽象层级 | 底层硬件API | 高层中间表示 |

### 10.4 PTO MLIR后端的实现要点

**1. 移除CCE特定的同步调用**：
- 删除`buffer_info.py`文件（不再需要HardEvent和PipeType映射）
- 修改`buffer.py`，使用PTO MLIR同步操作

**2. 添加PTO MLIR同步操作**：
- 在`system_ops.py`中添加`record_event`、`wait_event`、`sync_set`、`sync_wait`函数
- 在`sync.cpp`中注册这些操作
- 在`backend_910b_pto_ops.cpp`中添加代码生成

**3. 更新Buffer类**：
- `wait(src_op, dst_op)`：使用`pto.wait_event`
- `set(src_op, dst_op)`：使用`pto.record_event`
- `set_cross_core()`：使用`pto.sync.set`
- `wait_cross_core()`：使用`pto.sync.wait`

### 10.5 PipeEventKind的使用

**PipeEventKind定义**（参考`ptoas_ir.md`）：
- EVENT_LOAD_FROM_GM = 0   # Load from GM
- EVENT_STORE_FROM_ACC = 1  # Store from accumulator
- EVENT_STORE_FROM_VEC = 2  # Store from vector/UB
- EVENT_MOVE_MAT_TO_LEFT = 3  # Move: MAT -> LEFT
- EVENT_MOVE_MAT_TO_SCALAR = 4  # Move: MAT -> scalar
- EVENT_MOVE_MAT_TO_BIAS = 5  # Move: MAT -> BIAS
- EVENT_MOVE_MAT_TO_VEC = 6  # Move: MAT -> VEC
- EVENT_MOVE_VEC_TO_MAT = 7  # Move: VEC -> MAT
- EVENT_COMPUTE_MATMUL = 8  # Matrix multiplication
- EVENT_COMPUTE_VEC = 9  # Vector operation
- EVENT_VEC_WAITPOINT = 10  # Vector wait event

**使用示例**：
```python
# 核内同步：从GM加载到Vec计算
buffer.set(src_op=0, dst_op=9)  # EVENT_LOAD_FROM_GM -> EVENT_COMPUTE_VEC
buffer.wait(src_op=0, dst_op=9)
```

### 10.6 代码生成示例

**Python代码**：
```python
import pypto.language.manual as plm

# 创建buffer
buffer = plm.Buffer([64, 128], plm.DataType.FP16, plm.MemorySpace.Vec, plm.SyncType.INNER_CORE_SYNC)
buffer.init()

# 核内同步
buffer.set(src_op=0, dst_op=9)  # EVENT_LOAD_FROM_GM -> EVENT_COMPUTE_VEC
buffer.wait(src_op=0, dst_op=9)

buffer.uninit()
```

**生成的PTO MLIR代码**：
```mlir
pto.record_event [#pto.pipe_event_type<0>, #pto.pipe_event_type<9>, #pto.event<EVENT_ID0>]
pto.wait_event [#pto.pipe_event_type<0>, #pto.pipe_event_type<9>, #pto.event<EVENT_ID0>]
```

### 10.7 跨核同步示例

**Python代码**：
```python
import pypto.language.manual as plm

# 创建跨核同步buffer
buffer = plm.Buffer([64, 128], plm.DataType.FP16, plm.MemorySpace.Vec, plm.SyncType.CROSS_CORE_SYNC_BOTH)
buffer.init()

# 跨核同步
buffer.set_cross_core()
buffer.wait_cross_core()

buffer.uninit()
```

**生成的PTO MLIR代码**：
```mlir
pto.sync.set #pto.pipe<PIPE_FIX>, 0
pto.sync.set #pto.pipe<PIPE_FIX>, 16
pto.sync.wait #pto.pipe<PIPE_V>, 0
pto.sync.wait #pto.pipe<PIPE_V>, 16
```

---

**文档版本**：6.0（PTO MLIR后端）
**最后更新**：2026-03-06
**作者**：PyPTO团队
EOFMARKER`
