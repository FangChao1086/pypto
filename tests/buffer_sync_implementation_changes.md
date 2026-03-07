# Buffer和同步功能实现修改记录（PTO MLIR后端）

## 修改概述

本文档记录了根据`pypto_buffer_design_beginner_guide_training.md`文档实现Buffer和同步功能的所有修改，专门针对**PTO MLIR后端**。

---

## 已完成的修改

### 修改1：创建buffer_type.py文件

**文件路径**：`pypto/python/pypto/language/manual/buffer_type.py`

**修改原因**：
- 定义Buffer管理所需的核心类型枚举
- 提供类型安全的API
- 与PTO MLIR的设计保持一致

**修改作用**：
- `BufferPolicy`：定义buffer策略（单缓冲、双缓冲、三缓冲、四缓冲）
- `SyncType`：定义同步类型（不同步、核内同步、跨核同步）

---

### 修改2：创建buffer.py文件

**文件路径**：`pypto/python/pypto/language/manual/buffer.py`

**修改原因**：
- 实现Buffer类的同步管理功能
- 支持核内同步和跨核同步
- 使用PTO MLIR同步操作（而非CCE特定的API）

**修改作用**：
- `init()`：初始化事件ID，设置初始标志
- `uninit()`：释放事件ID
- `wait(src_op, dst_op)`：等待指定事件（使用`pto.wait_event`）
- `set(src_op, dst_op)`：设置指定事件（使用`pto.record_event`）
- `set_cross_core()`：设置跨核同步（使用`pto.sync.set`）
- `wait_cross_core()`()`：等待跨核同步（使用`pto.sync.wait`）

**注意**：
- 同步相关的方法（init、uninit、wait、set、set_cross_core、wait_cross_core等）已实现
- `wait()` 和 `set()` 方法调用 `pto.wait_event` 和 `pto.record_event` 实现核内同步
  - `wait(event_type)` 和 `set(event_type)` 方法接受 `event_type` 参数（HardEvent枚举值）
  - 根据 `event_type` 自动选择使用 `_p2c_event_id` 或 `_c2p_event_id`
  - L0_P2C (2) 和 ACC_P2C (4) 使用 `_p2c_event_id`
  - L0_C2P (3) 和 ACC_C2P (5) 使用 `_c2p_event_id`
- `set_cross_core()` 和 `wait_cross_core()` 方法调用 `pto.sync.set` 和 `pto.sync.wait` 实现跨核同步
  - 跨核同步的详细实现（区分UB/GM和L1）已根据ops-transformer完整实现
  - 跨核同步使用 `pto.sync.set` 和 `pto.sync.wait` 进行同步
  - 一个AIC对应两个AIV（AIV0和AIV1），使用event_id和event_id+16进行同步
  - 不再使用 `is_aic()` 和 `is_aiv()` 函数，因为在PTO MLIR中这些操作是全局的

---

### 修改3：创建buffer_policy.py文件

**文件路径**：`pypto/python/pypto/language/manual/buffer_policy.py`

**修改原因**：
- 实现不同buffer策略的管理
- 支持单缓冲、双缓冲、三缓冲、四缓冲策略
- 提供统一的API接口

**修改作用**：
- `SingleBufferPolicy`：单buffer策略（无乒乓）
- `DoubleBufferPolicy`：双buffer策略（ping-pong轮转）
- `TripleBufferPolicy`：三buffer策略（轮转）
- `QuadBufferPolicy`：四buffer策略（队列模式）

---

### 修改4：修改block_ops.py文件

**文件路径**：`pypto/python/pypto/ir/op/block_ops.py`

**修改原因**：
- 在`make_tile`函数中添加`buffer_policy`和`sync_type`参数
- 支持buffer管理的功能

**修改作用**：
- `make_tile`函数现在接受`buffer_policy`和`sync_type`参数
- 这些参数被传递到IR层，用于生成相应的PTO MLIR代码

---

### 修改5：修改manual_ops.py文件

**文件路径**：`pypto/python/pypto/language/manual/op/manual_ops.py`

**修改原因**：
- 添加`create_buffer`函数
- 提供简单的buffer创建API

**修改作用**：
- `create_buffer`函数：创建计算结果的buffer
- 支持buffer策略和同步类型

---

### 修改6：修改__init__.py文件

**文件路径**：`pypto/python/pypto/language/manual/__init__.py`

**修改原因**：
- 导出buffer相关的类型和函数
- 方便用户使用`import pypto.language.manual as plm`

**修改作用**：
- 导出`BufferPolicy`、`SyncType`类型
- 导出`SingleBufferPolicy`、`DoubleBufferPolicy`、`TripleBufferPolicy`、`QuadBufferPolicy`类
- 导出`create_buffer`函数

---

## 已实现的后续工作

### 1. IR层实现

#### 1.1 添加PTO MLIR同步操作到IR层

**文件**：`src/ir/op/sync_ops/sync.cpp`

**修改内容**：
- 添加`pto.record_event`操作：记录事件用于同步
- 添加`pto.wait_event`操作：等待记录的事件
- 添加`pto.sync.set`操作：设置跨核同步标志
- 添加`pto.sync.wait`操作：等待跨核同步标志

#### 1.2 添加buffer_policy和sync_type到TileView

**文件**：`include/pypto/ir/type.h`

**修改内容**：
- 在`TileView`结构体中添加`buffer_policy`字段
- 在`TileView`结构体中添加`sync_type`字段
- 更新构造函数以支持这两个新字段

#### 1.3 添加buffer_policy和sync_type参数到make_tile操作

**文件**：`src/ir/op/block_ops/memory.cpp`

**修改内容**：
- 在`DeduceBlockCreateTileType`函数中提取`buffer_policy`和`sync_type`参数
- 将这些参数设置到`TileView`中
- 在操作注册中添加这两个属性

### 2. Python层实现

#### 2.1 添加PTO MLIR同步操作到Python层

**文件**：`python/pypto/ir/op/system_ops.py`

**修改内容**：
- 添加`record_event()`函数：记录事件
- 添加`wait_event()`函数：等待事件
- 添加`sync_set()`函数：设置跨核同步
- 添加`sync_wait()`函数：等待跨核同步

### 3. 代码生成层实现

#### 3.1 添加PTO MLIR同步操作的代码生成

**文件**：`src/backend/910B_PTO/backend_910b_pto_ops.cpp`

**修改内容**：
- 添加`pto.record_event`操作的代码生成
- 添加`pto.wait_event`操作的代码生成
- 添加`pto.sync.set`操作的代码生成
- 添加`pto.sync.wait`操作的代码生成

---

## 修改和新增文件总结

### 新增文件（3个）
1. `python/pypto/language/manual/buffer_type.py` - Buffer类型定义
2. `python/pypto/language/manual/buffer_policy.py` - Buffer策略类
3. `python/pypto/language/manual)buffer.py` - Buffer同步管理类

### 修改文件（10个）
1. `python/pypto/ir/op/block_ops.py` - 添加buffer_policy和sync_type参数到make_tile函数
2. `python/pypto/ir/op/system_ops.py` - 添加PTO MLIR同步相关函数
3. `python/pypto/language/manual/op/manual_ops.py` - 添加create_buffer函数
4. `python/pypto/py)to_core/codegen.pyi` - 更新类型定义
5. `python/pypto/py)to_core/ir.pyi` - 更新类型定义
6. `python/pypto/language/manual/__init__.py` - 添加buffer相关导入和导出
7. `src/ir/op/sync_ops/sync.cpp` - 添加PTO MLIR同步操作到IR层
8. `src/ir/op/block_ops/memory.cpp` - 添加buffer_policy和sync_type参数
9. `include/pypto/ir/type.h` - 添加buffer_policy和sync_type到TileView
10. `src/backend/910B_PTO/backend_910b_pto_ops.cpp` - 添加PTO MLIR同步操作的代码生成

### 文档文件（2个）
1. `tests/buffer_sync_implementation_changes.md` - 实现修改记录
2. `tests/pypto_buffer_design_beginner_guide_training.md` - 设计指南

---

**文档版本**：5.0 (PTO MLIR后端)
**最后更新**：2026-03-06
**作者**：PyPTO团队
