---
title: 分布式系统与AI基础设施笔记
category: [笔记]
date: 2026-04-13 00:00
tags: [分布式系统, Ray, gRPC, 微服务, RAG, 消息队列]
---

## 1. Ray 与 Ray Data

### 1.1 Ray

Ray 是一个**分布式计算框架**，专为 Python 设计，核心目标是让单机代码轻松扩展到集群。

**核心抽象：**

- **Task**：无状态函数，`@ray.remote` 装饰后异步并行执行
- **Actor**：有状态对象，分布式进程，维护内部状态
- **Object Store**：共享内存对象存储，跨进程/节点零拷贝传输

```python
@ray.remote
def process(x):
    return x * 2

futures = [process.remote(i) for i in range(100)]  # 并行
results = ray.get(futures)
```

**典型用途：**

- 超参数搜索（Ray Tune）
- 强化学习（RLlib）
- 模型服务（Ray Serve）
- 批量推理 / 数据处理（Ray Data）

### 1.2 Ray Data

Ray Data 是 Ray 生态的**分布式数据处理库**，定位类似 Spark 但更适合 ML 工作流，尤其是**大规模批量推理**。

| 特性 | 说明 |
|------|------|
| 流式处理 | 数据分块（Dataset）流水线执行，不需全量加载进内存 |
| GPU 友好 | 原生支持将数据批次送入 GPU，配合 vLLM/PyTorch 使用 |
| 异构资源 | CPU 预处理 + GPU 推理可自动形成流水线 |
| 惰性求值 | 操作链式声明，`.materialize()` 时才执行 |

```python
import ray
ds = ray.data.read_csv("/data/imdb.csv")

ds = ds.map_batches(
    MyModel,
    concurrency=4,       # 4 个 Actor 并行
    num_gpus=1,          # 每个 Actor 分配 1 GPU
    batch_size=32,
)
ds.write_parquet("/output/")
```

## 2. Ray vs Spark vs Kubernetes

### 2.1 Ray Data vs Spark

| | Spark | Ray / Ray Data |
|--|-------|----------------|
| 起源 | 大数据 ETL、SQL 分析 | ML 训练、推理、强化学习 |
| 内存模型 | JVM 堆内存，跨节点需序列化 | 共享内存，同机零拷贝 |
| GPU 支持 | 需 Rapids 插件 | 原生 GPU 调度 |
| 流水线 | Stage 间有 shuffle barrier | 真流式流水线 |
| 有状态计算 | 无状态为主 | Actor 原生有状态，模型常驻内存 |
| 语言 | Scala 原生，Python 是二等公民 | Python 原生 |

> **Spark** 擅长 PB 级结构化数据的 SQL/ETL；**Ray** 擅长 ML 推理、训练这类需要 GPU、有状态、Python 原生的工作负载。

### 2.2 Ray vs Kubernetes

```
┌─────────────────────────────────┐
│   Ray / Ray Data / Ray Serve    │  ← 应用层：任务调度、ML 工作流
├─────────────────────────────────┤
│         Kubernetes              │  ← 基础设施层：容器编排、资源管理
├─────────────────────────────────┤
│       物理机 / 云主机            │
└─────────────────────────────────┘
```

| | Kubernetes | Ray |
|--|-----------|-----|
| 调度粒度 | Pod（容器） | Task / Actor（函数/对象） |
| 调度延迟 | 秒级（容器启动） | 毫秒级（进程内调度） |
| GPU 感知 | 粗粒度（整卡分配） | 细粒度（分数 GPU、共享） |
| 编程模型 | YAML 配置 | Python 代码 |
| 数据传递 | 需通过网络/存储 | 共享内存直接传递 Tensor |

生产环境通常配合使用：**KubeRay** 在 K8s 上部署 Ray 集群，各司其职。

## 3. gRPC 与序列化

### 3.1 序列化 / 反序列化

**本质：** 内存中的对象 ↔ 可传输/存储的字节流

| 格式 | 特点 | 典型场景 |
|------|------|---------|
| JSON | 可读、跨语言、慢、体积大 | REST API |
| Protobuf | 二进制、快、体积小、需 schema | gRPC |
| MessagePack | 二进制 JSON、无需 schema | 游戏、缓存 |
| Pickle | Python 专用、不安全、快 | Python 进程间 |
| Arrow | 列存、零拷贝、ML 友好 | Ray、Spark |

### 3.2 gRPC

**gRPC = HTTP/2 + Protobuf + 代码生成**

```protobuf
// service.proto
service Inference {
    rpc Predict (Request) returns (Response);
}
message Request  { string text = 1; }
message Response { float score = 1; }
```

| | REST + JSON | gRPC + Protobuf |
|--|------------|-----------------|
| 协议 | HTTP/1.1 | HTTP/2 |
| 序列化 | JSON（文本） | Protobuf（二进制） |
| 速度 | 慢 | 快 5-10x |
| 流式 | 不支持 | 双向流 |
| 跨语言 | 手写客户端 | 自动代码生成 |

### 3.3 进程间通信（IPC）

**同机通信速度排序：**

```
共享内存  >  管道/FIFO  >  Unix Socket  >  TCP Socket  >  文件
```

| 方式 | 特点 |
|------|------|
| 共享内存 | 最快，零拷贝，需同步锁 |
| 管道 pipe | 简单，父子进程间 |
| Unix Domain Socket | 比 TCP 快，无网络栈开销 |
| TCP Socket | 可跨机器，有开销 |
| 信号 Signal | 只能传信号编号，不传数据 |

**跨语言 IPC 方案对比：**

| 方案 | schema | 性能 | 典型用途 |
|------|--------|------|---------|
| gRPC/Protobuf | 必须 | 高 | 微服务 |
| Thrift | 必须 | 高 | Meta 内部服务 |
| JSON-RPC | 无需 | 低 | 简单场景 |
| Arrow Flight | 无需（Arrow格式） | 极高 | ML数据传输 |

Ray 的 Object Store 基于共享内存，Python 和 C++ 扩展共享同一块内存中的 Tensor，零拷贝。

## 4. 微服务

### 4.1 对比单体架构

```
单体架构                    微服务架构
┌─────────────────┐         ┌──────┐ ┌──────┐ ┌──────┐
│  用户模块        │         │ 用户  │ │ 订单  │ │ 支付  │
│  订单模块        │   →     │ 服务  │ │ 服务  │ │ 服务  │
│  支付模块        │         └──────┘ └──────┘ └──────┘
└─────────────────┘              各自独立进程/容器
  一个进程，共享内存
```

| 特性 | 说明 |
|------|------|
| 独立部署 | 每个服务单独发布，不影响其他服务 |
| 独立扩容 | 推荐服务压力大 → 单独加机器 |
| 技术异构 | 不同服务可用不同语言 |
| 故障隔离 | 一个服务挂了不影响其他 |

### 4.2 代价

- **复杂度高**：网络调用代替函数调用，有延迟、有失败
- **分布式事务难**：下单 + 扣库存 + 扣余额要保证一致性
- **运维成本高**：需要服务发现、链路追踪、API 网关

## 5. RAG（检索增强生成）

### 5.1 解决的问题

LLM 的知识是训练时冻结的，无法知道公司内部文档、最新信息、私有数据库。

### 5.2 工作流程

```
离线阶段（建库）:
文档 → 切块(chunk) → Embedding模型 → 向量 → 向量数据库

在线阶段（查询）:
问题 → Embedding → 向量相似度搜索 → Top-K片段 → LLM生成答案
```

### 5.3 关键技术选型

| 组件 | 选项 |
|------|------|
| Embedding 模型 | BGE、text-embedding-ada、E5 |
| 向量数据库 | Faiss（本地）、Milvus、Pinecone、Weaviate |
| 重排序（可选） | Cross-encoder reranker 提升精度 |
| LLM | GPT-4、Claude、本地 LLaMA |

## 6. 消息队列（Message Queue）

### 6.1 核心作用：解耦 + 异步 + 削峰

```
没有MQ（同步）:                   有MQ（异步）:
用户下单 → 发邮件(2s)              用户下单 → 写队列(5ms) → 立即返回
         → 发短信(1s)  = 3s延迟           ↓
         → 更新报表(1s)            消费者异步处理邮件/短信/报表
```

### 6.2 三大核心场景

**削峰填谷：**
```
秒杀活动：10万请求/秒 → [Queue] → 数据库承受的1000请求/秒
```

**服务解耦：**
```
订单服务 → [Queue: order.created] → 库存服务 / 积分服务 / 物流服务
各服务互不知道对方存在
```

**可靠传递：** 网络抖动时消息不丢失，消费失败可重试，支持持久化。

### 6.3 主流产品对比

| | Kafka | RabbitMQ | Redis Stream |
|--|-------|----------|-------------|
| 吞吐量 | 极高（百万/s） | 中（万/s） | 高 |
| 持久化 | 磁盘，可回放 | 内存为主 | 内存/RDB |
| 消费模式 | 拉取，消费组 | 推送 | 拉取 |
| 典型场景 | 日志、流计算 | 业务解耦 | 轻量任务队列 |

## 7. 三者在 AI 系统中的协作

```
用户请求
    │
    ▼
[API Gateway] ← 微服务入口
    │
    ├→ [RAG服务]  检索相关文档 → 向量DB查询
    │
    ├→ [消息队列] 异步任务
    │      ├→ 推理结果写入日志
    │      └→ 触发缓存预热
    │
    └→ [vLLM推理服务]  实际生成
```
