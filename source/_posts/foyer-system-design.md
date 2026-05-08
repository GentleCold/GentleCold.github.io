---
title: Foyer系统设计分析
category: [笔记]
date: 2026-05-08 12:04
tags: [Cache, Rust, Storage]
---

资料：Foyer: A Hybrid Cache in Rust - Past, Present and Future

链接：https://blog.mrcroxx.com/posts/foyer-a-hybrid-cache-in-rust-past-present-and-future/

代码：https://github.com/foyer-rs/foyer

说明：Foyer不是传统会议论文，而是一篇系统设计文章和开源项目。这里按系统设计分析的方式梳理它的设计动机、核心机制和对缓存系统的启发。

## 1. 背景

Foyer讨论的是通用系统里的hybrid cache，也就是把内存缓存和磁盘缓存组合成一个统一缓存层。

传统缓存常见有两类：

1. 纯内存缓存：延迟低、实现简单，但容量受DRAM限制，成本高。
2. 纯磁盘缓存：容量大、成本低，但访问延迟和I/O调度复杂度明显更高。

实际系统往往需要二者结合。例如数据库、对象存储、RAG文档缓存、LLM KV cache offload系统里，热点数据应该留在内存，温数据可以落到SSD，冷数据最终回源或重算。

Foyer的目标不是做某个业务系统的专用缓存，而是提供一个Rust里的通用hybrid cache框架：上层使用一个cache API，下层同时管理memory tier和disk tier。

一句话概括：**Foyer试图把“内存快、磁盘大”的两级缓存做成可复用的Rust系统组件。**

## 2. 问题

Hybrid cache看起来只是多加一层SSD，但实际工程问题比单层LRU复杂很多。

### 2.1 数据应该进哪一层

如果所有新对象都先进入内存，很容易发生cache pollution：一次性扫描、低复用大对象会挤掉真正的热点。

如果所有对象都写入磁盘，又会放大写入量，让SSD带宽和寿命成为瓶颈。

因此缓存系统需要回答两个问题：

1. admission：一个对象值不值得进入缓存。
2. promotion：从磁盘读到的对象是否应该提升到内存。

这和简单LRU不同。LRU只问“谁最久没用”，hybrid cache还要问“这个对象值得占用哪一级介质”。

### 2.2 并发miss会造成放大

在高并发服务里，多个请求可能同时访问同一个不存在的key。如果每个请求都独立回源，就会出现cache stampede。

更合理的行为是：第一个请求负责fetch，后续同key请求等待它的结果。这样可以把N次回源压成一次。

### 2.3 磁盘层不是HashMap

内存层可以简单地维护hash table加LRU链表，但磁盘层需要处理：

1. 文件布局和空间回收。
2. 异步I/O。
3. 写入合并和读放大。
4. 崩溃恢复或元数据一致性。
5. 大小不等value导致的碎片。

因此hybrid cache不是给内存LRU加一个`std::fs::write`就够了，它需要单独设计磁盘对象布局和I/O管线。

## 3. 核心设计

Foyer把缓存分成两个主要层次：

1. memory cache：低延迟、保存最热对象。
2. storage cache：基于磁盘或SSD，保存更大容量的温对象。

对上层来说，Foyer提供类似普通cache的接口；对内部来说，它要在两层之间做准入、驱逐、提升和后台I/O。

### 3.1 HybridCache抽象

Hybrid cache的关键抽象是让调用者不直接关心对象现在在哪一层。

访问流程可以理解为：

```text
get(key)
-> 查memory cache
-> miss则查storage cache
-> storage hit则读取并可能promote到memory
-> storage miss则回源fetch
-> fetch结果按策略写入缓存
```

这样上层业务只看到一次`get`，而不是手动写“先查内存、再查磁盘、再回源”的重复逻辑。

这类抽象的价值在于，它把缓存一致性、并发miss合并、I/O调度和层间迁移集中到框架内部。

### 3.2 Fetch去重

Foyer的一个重要能力是把cache miss和fetch函数绑定起来。

当多个任务同时请求同一个key时，系统可以让它们共享同一个in-flight fetch。第一个请求触发真实加载，后续请求等待同一个结果。

这个设计解决的是缓存系统里很常见的thundering herd问题。它对后端数据库、对象存储、远端KV服务、LLM KV block重算都很重要：miss本身不可怕，怕的是同一个miss被并发放大。

### 3.3 Admission和Eviction

Foyer支持不同的缓存策略。核心思想是把“是否接纳”和“驱逐谁”从固定LRU里抽象出来。

常见策略包括：

1. LRU：实现简单，适合时间局部性明显的负载。
2. LFU或TinyLFU类准入：利用访问频率估计，避免一次性数据污染缓存。
3. S3-FIFO一类FIFO-family策略：用更低元数据成本接近较好的命中率。

对hybrid cache来说，admission比单层缓存更重要。因为每次错误接纳不仅浪费内存，也可能造成SSD写入放大。

### 3.4 磁盘层I/O

Foyer的storage cache需要把对象序列化到磁盘，并维护key到磁盘位置的索引。

这里有几个典型取舍：

1. 写路径要尽量批量化，避免小对象随机写拖垮SSD。
2. 读路径要支持异步I/O，不能让业务线程长时间阻塞。
3. 空间回收要和驱逐策略配合，防止磁盘文件无限膨胀。
4. 元数据需要足够轻量，否则索引本身会成为内存瓶颈。

这也是Foyer和普通内存cache crate最大的差异：它不只是一个替换策略库，而是一个完整的缓存存储系统。

## 4. 和CacheLib的关系

Foyer文章明确提到，业界已有Facebook/Meta的CacheLib这类成熟hybrid cache系统。

CacheLib的价值在于证明了通用缓存库可以服务多种大规模在线系统：同一套缓存内核可以被数据库、图存储、对象服务等复用。

Foyer的不同点在于它选择Rust生态：

1. 用Rust所有权和类型系统约束内存安全。
2. 和Tokio、async ecosystem结合。
3. 给Rust服务提供一个不用绑定C++库的hybrid cache选择。

所以Foyer不是在理论上发明hybrid cache，而是在Rust系统生态里重建这类能力。

## 5. 对KV Cache系统的启发

Foyer的设计对LLM KV cache offload也有直接启发。

KV cache系统通常也有多级存储：

1. GPU HBM：最快，但容量最贵。
2. CPU pinned memory：容量更大，适合跨请求复用和RDMA传输。
3. SSD：更便宜，适合保存温KV block。
4. 远端节点内存：在分布式场景里可以作为另一级共享缓存。

这和Foyer的hybrid cache模型非常接近，只是KV cache的value更大、更结构化，并且读写路径受GPU DMA、RDMA和attention调度约束。

### 5.1 准入比驱逐更关键

在KV cache复用场景里，错误写入代价很高。一个长prompt可能产生大量KV block，如果它未来不会复用，把它写进CPU或SSD缓存就是纯开销。

因此系统不能只在满了之后驱逐，还应该在写入前做admission判断。例如：

1. 短prompt或低复用概率请求直接bypass。
2. 对重复前缀、高频系统prompt、RAG模板保守接纳。
3. 用TinyLFU或滑动窗口频率估计过滤一次性block。

这和Foyer强调的admission思路一致。

### 5.2 Fetch去重对应KV重算去重

KV cache miss后，系统可能需要从SSD读、从远端RDMA拉取，或者重新prefill计算。

如果多个请求同时miss同一批block，应该合并这些in-flight操作。否则会出现：

1. 多次SSD读同一个block。
2. 多次RDMA拉同一个block。
3. 多个prefill worker重复计算同一段prefix。

Foyer的fetch去重抽象可以直接映射到KV block层：同一个block hash只允许一个加载任务，其它请求等待结果。

### 5.3 磁盘层要按block设计

KV block通常是大对象，且有固定或半固定大小。相比普通key-value缓存，它更适合用ring buffer或分段文件做顺序写。

这和Foyer storage cache里的核心问题相同：磁盘层不能只看key-value语义，还要关心物理布局、I/O对齐、批量写入和回收策略。

## 6. 局限

Foyer作为通用cache库，也有通用抽象带来的边界。

首先，业务语义有限。缓存库通常不知道某个value是否真的会被复用，只能根据访问历史做统计推断。对于强语义场景，例如LLM prefix cache，业务侧可能能提供更强的admission hint。

其次，hybrid cache的性能高度依赖负载。如果访问是纯随机、几乎没有复用，那么再复杂的策略也只能增加开销。Foyer适合有明显热点或温数据复用的系统。

第三，磁盘层收益取决于SSD能力和对象大小。小对象随机读写、过高写放大、低队列深度都会削弱hybrid cache效果。

最后，通用库很难覆盖GPU/RDMA这类专用数据路径。对KV cache系统来说，Foyer的思想有价值，但实现上仍需要面向pinned memory、GPU地址、RDMA注册和block layout做专门优化。

## 7. 总结

Foyer的贡献可以理解为：在Rust里提供一个面向生产系统的hybrid cache抽象，而不是只提供一个内存LRU。

它的核心点是：

1. 把memory cache和storage cache封装成统一访问接口。
2. 用admission策略减少cache pollution和SSD写放大。
3. 用fetch去重避免并发miss放大。
4. 把磁盘层作为缓存系统的一等组件，处理异步I/O、布局和回收。
5. 在Rust生态中补齐类似CacheLib的通用缓存基础设施。

一句话概括：**Foyer的重点不是某个新的替换算法，而是把内存、SSD、准入、驱逐和并发miss合并组织成一个可复用的Rust hybrid cache系统。**
