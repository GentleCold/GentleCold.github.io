---
title: Foyer技术要点分析
category: [笔记]
date: 2026-05-08 12:04
tags: [Cache, Rust, Storage]
---

资料：

- Foyer: A Hybrid Cache in Rust - Past, Present and Future
- Foyer docs.rs API
- Foyer GitHub README

链接：

- https://blog.mrcroxx.com/posts/foyer-a-hybrid-cache-in-rust-past-present-and-future/
- https://docs.rs/foyer/latest/foyer/
- https://github.com/foyer-rs/foyer

Foyer不是论文，而是一个Rust hybrid cache项目。这里不按“论文调研”写，而是拆它用了哪些技术点、每个点解决什么问题、工程上为什么要这么设计。

## 1. Hybrid Cache抽象

Foyer的核心抽象是`HybridCache`：对外表现得像一个普通cache，对内同时管理内存层和磁盘层。

普通内存cache的路径通常是：

```text
get(key) -> memory hit/miss
```

Foyer的路径更接近：

```text
get(key)
-> 查内存cache
-> 内存miss后查磁盘cache
-> 磁盘hit则异步读取value，并可能插回内存
-> 磁盘miss则返回miss，或者走fetch闭包
```

这个抽象的关键不是“多查一次磁盘”，而是把层级关系隐藏起来。调用者不需要手写两套cache、不需要自己处理磁盘I/O、不需要自己判断什么时候promote，也不需要在每个业务点重复处理并发miss。

这类设计适合value比较大、复用概率存在但DRAM放不下全部数据的系统。例如RisingWave这类状态存储系统、对象缓存、Embedding/RAG缓存、LLM KV block缓存。

## 2. 内存层：fast path和策略插件

Foyer的内存层负责最低延迟的fast path。它不能太复杂，否则所有请求都会被元数据开销拖慢。

### 2.1 Sharding

高并发cache如果只有一个全局锁，热点不是数据本身，而是元数据锁。Foyer这类系统通常会把key空间切成多个shard，每个shard维护自己的索引和策略状态。

sharding的好处是：

1. 降低锁竞争。
2. 让访问、插入、驱逐尽量局部化。
3. 方便并行执行策略维护。

代价是全局最优驱逐会变难。一个shard满了，不代表整个cache都满；某个shard的victim也不一定是全局最冷对象。所以sharded cache常见取舍是牺牲一点全局最优，换并发吞吐。

### 2.2 Eviction不是固定LRU

Foyer把eviction策略做成可替换组件，而不是写死LRU。原因是LRU在很多真实负载下并不好：

1. 顺序扫描会污染cache。
2. 一次性大批量访问会把热点挤出去。
3. LRU只看recency，不看frequency。

Foyer关注的策略包括FIFO-family、S3-FIFO、LRU、LFU/TinyLFU一类思路。

S3-FIFO这类策略的核心优势是元数据更轻。它不需要每次hit都把节点移动到链表头，减少了高并发下的写元数据压力。对于cache系统，这一点很重要：命中路径如果还要频繁修改共享结构，hit也会变贵。

### 2.3 Admission和Eviction分离

很多人理解cache时只关注“满了驱逐谁”。Foyer更重要的点是把admission也放进设计里。

Eviction回答：

```text
cache满了，谁应该被踢出去？
```

Admission回答：

```text
这个新对象值不值得进cache？
```

在hybrid cache里，admission尤其关键。因为错误接纳不只会污染DRAM，还可能造成SSD写放大。一个只访问一次的大value，如果被写入磁盘层，后续没有命中，那这次写就是纯损耗。

TinyLFU类准入策略的典型做法是用近似频率结构估计key热度，只有候选对象比潜在victim更值得缓存时才接纳。这里常见数据结构是Count-Min Sketch，优点是内存开销可控，缺点是只能近似计数，需要周期性aging避免历史热点永久占优。

## 3. 磁盘层：不是把HashMap落盘

Foyer最有价值的部分在磁盘层。磁盘cache和内存cache的差异非常大，不能简单把value序列化后写文件。

磁盘层至少要解决：

1. 文件空间怎么分配。
2. 对象大小不等怎么处理。
3. 写入如何批量化。
4. 读取如何异步化。
5. 元数据索引怎么维护。
6. 崩溃后如何恢复。
7. 如何减少文件系统和页缓存干扰。

Foyer把磁盘层称为storage engine，并提供多种engine配置。这一点类似数据库：同一个cache API下面，底层存储引擎可以针对value大小和I/O模式做不同实现。

## 4. Small / Large / Mixed引擎

Foyer文档里可以看到不同磁盘引擎思路，例如Small、Large、Mixed以及BlockEngine相关配置。

它们背后的问题是：小对象和大对象不能用同一种布局高效处理。

### 4.1 Small对象

小对象的问题是元数据和I/O放大。

如果每个小value都单独写一次磁盘，就会产生大量小随机写。SSD虽然随机读写比HDD强，但高QPS小I/O仍然会浪费带宽和CPU。

因此小对象更适合被打包：

```text
多个小entry -> 聚合成segment/block -> 顺序写入磁盘
```

这样可以把多次小写合并成一次较大的顺序写。代价是读取单个对象时可能要读出更大的block，再从block里解析目标entry。

### 4.2 Large对象

大对象的问题相反：如果强行打包，读写放大很明显。

大value通常适合独立分配空间，或者按大block切分。这样读一个大对象时不会被很多无关小对象拖累，也更容易控制磁盘空间回收。

### 4.3 Mixed引擎

真实负载通常既有小对象也有大对象，所以Mixed引擎的意义是按value大小分流。

一种典型结构是：

```text
value size <= threshold -> small engine
value size > threshold  -> large engine
```

这个阈值不是纯理论参数，而是和设备、对象分布、压缩率、访问模式有关。阈值太小，大量中等对象走large路径，元数据和空间碎片可能变多；阈值太大，中等对象被打包后读放大明显。

## 5. BlockEngine和块化存储

BlockEngine的核心是把磁盘cache组织成块或segment。

块化的好处：

1. 顺序写友好。
2. 易于批量刷盘。
3. 元数据可以按block管理，减少每个entry的独立I/O成本。
4. 回收时可以按block粒度判断有效数据比例。

但块化也引入问题：一个block里可能只有少量entry仍然有效，其它entry已经被覆盖或删除。此时如果保留整个block，会浪费空间；如果马上重写有效entry，又会增加写放大。

这就是为什么Foyer会有reinsertion、eviction picker一类机制。

## 6. Reinsertion：缓存里的轻量GC

Reinsertion可以理解为磁盘cache里的轻量级垃圾回收。

当一个block或segment要被回收时，里面可能还有一些仍然值得保留的entry。系统可以选择把这些entry重新插入新位置，而不是直接丢掉。

它解决的问题是：

1. 防止热点entry因为所在block被整体回收而误删。
2. 提高磁盘层命中率。
3. 在空间回收和数据保留之间做折中。

但reinsertion不是免费午餐。重新插入会带来额外写入，所以它必须配合策略判断：只有仍然有价值的entry才值得搬迁。否则系统会变成不停地把冷数据从一个block搬到另一个block。

这和LSM-tree compaction有一点相似：都在做空间回收和有效数据搬迁；区别是cache系统可以更激进地丢数据，因为cache不是唯一真相来源。

## 7. Eviction Picker

磁盘cache满了以后，不能只随机删文件。Foyer的eviction picker负责挑选要回收的对象、block或segment。

一个好的picker需要综合几类信号：

1. recency：最近是否访问过。
2. frequency：历史访问频率。
3. size：对象多大，回收后能释放多少空间。
4. block有效率：回收一个block会浪费多少仍有效数据。
5. 写放大：保留有效entry需要额外搬迁多少数据。

内存cache驱逐一个entry释放的是DRAM；磁盘cache驱逐一个block释放的是磁盘空间，但可能牵连同block里的多个entry。因此磁盘层eviction比内存LRU更接近存储系统里的空间管理。

## 8. Direct I/O和文件系统开销

Foyer文档和文章都强调过文件系统层开销。Hybrid cache通常不希望操作系统页缓存再缓存一遍数据，因为这会出现双重缓存：

```text
应用内存cache
-> OS page cache
-> SSD
```

这会带来两个问题：

1. 应用以为自己只用了固定DRAM，实际OS page cache又吃了一份内存。
2. cache替换策略被拆成两套：应用层一套，内核page cache一套，二者互相不知道。

Direct I/O的思路是绕过page cache，让应用自己管理缓存。代价是I/O必须满足对齐要求，例如buffer地址、offset、length要按设备块大小对齐。

所以Foyer这类系统需要自己管理I/O buffer，并处理alignment、padding、block size等细节。

## 9. 异步I/O和Runtime

Foyer面向Rust async生态，磁盘读写不能阻塞业务任务。

典型路径是：

1. get请求在async任务里发起。
2. 内存miss后向storage engine提交异步读。
3. I/O完成后反序列化value。
4. 根据策略插回内存层。
5. 唤醒等待者。

这里有两个关键点。

第一，I/O并发度要受控。无限制地向SSD提交读写会造成队列拥塞，反而增加尾延迟。因此需要配置read/write并发、队列深度、后台flush任务等。

第二，CPU任务和I/O任务要隔离。序列化、压缩、校验、解压如果都跑在核心业务runtime上，会影响请求调度。高性能cache通常会把重CPU工作放到专门线程池或后台任务里。

## 10. Request Deduplication

Foyer支持request deduplication，解决cache stampede问题。

场景是多个请求同时访问同一个key：

```text
T1: get(k) -> miss -> fetch(k)
T2: get(k) -> miss -> fetch(k)
T3: get(k) -> miss -> fetch(k)
```

如果没有去重，后端会被打三次。对数据库、对象存储、远端服务来说，这是典型的雪崩放大。

有dedup后，路径变成：

```text
T1: 创建 in-flight fetch(k)
T2: 发现已有 in-flight fetch(k)，等待
T3: 发现已有 in-flight fetch(k)，等待
T1完成后，T2/T3共享结果
```

这个机制在hybrid cache里还可以用于磁盘读取：同一个key的并发storage miss/read不应该重复提交多个SSD I/O。

对LLM KV cache系统来说，这一点尤其有用。同一个prefix block如果被多个请求同时需要，应该只从SSD/RDMA/prefill路径加载一次。

## 11. 序列化和压缩

磁盘cache必须把内存里的value变成字节。Foyer提供序列化/反序列化扩展点，同时也支持压缩相关能力。

序列化层要关注：

1. value格式是否稳定。
2. 反序列化是否会产生大量copy。
3. key和metadata是否需要一起写入。
4. 版本升级后旧数据是否还能读。

压缩层的收益和风险都很明确：

1. 压缩可以减少磁盘空间和I/O带宽。
2. 压缩会增加CPU开销。
3. 小对象压缩收益可能不明显。
4. 大对象压缩如果能显著减少读写量，可能降低端到端延迟。

所以压缩不应该无脑开启。它适合I/O瓶颈明显、CPU还有余量、value可压缩性较好的负载。

## 12. Flush和恢复

磁盘cache虽然不是权威存储，但重启后如果能恢复缓存，会明显减少冷启动成本。

恢复能力需要解决两个问题：

1. 哪些磁盘entry是完整写入的。
2. 内存索引如何从磁盘metadata重建。

如果写入过程中进程崩溃，磁盘上可能留下半个block或不完整entry。恢复逻辑必须能识别并跳过这些数据，不能把损坏entry放回索引。

常见做法包括：

1. block header记录magic、version、length。
2. entry带checksum。
3. commit marker或两阶段状态区分writing/committed。
4. 启动时扫描metadata重建索引。

Foyer提供recover相关能力，本质上是在cache层做轻量持久化元数据管理。注意它和数据库WAL不同：cache允许丢数据，但不能读错数据。

## 13. 可观测性

Hybrid cache如果没有metrics，很难调。

至少需要观察：

1. memory hit / storage hit / miss比例。
2. admission reject数量。
3. eviction数量和原因。
4. storage read/write latency。
5. I/O队列深度。
6. reinsertion数量和写放大。
7. 序列化、压缩、解压耗时。
8. 恢复耗时和恢复entry数量。

Foyer提供observability相关接口和配置，这对生产系统很关键。因为cache问题经常不是“能不能跑”，而是“为什么命中率低、为什么尾延迟高、为什么SSD写入量异常”。

## 14. 调参逻辑

Foyer的参数不是越大越好，应该按负载调。

### 14.1 内存容量

内存层太小，热点无法保留，磁盘读压力上升。内存层太大，DRAM成本高，也可能和业务内存抢资源。

经验上应该先观察对象热度分布。如果top N热点已经覆盖大部分访问，内存层只需要覆盖热点；如果访问分布很平，继续加内存收益有限。

### 14.2 磁盘容量

磁盘层容量决定温数据窗口。容量太小会频繁evict，命中率低；容量太大则恢复扫描、元数据内存和设备成本都会上升。

### 14.3 Block大小

block太小，元数据多、I/O碎片化。block太大，读放大和空间浪费明显。

小对象多的负载适合较大的聚合block；大对象多的负载要避免过度打包。

### 14.4 Admission策略

如果负载有大量scan或一次性key，必须启用更强admission。否则cache会被污染。

如果负载本身复用率很高，过强admission可能反而误拒绝新热点。

### 14.5 I/O并发

读并发过低，SSD利用不足；读并发过高，尾延迟变差。

写并发过低，后台积压；写并发过高，会和读抢设备带宽。对在线服务来说，通常读延迟优先级高于写吞吐。

## 15. 技术取舍总结

Foyer的技术价值不在“Rust里写了一个LRU”，而在于它把多个缓存系统技术点组合到一个库里：

1. `HybridCache`统一内存层和磁盘层。
2. sharded memory cache降低并发锁竞争。
3. S3-FIFO/LRU/FIFO-family等策略降低元数据维护成本。
4. admission policy避免cache pollution和SSD写放大。
5. Small/Large/Mixed engine按对象大小选择磁盘布局。
6. BlockEngine用块化/segment化方式组织磁盘cache。
7. reinsertion在回收时保留仍有价值的entry。
8. eviction picker把驱逐从entry级扩展到block/segment级。
9. Direct I/O绕过OS page cache，避免双重缓存。
10. async I/O和runtime配置控制读写并发与尾延迟。
11. request deduplication合并并发miss和并发fetch。
12. 序列化、压缩、flush、recover补齐生产可用性。
13. metrics/observability让命中率、写放大和延迟可诊断。

一句话概括：**Foyer是把缓存策略、异步I/O、磁盘布局、准入控制、回收机制和Rust async生态组合起来的通用hybrid cache系统；它真正值得学的是这些工程技术点如何协同，而不是“用了内存加SSD”这个表层概念。**
