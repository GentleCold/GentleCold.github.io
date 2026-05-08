---
title: ScaleEvict论文调研
category: [笔记]
date: 2026-05-08 11:57
tags: [Distributed Storage, RDMA, Cache]
---

论文：ScaleEvict: Altruistic Eviction for RDMA-enabled Distributed Storage Engines

作者：Till Steinert, Muhammad El-Hindi, Tobias Ziegler, Viktor Leis, Carsten Binnig

发表：DaMoN'26, 2026-05-31 至 2026-06-05

链接：https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/papers/ScaleEvict_DaMoN2026.pdf

## 1. 背景

ScaleEvict讨论的是RDMA-enabled distributed storage engine里的缓存驱逐问题。它的系统背景是transparent shared-cache架构：集群里每个节点不只访问自己的本地DRAM，也可以通过RDMA访问其他节点的DRAM；如果聚合DRAM里都没有目标page，才退回到本地SSD。

这类系统里，不同访问路径的成本差异很大：

1. 本地DRAM最快。
2. 远程DRAM通过RDMA访问，延迟大约是微秒级。
3. SSD访问明显更慢，论文解析中给出的量级约为几十微秒。

因此，一个page从本地缓存被驱逐，并不一定同等危险。如果集群里其他节点还有副本，那么之后可以通过RDMA远程读回来；如果这是集群中唯一副本，那么驱逐后下一次访问就可能落到SSD。

## 2. 问题：本地LRU是自私的

传统LRU只从单个节点的局部视角做决策：哪个page在本地最久没有访问，就优先驱逐哪个page。

这个规则在单机缓存里很自然，但在共享缓存架构里会带来一个问题：**每个节点只优化自己的局部命中率，不关心集群整体DRAM里是否存了太多重复page。**

结果是：

1. 同一个page可能在多个节点的本地缓存中重复存在。
2. 聚合DRAM看似很大，实际有效容量被重复副本吃掉。
3. 某些全局唯一page被LRU驱逐后，系统只能从SSD重新加载。
4. 当工作集接近或超过聚合DRAM容量时，吞吐和尾延迟会明显恶化。

也就是说，LRU不知道一个page在全局缓存中的复制状态。它只知道“这个page在我这里冷不冷”，不知道“这个page被我删掉后，集群是否还留着副本”。

## 3. 核心思想

ScaleEvict的核心思想是altruistic eviction，也就是“利他驱逐”。

这里的利他不是说牺牲本节点性能，而是说驱逐决策要考虑集群整体价值：优先删掉那些对全局缓存容量损失最小、后续恢复成本最低的page。

论文把page大致分成三类：

| 类型 | 含义 | 驱逐优先级 | 原因 |
|---|---|---|---|
| Duplicate | 多个节点都有缓存副本 | 最高 | 驱逐后仍可从其他节点DRAM读回 |
| Unique-Remote | 只有一个非目录节点持有副本 | 中等 | 仍可能通过远程DRAM访问，但副本唯一 |
| Unique-Local | 只有目录节点或本地关键位置持有副本 | 最低 | 驱逐后更容易退化到SSD |

因此ScaleEvict不是简单替换LRU的recency信号，而是在recency之外加入复制状态和全局价值判断。

一句话概括：**LRU问“我多久没用它”，ScaleEvict还要问“别人是不是也有它，以及我删掉它会不会让集群失去这个page的DRAM副本”。**

## 4. 为什么可以实现

ScaleEvict能成立的关键，是系统本来就有目录协议。

在ScaleStore这类系统里，目录节点需要维护page的全局状态，例如哪些节点缓存了这个page、当前一致性状态是什么。ScaleEvict复用了这部分信息，而不是引入一个全新的全局集中式缓存管理器。

大致流程是：

1. 工作节点先根据本地状态生成一批驱逐候选。
2. 候选被发送到对应目录节点。
3. 目录节点根据全局复制状态判断哪些page适合被驱逐。
4. 工作节点执行被批准的驱逐。

这样做的好处是，ScaleEvict没有要求每个节点实时同步完整缓存状态。它只在驱逐路径上批量查询目录节点，用已有一致性元数据来增强本地驱逐决策。

## 5. 设计细节

ScaleEvict的驱逐评分可以理解为三个信号的组合：

1. `local_recency`：本地访问时间，保留LRU一类策略对局部热点的判断。
2. `replication_state`：page在集群中的复制状态，是Duplicate还是Unique。
3. `global_utility`：page对集群整体缓存效率的估计价值。

真正重要的是排序方向：在本地冷page里，优先驱逐Duplicate page；对于Unique page，要更谨慎，因为这类驱逐可能把未来访问推到SSD。

这个思路和很多单机缓存算法不同。单机缓存优化的是“我的缓存里该留谁”，ScaleEvict优化的是“整个集群有限DRAM里该保留哪些唯一内容”。

## 6. 实验结果

论文解析中给出的实验基于4到6节点集群，每节点约50GiB buffer pool，负载主要使用YCSB，对比ScaleStore原生LRU。

主要结论有几个：

1. **更高资源效率**：在out-of-memory工作集下，ScaleEvict用约三分之二的DRAM就能达到LRU相同吞吐。
2. **更高吞吐**：相同DRAM容量下，ScaleEvict在部分设置中吞吐可达到LRU的约2倍。
3. **更低尾延迟**：当聚合DRAM能容纳工作集时，ScaleEvict显著减少因错误驱逐唯一page导致的SSD访问，p99延迟改善明显。
4. **读写混合下仍有效**：在100%读、读多写少、读写均衡、写多读少等配置下，ScaleEvict总体保持优势。

这些结果说明它提升的不是某个微观操作，而是聚合DRAM的有效容量。重复副本越多、工作集越接近内存边界，ScaleEvict相对LRU的价值越明显。

## 7. 适用场景

ScaleEvict适合这类系统：

1. 集群节点之间有低延迟RDMA网络。
2. 系统使用shared-cache或disaggregated memory风格架构。
3. 远程DRAM访问明显快于SSD访问。
4. 目录协议或类似元数据层已经知道page副本分布。
5. 工作集大到会对聚合DRAM形成压力。

如果系统没有远程内存访问能力，或者所有page都严格分片、几乎没有跨节点副本，那么ScaleEvict的收益会小很多。

## 8. 局限

ScaleEvict依赖目录节点掌握较准确的全局复制状态。如果目录信息滞后，驱逐决策可能不再最优。

它也会增加驱逐路径复杂度：本地节点不能只维护一个简单LRU链表，还要批量生成候选、和目录节点交互、处理批准或拒绝结果。虽然论文强调可以异步批量化，但工程实现仍然比本地LRU复杂。

此外，ScaleEvict主要解决缓存副本冗余和唯一page保护问题。如果性能瓶颈来自写放大、锁竞争、网络拥塞或应用层热点，它不能单独解决所有问题。

## 9. 总结

ScaleEvict的贡献是把分布式存储引擎里的缓存驱逐从“节点本地最优”推进到“集群全局更优”。

它的核心逻辑是：

1. 远程DRAM比SSD快很多，所以只要集群里还有副本，驱逐本地副本的代价相对可控。
2. LRU不知道page是否有其他副本，因此可能保留重复page、误删唯一page。
3. ScaleEvict利用目录协议提供的复制状态，优先驱逐Duplicate page，保护Unique page。
4. 结果是减少聚合DRAM浪费，提高有效缓存容量，降低SSD fallback概率。

一句话概括：**ScaleEvict不是让每个节点缓存命中率最高，而是让整个集群的DRAM少存重复内容、多保留唯一内容，从而提升共享缓存架构下的吞吐和尾延迟。**
