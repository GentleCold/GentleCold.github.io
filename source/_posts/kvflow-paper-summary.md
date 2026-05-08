---
title: KVFlow论文调研
category: [笔记]
date: 2026-05-08 18:20
tags: [LLM, Agent, KV Cache, Prefix Cache, LLM Inference]
---

论文：KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows

作者：Zaifeng Pan, Ajjkumar Patel, Zhengding Hu, Yipeng Shen, Yue Guan, Wan-Lu Li, Lianhui Qin, Yufei Ding

版本：arXiv:2507.07400v1, 2025-07-10

链接：https://arxiv.org/abs/2507.07400

## 1. 先说结论

KVFlow解决的是一个很具体但很真实的问题：**多Agent工作流里，LLM serving系统的KV cache淘汰策略不理解Agent未来执行顺序，因此会把马上要复用的固定prompt KV cache淘汰掉。**

已有prefix cache可以复用相同prompt前缀的KV cache，但这只是解决了“能不能复用”的问题。真正部署时，GPU显存有限，cache迟早要被淘汰。传统系统常用LRU，也就是最近最少使用的cache先被删。但在Agent workflow里，LRU经常做错决策：

```text
Planner -> Executor -> Expresser -> Reviewer -> Planner -> ...
```

假设当前正在执行Executor，Expresser的cache可能已经有一段时间没被访问，于是LRU认为它很冷。但从workflow看，Expresser下一步就要执行，它其实非常热。LRU只看过去，不看未来，所以会在Agent场景下误删高价值KV cache。

KVFlow的核心思路是：把Agent workflow的结构告诉serving backend，让backend知道哪个Agent快要执行、哪个Agent短期不会执行。这样KV cache管理就可以从“最近有没有访问过”升级成“未来多久会再次使用”。

它主要做了三件事：

1. 用Agent Step Graph表示Agent执行顺序，并给每个Agent计算`steps-to-execution`。
2. 用workflow-aware eviction policy替代LRU，优先保留快要执行的Agent的KV cache。
3. 用overlapped KV prefetching提前把下一步Agent的KV从CPU拉回GPU，避免请求真正执行时被cache miss阻塞。

论文基于SGLang实现原型。实验中，KVFlow相比SGLang with hierarchical radix cache，在长固定prompt的单workflow场景最高取得1.83x加速；在多workflow并发场景最高取得2.19x加速。

一句话概括：

**KVFlow不是新的attention算法，也不是KV压缩算法，而是一个面向Agent工作流的KV cache调度和搬运策略。**

## 2. 背景：Agent workflow为什么特别适合prefix cache

LLM推理一般分成两个阶段：

1. **prefill**：处理输入prompt，生成整段prompt对应的KV cache。
2. **decode**：逐token生成输出，复用prefill阶段生成的KV cache，并追加新token的KV。

如果prompt很长，prefill成本会很高。尤其是Agent应用里，每个Agent通常都有一段固定prompt：

```text
system prompt
角色定义
工具说明
输出格式约束
few-shot examples
安全策略
任务背景
```

这些内容在同一个Agent的多次调用中高度重复。例如Planner每次都带着Planner的系统提示词，Reviewer每次都带着Reviewer的评审规则。只要这些固定前缀的token序列完全相同，serving系统就可以复用之前算好的KV cache，跳过重复prefill。

这就是prefix cache的价值。

但是Agent workflow比普通聊天还有一个更强的结构：多个Agent之间往往按某种可预测顺序执行。例如：

```text
User request
  -> Planner
  -> Tool Executor
  -> Result Expresser
  -> Critic
  -> Planner
```

这意味着系统不仅知道“某个prefix以前出现过”，还可能知道“哪个Agent马上会再次出现”。KVFlow要利用的就是这部分workflow信息。

## 3. 问题：LRU在Agent workflow里看错了热度

传统cache淘汰里，LRU是一个很自然的基线。它的假设是：最近访问过的数据更可能很快再次访问，最近很久没访问的数据更可能是冷数据。

这个假设在很多通用负载里成立，但在Agent workflow里会被执行顺序打破。

看一个简单例子：

```text
Step 1: Planner
Step 2: Executor
Step 3: Expresser
Step 4: Reviewer
Step 5: Planner
```

当前执行到Step 2，也就是Executor。此时从LRU视角看：

```text
Planner刚用过，很热
Executor正在用，很热
Expresser很久没用，偏冷
Reviewer更久没用，也偏冷
```

但从workflow视角看：

```text
Expresser下一步就要用，最应该保留
Reviewer两步后要用，也应该尽量保留
Planner虽然刚用过，但要到Step 5才再次用
```

LRU的排序和真实未来复用价值并不一致。显存紧张时，LRU可能把Expresser的固定prompt KV淘汰掉。下一步Expresser执行时，系统只能：

1. 重新prefill Expresser的固定prompt。
2. 或者从CPU/host memory把KV cache reactive load回GPU。

前者浪费GPU计算，后者引入CPU到GPU的数据搬运等待。两者都会增加延迟。

因此KVFlow对问题的定义可以写成：

**prefix cache命中不只取决于有没有算过，还取决于算过的KV有没有在正确时间留在GPU里。**

## 4. Agent Step Graph

KVFlow首先把Agent执行顺序抽象成Agent Step Graph。

图里的节点表示Agent invocation，边表示执行依赖。KVFlow关心的不是自然语言内容，而是workflow层面的未来执行距离。论文为每个Agent定义了一个`steps-to-execution`值，表示这个Agent距离下一次执行还有多少个workflow step。

例如当前在Planner，接下来是Executor、Expresser、Reviewer：

```text
Current: Planner

Executor: 1
Expresser: 2
Reviewer: 3
Some inactive agent: +inf
```

这个值越小，说明Agent越快要执行，其固定prompt KV cache越应该留在GPU。

对于有分支的workflow，情况会稍复杂：

```text
Planner
  -> Searcher
  -> Coder
  -> Reviewer
```

Planner之后可能走Searcher，也可能走Coder。KVFlow采用偏保守的做法：对可能在下一步执行的Agent都赋较高优先级，也就是都认为它们很快可能被用到。这样可以减少漏预取或误淘汰，但代价是显存和预取带宽压力会变大。

这个抽象的关键点在于：KVFlow不需要理解Agent在语义上要做什么，它只需要知道执行图和Agent固定prompt的边界。

## 5. Workflow-aware eviction

有了`steps-to-execution`之后，KVFlow就可以替换LRU淘汰策略。

最简单的想法是：

```text
优先保留steps-to-execution小的Agent
优先淘汰steps-to-execution大的Agent
```

也就是越快要执行，越不该被删；越晚才执行，越可以被删。

这有点接近Belady optimal caching的思想。Belady策略会淘汰未来最晚再被访问的数据，但真实系统通常不知道未来。Agent workflow提供了一部分未来信息，因此KVFlow可以用workflow graph近似未来访问顺序。

不过实际prefix cache不是简单的“每个Agent一整块KV”。SGLang这类系统通常使用radix tree或类似的tree-structured cache来存前缀。多个Agent可能共享一部分前缀，例如：

```text
通用系统说明
项目背景
工具定义
  -> Planner专属prompt
  -> Executor专属prompt
  -> Reviewer专属prompt
```

如果只按Agent整体淘汰，就无法正确处理共享前缀。一个共享prefix可能同时被多个快要执行的Agent需要，不能因为某个子Agent暂时不活跃就直接删掉。

因此KVFlow把优先级下沉到KV cache tree的节点级别。大致规则可以理解为：

1. Agent固定prompt对应的KV节点，根据该Agent的`steps-to-execution`赋优先级。
2. 多Agent共享的prefix节点，继承子节点中最值得保留的优先级。
3. 动态suffix优先级较低，因为它通常和具体请求或工具结果绑定，跨Agent复用价值低。

这样做的结果是：系统可以保留高价值共享前缀，同时淘汰短期不会被用到的Agent专属KV或动态后缀。

## 6. Overlapped KV prefetching

只做更聪明的淘汰还不够。因为GPU显存有限，某些KV cache仍然可能被offload到CPU。问题是：什么时候把它们搬回GPU？

传统层级cache一般是reactive loading：

```text
请求到达
发现需要的KV不在GPU
从CPU加载KV到GPU
加载完成后再继续执行
```

这个流程会让请求阻塞在CPU到GPU的数据搬运上。

KVFlow利用workflow信息提前做prefetch：

```text
当前Agent正在GPU上生成
后台线程预测下一步Agent
提前把下一步Agent的KV从CPU拉回GPU
当前Agent结束后，下一步Agent可以直接命中GPU cache
```

论文的判断是：LLM forward主要消耗GPU compute，而CPU到GPU搬运主要消耗PCIe或host-device transfer带宽。两者可以一定程度重叠。只要预取能藏在当前Agent生成时间里，下一步Agent就不需要等待KV load。

这也是KVFlow名字里Flow的含义之一：KV不是等miss发生后再被动搬运，而是沿着Agent workflow提前流向将要执行的位置。

## 7. Status-aware scheduling

预取带来一个新的调度问题：某个请求可能已经可以执行，但它需要的KV cache还在loading中。

如果调度器不理解cache状态，就可能把这个请求调上GPU，然后发现KV没准备好，最后GPU空等。KVFlow为KV cache节点维护状态，例如：

```text
in GPU memory
backup in CPU memory
loading
offloading
```

调度器会尽量选择KV已经ready的请求执行。对于KV还在loading的请求，可以先暂时跳过，执行其他ready请求。这种策略在高并发场景中尤其重要，因为系统可以用其他workflow的计算填补当前workflow等待KV搬运的空隙。

所以KVFlow不是单一的eviction policy，而是三部分协同：

```text
workflow-aware eviction
  -> 少删马上要用的KV

overlapped prefetching
  -> 提前把下一步要用的KV搬回GPU

status-aware scheduling
  -> 避免GPU被未完成的KV loading阻塞
```

## 8. 和SGLang / HiCache的关系

KVFlow原型基于SGLang v0.4.4实现。理解它的位置，可以先看几类系统能力：

| 系统/机制 | 解决的问题 |
|---|---|
| RadixAttention / radix cache | 复用相同前缀的KV cache |
| PagedAttention | 管理KV cache显存页，降低碎片和分配成本 |
| HiCache / hierarchical cache | GPU放不下的KV可以放到CPU或更低层级 |
| KVFlow | 决定Agent workflow中哪些KV该留、哪些KV该提前搬回GPU |

也就是说，KVFlow不是替代SGLang的prefix cache，而是在prefix cache和hierarchical cache之上加入workflow-aware策略。

SGLang的radix cache已经能复用共享前缀，但默认LRU仍然可能删错。HiCache可以把KV offload到CPU，但如果只是reactive loading，miss时仍然会阻塞。KVFlow针对这两个点分别做了改进：

1. 用workflow-aware eviction减少错误淘汰。
2. 用prefetching减少reactive load等待。

## 9. 实验设置

论文实验主要比较三类系统：

1. **SGLang**：GPU-only radix prefix cache。cache miss后需要重新prefill。
2. **SGLang with HiCache**：有层级cache，可以从CPU加载KV，但主要是reactive loading。
3. **KVFlow**：workflow-aware eviction + proactive prefetching + status-aware scheduling。

硬件和模型包括：

1. Llama-3.1-8B on NVIDIA A10G。
2. Qwen2.5-32B on NVIDIA H100。

论文使用了几类负载：

1. 单workflow的sequential multi-agent workload。
2. 多workflow并发场景。
3. PEER-style realistic workflow模拟。

参数上，论文重点观察固定prompt长度、动态prompt长度、输出长度、并发workflow数量对性能的影响。

## 10. 主要结果

### 10.1 单workflow场景

在单workflow的10-agent sequential workload里，KVFlow在长固定prompt时收益最明显。

论文报告，在A10G上，固定prompt为8192 tokens、动态prompt为32 tokens、输出为32 tokens时：

1. KVFlow相比SGLang with HiCache最高取得1.83x加速。
2. KVFlow相比GPU-only SGLang最高取得2.91x加速。

这里的原因比较直观：

1. 固定prompt很长，重新prefill代价高。
2. Agent调用顺序稳定，workflow-aware eviction可以准确保留下一步需要的KV。
3. 输出较短，总延迟里prefill/cache miss占比较高。

如果输出token变多，收益会下降。因为长输出下decode阶段占总时间比例变大，而KVFlow主要优化的是固定prompt prefill和KV搬运，不会直接减少每个输出token的decode计算。

### 10.2 多workflow并发场景

高并发时，KVFlow的价值不只在单个workflow内部，也体现在全局调度。

论文报告，在单H100多workflow并发场景中：

1. KVFlow相比SGLang最高1.25x加速。
2. KVFlow相比LRU-based HiCache reactive loading最高2.19x加速。

这个结果说明：CPU层级cache本身不等于低延迟。如果miss发生时才加载，调度器仍然会被KV搬运卡住。KVFlow通过提前预取和status-aware scheduling，让多个workflow之间可以互相填补等待时间。

### 10.3 PEER-style workflow

论文也测试了更接近真实Agent设置的PEER-style workflow。这个负载里的prompt长度从几十到几百tokens不等，固定prompt不像8192 tokens那样夸张。

结果里KVFlow仍有收益，但幅度明显变小，最高大约在1.08x到1.12x量级。

这个结果很有参考价值：KVFlow不是所有Agent应用都会有巨大收益。它最适合的场景是：

1. 固定prompt很长。
2. workflow结构比较稳定。
3. GPU显存紧张，cache经常被淘汰或offload。
4. prefill/cache miss在总延迟里占比较高。

如果prompt很短，或者主要时间花在工具调用、网络IO、长decode上，KVFlow的端到端收益就会变小。

## 11. 为什么这个设计有效

KVFlow有效的根本原因是：它把应用层知道的未来信息传给了serving层。

普通LLM serving backend看到的是一批请求：

```text
request A
request B
request C
```

它不知道这些请求分别属于哪个Agent，也不知道它们在同一个workflow里的前后关系。于是backend只能用局部信号，例如LRU、cache大小、当前batch状态。

Agent应用层其实知道更多：

```text
当前是Planner
下一步可能是Executor
再下一步可能是Reviewer
Planner的固定prompt在哪里结束
Executor和Reviewer共享了哪些系统说明
```

KVFlow做的事情就是打通这个信息边界。

从系统角度看，这类优化经常有很高性价比：它没有改变模型参数，也没有改变attention语义，只是让资源管理策略获得更准确的未来访问预测。

## 12. 局限

### 12.1 依赖workflow可见性

KVFlow需要知道Agent Step Graph。如果Agent执行完全由模型临时决定，且下一步调用哪个Agent高度不可预测，那么`steps-to-execution`就不准。

对于条件分支，KVFlow可以保守地预取多个可能Agent。但如果分支很多，预取会带来额外GPU显存压力和PCIe带宽压力。

### 12.2 需要应用层和serving层协作

KVFlow不是一个完全透明的backend优化。它需要应用层提供一些metadata，例如：

```text
workflow_id
agent_id
fixed prompt boundary
possible next agents
steps-to-execution
```

这意味着真实落地时，需要改Agent框架、request schema或serving API。对于已经上线的大规模系统，这个集成成本不可忽略。

### 12.3 主要优化固定prompt prefix

KVFlow最擅长的是Agent固定prompt KV复用。对于下面这些场景，它不能直接解决所有问题：

1. RAG检索出来的文档每次顺序不同。
2. 工具返回结果很长且高度动态。
3. 长对话历史经常被压缩或重排。
4. 非前缀片段复用，例如多个请求共享中间某段文档。

这些问题更接近CacheBlend、RAGCache或跨请求KV复用系统关注的范围。KVFlow和它们是互补关系。

### 12.4 没有根治KV存储碎片化

论文提到，SGLang中的KV cache存储布局可能比较碎片化，导致CPU到GPU搬运不能充分利用PCIe带宽。KVFlow通过overlap把一部分搬运时间藏起来，但并没有从底层内存布局上彻底解决碎片化问题。

### 12.5 实际收益取决于端到端瓶颈

如果一个Agent应用主要慢在：

1. 外部工具调用。
2. 网络请求。
3. 数据库查询。
4. 长输出decode。
5. 人类审批。

那么KVFlow即使降低了LLM prefill/cache miss成本，端到端收益也可能有限。

## 13. 工程启发

KVFlow给Agent serving的启发非常直接：不要把Agent workflow拆成一堆彼此独立的LLM请求。

更合理的做法是让serving backend知道这些请求之间的结构关系：

```text
workflow_id: 当前请求属于哪个workflow
agent_id: 当前请求属于哪个Agent
fixed_prefix_range: 哪些token是稳定固定prompt
dynamic_suffix_range: 哪些token是本轮动态输入
next_agents: 下一步可能执行哪些Agent
cache_priority: 未来复用价值大概多高
```

这样backend才能做更聪明的事情：

1. 固定prompt放在更稳定的位置。
2. 工具结果和动态输入尽量放在后缀，避免破坏前缀cache。
3. 对即将执行的Agent提前预取KV。
4. 对短期不会执行的Agent降低GPU驻留优先级。
5. 对共享prefix更谨慎淘汰。

这也说明Agent框架和LLM serving系统之间需要更清晰的接口。应用层只做prompt拼接，backend只做无状态推理，这种边界在简单聊天场景够用，但在复杂Agent workflow里会损失很多优化机会。

## 14. 适用场景

KVFlow适合这些场景：

1. 多Agent系统有明确workflow。
2. 每个Agent有较长且稳定的固定prompt。
3. Agent会在一个任务中反复被调用。
4. 多个Agent共享一部分长前缀。
5. GPU显存不足以保留所有Agent的KV cache。
6. CPU内存相对充足，可以作为KV cache后备层。
7. prefill或cache miss是主要延迟来源。

它不太适合这些场景：

1. 单轮聊天，没有稳定多Agent结构。
2. prompt很短，prefill成本不高。
3. workflow完全动态，下一步Agent很难预测。
4. 主要瓶颈在外部工具或网络IO。
5. 输出很长，decode占据绝大多数时间。
6. 系统无法修改request schema，也无法向backend传递workflow metadata。

## 15. 总结

KVFlow的贡献不是提出一个复杂的新模型，而是指出了Agent serving里的一个信息错位：

**应用层知道workflow未来会怎么走，但serving backend的cache策略只看到了过去访问记录。**

LRU在普通请求里是合理近似，但在多Agent workflow里会把“很久没用”和“马上要用”混淆起来。KVFlow通过Agent Step Graph、workflow-aware eviction、overlapped prefetching和status-aware scheduling，把Agent执行结构转化成KV cache管理策略。

它最值得借鉴的原则是：

**KV cache不应该只按最近访问时间管理，而应该按未来复用价值管理。**

对于正在做Agent系统的人来说，KVFlow还有一个更大的提示：优化Agent性能时，不要只盯着prompt长度和模型速度，也要看serving层是否理解workflow。很多重复prefill和cache miss不是模型问题，而是应用层语义没有传递到系统层导致的资源管理问题。
