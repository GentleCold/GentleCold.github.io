---
title: vLLM最新版调度系统与Continuous Batching详解
category: [笔记]
date: 2026-05-08 15:30
tags: [VLLM, LLM Inference, Scheduler, Batch, KV Cache]
---

## 1. 先说结论

版本说明：本文参考的是2026-05-08访问的vLLM官方`latest`文档和API源码页面。vLLM文档明确提示`latest`是developer preview，不等同于latest stable release。因此生产环境要以你实际安装的vLLM版本为准，最好用：

```bash
vllm serve --help
```

确认参数是否存在。

这篇文章讲vLLM最新版调度系统，重点是：

1. scheduler每一步到底在干什么。
2. batch size在vLLM里不是一个简单数字，而是多个预算共同限制。
3. continuous batching为什么能提升吞吐。
4. `max_num_batched_tokens`、`max_num_scheduled_tokens`、`max_num_seqs`分别控制什么。
5. chunked prefill如何让长prompt不要堵住decode。
6. KV cache不够时，为什么会preempt。

最重要的结论先放前面：

**vLLM V1 scheduler不是简单地区分“prefill队列”和“decode队列”，而是用每个请求的`num_computed_tokens`追赶它当前应该计算到的位置。每一步调度时，先给RUNNING请求分配token预算，再把WAITING请求加入进来。**

用一句话说：

**vLLM的调度单位不是“请求”，而是“这一轮给每个请求算多少个新token”。**

这就是continuous batching的基础。

## 2. 为什么LLM serving需要调度

先看最朴素的推理方式。

假设来了3个请求：

```text
请求A：prompt 1000 tokens，生成100 tokens
请求B：prompt 50 tokens，生成20 tokens
请求C：prompt 200 tokens，生成80 tokens
```

最简单的服务方式是：

```text
处理A -> A全部生成完
处理B -> B全部生成完
处理C -> C全部生成完
```

这叫一个请求一个请求处理。

问题很明显：

1. GPU经常吃不满。
2. B这种短请求可能被A长请求堵住。
3. decode阶段每个请求每步只生成1个token，单独跑非常浪费。
4. 新请求来了不能马上加入当前GPU计算，只能等前面请求结束。

所以推理服务必须调度。

调度器要解决的问题是：

1. 哪些请求这一轮可以上GPU。
2. 每个请求这一轮算多少token。
3. KV cache够不够。
4. 如果不够，谁要等，谁要被抢占。
5. 新请求什么时候加入batch。
6. 已完成请求什么时候释放KV cache。

## 3. Prefill和Decode的差别

LLM推理有两个阶段：

1. prefill
2. decode

### 3.1 Prefill

prefill处理prompt。

如果prompt有$T$个token：

$$
x_1, x_2, \ldots, x_T
$$

prefill要一次性把这些token喂进模型，算出每层KV cache。

prefill特点：

1. token多。
2. 计算量大。
3. 矩阵乘比较大，容易吃满GPU算力。
4. 长prompt会显著影响TTFT。

### 3.2 Decode

decode每次生成新token。

每一步通常只为每个请求生成1个token：

```text
第1步：每个请求生成1个token
第2步：每个请求再生成1个token
第3步：继续
```

decode特点：

1. 每个请求每步新增token少。
2. 需要读历史KV cache。
3. 长上下文下容易受显存带宽限制。
4. 单个请求单独decode很浪费，多请求batch起来才高效。

## 4. 传统Static Batching的问题

传统batching可以理解成：

```text
凑一批请求 -> 这批请求一起跑 -> 全部结束 -> 再凑下一批
```

例如：

```text
Batch 1: A, B, C
```

如果A要生成100个token，B只要生成20个token，C要生成80个token，那么：

1. B第20步就完成了。
2. 但如果batch是静态的，B的位置可能空着。
3. 新请求D不能立刻进来填B的位置。
4. GPU batch利用率逐步下降。

可以画成这样：

```text
step 1-20:   A B C  都在跑
step 21-80:  A _ C  B结束，位置空着
step 81-100: A _ _  C也结束，只剩A
```

这就是static batching的浪费。

## 5. Continuous Batching是什么

continuous batching，也叫in-flight batching。

核心思想是：

**batch不是固定的一组请求。每一步GPU计算前，scheduler都重新决定这一轮有哪些请求参与。**

还是上面的例子：

```text
step 1-20:   A B C
B结束后，新请求D进来
step 21-50:  A D C
C结束后，新请求E进来
step 81-100: A D E
```

这样GPU batch不会因为某些请求提前结束而空掉。

vLLM的continuous batching可以直白理解为：

1. 每一轮调度都看当前还有哪些请求没完成。
2. 已完成的请求释放KV cache。
3. 新请求只要有token预算和KV cache，就可以加入。
4. 每个请求这一轮可以只算一小段token。
5. 下一轮再重新调度。

所以continuous batching的好处是：

1. 提高GPU利用率。
2. 降低排队时间。
3. 支持长短请求混合。
4. 让decode阶段能持续保持较大batch。

代价是：

1. scheduler更复杂。
2. KV cache管理更复杂。
3. 每步都要构造新的执行计划。
4. 请求之间长度不同，attention/KV布局更复杂。

vLLM的PagedAttention和KV block管理，就是为了让这种动态batch能够高效运行。

## 6. vLLM V1 Scheduler的核心抽象

最新版vLLM V1 scheduler源码注释里有一个很关键的说法：

**scheduler里没有固定的“decode phase”或“prefill phase”。**

每个请求只看两个数字：

1. `num_computed_tokens`
2. `num_tokens_with_spec`

其中：

```text
num_tokens_with_spec =
len(prompt_token_ids)
+ len(output_token_ids)
+ len(spec_token_ids)
```

直白解释：

1. `num_computed_tokens`：这个请求已经被模型实际计算过多少token。
2. `num_tokens_with_spec`：这个请求现在理论上需要计算到哪里。
3. scheduler每一步做的事：让`num_computed_tokens`追上`num_tokens_with_spec`。

如果一个请求刚进来，还没prefill：

```text
num_computed_tokens = 0
num_tokens_with_spec = prompt长度
```

它需要补上整个prompt，这就是prefill。

如果一个请求已经完成prefill，刚采样出一个新token：

```text
num_computed_tokens = prompt长度
num_tokens_with_spec = prompt长度 + 1
```

它只需要补1个token，这就是decode。

如果启用了speculative decoding，可能一次有多个draft token：

```text
num_tokens_with_spec = prompt长度 + 已接受输出 + draft tokens
```

scheduler仍然只是看差距：

$$
\mathrm{num\_new\_tokens}
= \mathrm{num\_tokens\_with\_spec}
- \mathrm{num\_computed\_tokens}
$$

这就是最新版调度设计很统一的地方。

## 7. Scheduler里有哪些队列

从源码看，V1 scheduler主要有这些请求集合：

1. `waiting`
2. `skipped_waiting`
3. `running`
4. `finished_req_ids`

### 7.1 waiting

`waiting`里是等待被调度的新请求，或者被抢占后等待恢复的请求。

新请求来了，不是直接跑，而是先进入waiting。

### 7.2 running

`running`里是已经进入系统、占有或即将占有KV cache、可以在后续step继续被调度的请求。

注意：running不代表每一步一定会被执行。

源码里也有断言说明：

```text
some requests in the RUNNING queue may not be scheduled in this step
```

也就是说，请求在running队列里，但这一轮可能因为token预算、KV cache、encoder预算、pipeline等原因没有被安排。

### 7.3 skipped_waiting

`skipped_waiting`是一些暂时不能调度的waiting请求。

例如：

1. 等远端KV加载。
2. 等structured output grammar准备。
3. 因为LoRA数量限制暂时不能进来。
4. 因为某些异步依赖还没完成。

这些请求不是失败了，只是这轮先跳过，之后再尝试。

### 7.4 finished_req_ids

`finished_req_ids`记录两轮调度之间已经完成的请求。

worker/model runner需要知道这些请求结束了，这样才能释放KV cache等资源。

## 8. 一次schedule()到底做什么

最新版V1 scheduler的`scheduled()`主流程可以概括成：

```text
1. 初始化本轮token_budget
2. 先调度RUNNING请求
3. 再调度WAITING请求
4. 检查预算和running数量
5. 构造SchedulerOutput
6. 构造KV connector metadata
7. 更新num_computed_tokens等状态
```

更细一点：

```text
schedule()
  token_budget = max_num_scheduled_tokens

  for request in running:
      计算这个request还差多少token
      受long_prefill_token_threshold限制
      受token_budget限制
      分配KV blocks
      如果KV不够，可能preempt别的request
      记录本轮给它算多少token

  while waiting还有请求 and token_budget > 0:
      如果running数量达到max_num_seqs，停止加入新请求
      取一个waiting请求
      查prefix cache / external KV cache
      算它还需要多少token
      如果chunked prefill关闭且放不下，停止
      如果chunked prefill开启，就只调度能放下的部分
      分配KV blocks
      加入running
      记录本轮给它算多少token

  生成SchedulerOutput给model runner
  更新request.num_computed_tokens
```

最重要的是前两步：

1. **先照顾已经running的请求。**
2. **还有预算才接纳waiting的新请求。**

这避免了大量新请求不断插队，把已经在decode的请求饿死。

## 9. token_budget是什么

每一轮调度都有一个token预算：

```text
token_budget = max_num_scheduled_tokens
```

如果没有特别设置：

```text
max_num_scheduled_tokens = max_num_batched_tokens
```

它的意思是：

**这一轮scheduler最多能发给model runner多少个新token去计算。**

注意这里说的是“新token”，不是“请求数”。

例如：

```text
max_num_scheduled_tokens = 16
```

本轮可以这样安排：

```text
请求A decode 1 token
请求B decode 1 token
请求C decode 1 token
请求D prefill 13 tokens
总计 16 tokens
```

也可以这样：

```text
请求A prefill 16 tokens
总计 16 tokens
```

也可以这样：

```text
16个decode请求，每个1 token
总计 16 tokens
```

所以vLLM里batch size不能只看请求数量，还要看token数量。

## 10. max_num_batched_tokens是什么

`max_num_batched_tokens`是单次iteration最多处理多少token。

官方latest文档里的描述是：

```text
Maximum number of tokens that can be processed in a single iteration.
```

直白解释：

**它控制每一轮GPU计算的token总规模上限。**

如果它太小：

1. 长prompt会被切成很多小块。
2. prefill需要更多轮才能完成。
3. TTFT可能上升。
4. 每轮batch小，GPU可能吃不满。

如果它太大：

1. 单轮prefill可能很大。
2. decode请求可能被长prefill挤压。
3. 显存和临时activation压力更大。
4. CUDA graph/torch compile相关缓存也可能受影响。

它和`max_model_len`还有约束关系。

如果没有开启chunked prefill，那么：

$$
\mathrm{max\_num\_batched\_tokens}
\ge
\mathrm{max\_model\_len}
$$

否则长prompt可能根本无法一次放进一个iteration。

vLLM配置校验里也会检查：

1. 如果`max_num_batched_tokens < max_model_len`且没开chunked prefill，会报错。
2. `max_num_batched_tokens`必须大于等于`max_num_seqs`。
3. 如果`max_num_batched_tokens > max_num_seqs * max_model_len`，会给warning。

## 11. max_num_scheduled_tokens是什么

`max_num_scheduled_tokens`是scheduler这一轮最多发出多少token。

官方latest文档说：

```text
Maximum number of tokens that the scheduler may issue in a single iteration.
```

通常它等于`max_num_batched_tokens`。

为什么还要单独有这个参数？

因为有些情况下，模型runner会在batch里额外追加token，例如speculative decoding。此时scheduler发出去的token数可以小于runner最终处理的token容量。

简单理解：

1. `max_num_batched_tokens`：model runner这一轮最多能处理多少token。
2. `max_num_scheduled_tokens`：scheduler这一轮主动安排多少token。

普通非spec decode场景里，可以认为它们基本一样。

## 12. max_num_seqs是什么

`max_num_seqs`是单次iteration最多处理多少个sequence。

官方latest文档描述是：

```text
Maximum number of sequences to be processed in a single iteration.
```

它限制的是请求/序列数量，不是token数量。

例如：

```text
max_num_seqs = 4
max_num_batched_tokens = 32
```

那么最多同时有4个sequence进入这一轮计算。

可能的batch：

```text
A: 1 token
B: 1 token
C: 10 tokens
D: 20 tokens
总tokens = 32
总seqs = 4
```

不能这样：

```text
32个请求，每个1 token
总tokens = 32
总seqs = 32
```

因为超过了`max_num_seqs = 4`。

所以调度约束至少有两个维度：

$$
\sum_i \mathrm{scheduled\_tokens}_i
\le
\mathrm{max\_num\_scheduled\_tokens}
$$

以及：

$$
\mathrm{num\_running\_requests}
\le
\mathrm{max\_num\_seqs}
$$

这就是为什么“batch size”在vLLM里不能只说一个数字。

## 13. vLLM里的batch size到底是什么

初学者最容易把batch size理解成：

```text
batch size = 一次处理多少个请求
```

在训练里这样理解经常没问题。但在LLM serving里不够。

vLLM里至少有三种“batch大小”：

### 13.1 请求数batch

也就是这一轮有多少个sequence/request。

对应参数：

```text
max_num_seqs
```

decode阶段常常是这个更直观，因为每个请求通常只生成1个token。

### 13.2 token数batch

也就是这一轮总共处理多少token。

对应参数：

```text
max_num_batched_tokens
max_num_scheduled_tokens
```

prefill阶段更看这个，因为一个请求可能一下子有几千个prompt tokens。

### 13.3 KV cache batch

也就是当前系统里所有running请求占用了多少KV blocks。

这个没有简单等于某个batch参数。它取决于：

1. 请求数量。
2. 每个请求上下文长度。
3. block size。
4. prefix cache命中情况。
5. 是否有preemption。
6. 是否有DCP/PCP。

举例：

```text
请求A: 上下文 100K
请求B: 上下文 100 tokens
```

它们在`max_num_seqs`里都算1个请求，但KV cache占用完全不是一个量级。

所以vLLM调参时，要同时看：

1. `max_num_seqs`
2. `max_num_batched_tokens`
3. KV cache使用率
4. GPU显存
5. TTFT/TPOT

## 14. Chunked Prefill为什么重要

chunked prefill的意思是：

**一个长prompt不一定一次性prefill完，可以拆成多个chunk，分多轮算。**

例如一个请求prompt长度是10000 tokens。

如果：

```text
max_num_batched_tokens = 2048
enable_chunked_prefill = True
```

那么它可能被拆成：

```text
第1轮：prefill 2048 tokens
第2轮：prefill 2048 tokens
第3轮：prefill 2048 tokens
第4轮：prefill 2048 tokens
第5轮：prefill 1808 tokens
```

这样做的核心好处是：

**长prefill不会一次吃完整个iteration预算，decode请求有机会穿插进来。**

如果没有chunked prefill，长prompt可能要么一次性占满一轮，要么因为预算不够而无法调度。

源码里waiting请求调度时有一个关键逻辑：

```text
如果chunked prefill关闭，并且num_new_tokens > token_budget，
就停止调度这个waiting请求。

如果chunked prefill开启，
num_new_tokens = min(num_new_tokens, token_budget)
```

也就是说，开启chunked prefill后，长请求可以“先算一部分”。

## 15. Chunked Prefill和Continuous Batching的关系

continuous batching解决的是：

```text
每一步都可以重新组batch
```

chunked prefill解决的是：

```text
一个长prefill可以拆成多步加入batch
```

两者配合起来，就能处理长短请求混合。

例如：

```text
max_num_batched_tokens = 8

请求A：prompt 20 tokens
请求B：decode 1 token
请求C：decode 1 token
```

如果没有chunked prefill，A需要20 tokens，超过预算8，可能无法调度。

如果有chunked prefill：

```text
step 1:
  B decode 1
  C decode 1
  A prefill 6
  总计8

step 2:
  B decode 1
  C decode 1
  A prefill 6
  总计8

step 3:
  B decode 1
  C decode 1
  A prefill 6
  总计8

step 4:
  B decode 1
  C decode 1
  A prefill 2
  总计4
```

这样B、C不会因为A的长prompt完全卡住。

## 16. long_prefill_token_threshold是什么

`long_prefill_token_threshold`用于限制一次给长prefill请求调度多少token。

源码里RUNNING和WAITING调度都有类似逻辑：

```text
if 0 < long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = long_prefill_token_threshold
```

直白解释：

**如果某个请求这轮还差很多token，就最多只给它`long_prefill_token_threshold`个token。**

这样可以避免一个超长prefill请求吃掉太多token预算。

例如：

```text
max_num_scheduled_tokens = 4096
long_prefill_token_threshold = 512
```

某个长prompt还差10000 tokens没算。

不加threshold，它可能一口气拿走4096 tokens。

加了threshold，它这一轮最多拿512 tokens，剩下预算可以留给decode请求和其他prefill请求。

代价是：

1. 这个长请求自己的TTFT可能变高。
2. 但系统整体延迟更平滑。
3. decode请求TPOT更不容易被长prefill拖慢。

## 17. max_num_partial_prefills是什么

`max_num_partial_prefills`控制并发partial prefill数量。

如果它大于1，就允许多个prefill请求同时以chunk形式推进。

官方配置校验里有两个重要点：

1. `max_num_partial_prefills > 1`时必须启用chunked prefill。
2. 如果`long_prefill_token_threshold == 0`，vLLM会把它设成`max_model_len * 0.04`左右。

还有一个参数：

```text
max_long_partial_prefills
```

它限制“长prefill”并发数量，并且必须小于等于`max_num_partial_prefills`。

直白解释：

```text
max_num_partial_prefills:
  最多有多少个prefill请求可以被切块并发推进。

max_long_partial_prefills:
  其中最多有多少个是长prefill。
```

这可以避免很多超长prompt同时进入系统，把decode请求和短prefill全挤掉。

## 18. Prefix Cache在调度里的位置

vLLM支持prefix caching。

当一个WAITING请求第一次被调度时，如果：

```text
request.num_computed_tokens == 0
```

scheduler会尝试查本地prefix cache：

```text
kv_cache_manager.get_computed_blocks(request)
```

如果启用了KV connector，还会查外部KV：

```text
connector.get_num_new_matched_tokens(...)
```

如果命中了一部分prefix，调度器就不用重新计算这些tokens。

例如：

```text
请求A prompt: [系统提示 + 文档X + 问题1]
请求B prompt: [系统提示 + 文档X + 问题2]
```

如果B来的时候A的前缀KV还在cache里，那么B的：

```text
系统提示 + 文档X
```

可能直接复用。

调度上会变成：

```text
num_computed_tokens = 命中的prefix长度
num_new_tokens = request.num_tokens - num_computed_tokens
```

也就是说，prefix cache命中会减少这个请求需要调度的新token数。

## 19. KV Connector和远端KV加载

最新版scheduler里也考虑KV connector。

如果外部KV cache命中，但需要异步加载，调度器不会马上安排这个请求计算新token。

它会设置：

```text
load_kv_async = True
request.status = WAITING_FOR_REMOTE_KVS
```

然后把请求放到`skipped_waiting`里，等远端KV加载完成后再继续。

这能支持P/D分离、KV offloading、LMCache/Mooncake/NIXL等场景。

直白理解：

```text
本地prefix cache命中:
  可以马上少算一些token。

外部KV命中但还没搬到GPU:
  先等KV搬回来，再继续调度。
```

## 20. KV cache不够时发生什么

调度不只是token预算，还要看KV cache够不够。

每个请求要继续计算新token，就需要为这些token分配KV blocks：

```text
kv_cache_manager.allocate_slots(...)
```

如果分配成功，这个请求可以被调度。

如果分配失败，说明KV cache不够。

这时scheduler可能会preempt某个running请求。

源码里逻辑大致是：

1. 如果policy是priority，抢占最低优先级请求。
2. 否则从running队列末尾pop一个请求。
3. 释放它的KV cache。
4. 把它状态设成`PREEMPTED`。
5. 放回waiting队列。

被抢占请求会：

```text
num_computed_tokens = 0
```

也就是说，它之后要重新计算，除非prefix cache等机制还能命中部分内容。

所以preemption不是免费操作。

它的代价是：

1. 释放KV cache能救当前step。
2. 但被抢占请求后续可能要重算。
3. preemption太多通常说明KV cache压力过大。

常见原因：

1. `max_num_seqs`太大。
2. `max_model_len`太大。
3. 长上下文请求太多。
4. `gpu_memory_utilization`不够。
5. KV cache dtype太大。
6. prefix cache/offloading策略不合适。

## 21. Scheduling policy：FCFS和Priority

vLLM latest SchedulerConfig里有`policy`参数。

可选策略包括：

1. `fcfs`
2. `priority`

### 21.1 FCFS

FCFS就是first come first served。

请求按到达顺序处理。

优点：

1. 简单。
2. 行为容易理解。
3. 比较公平。

缺点：

1. 长请求可能排在前面，影响短请求。
2. 不适合所有业务优先级相同但SLA不同的场景。

### 21.2 Priority

priority按请求优先级调度。

官方文档说明：

```text
lower value means earlier handling
```

也就是优先级数值越小越早处理。

如果优先级相同，再看到达时间。

适合：

1. 付费用户优先。
2. 交互式请求优先。
3. 后台批处理低优先级。

代价：

1. 低优先级请求可能等待更久。
2. 需要业务层正确设置priority。
3. 如果滥用，可能造成饥饿问题。

## 22. Async Scheduling是什么

`async_scheduling`用于减少GPU利用率空隙。

官方latest文档说，async scheduling可以避免GPU utilization gaps，从而改善latency和throughput。

直白理解：

普通同步调度可能是：

```text
GPU跑完一步
CPU scheduler开始算下一步计划
GPU等待
下一步计划好了
GPU继续跑
```

async scheduling希望让CPU调度和GPU执行更重叠：

```text
GPU跑当前step
CPU提前准备后续调度
GPU少等CPU
```

收益：

1. 降低GPU空等。
2. 提升吞吐。
3. 降低调度开销对延迟的影响。

限制：

1. 行为更复杂。
2. 和structured outputs、spec decode、pipeline parallel等能力可能有兼容性限制。
3. 具体支持情况要看版本。

## 23. 一个完整调度例子

假设配置：

```text
max_num_scheduled_tokens = 8
max_num_seqs = 4
enable_chunked_prefill = True
```

当前系统里：

```text
running:
  A: 已完成prefill，下一步decode 1 token
  B: 已完成prefill，下一步decode 1 token

waiting:
  C: 新请求，prompt 10 tokens
  D: 新请求，prompt 3 tokens
```

### Step 1

先调度running：

```text
A: 1 token
B: 1 token
剩余token_budget = 6
```

再调度waiting。

C需要10 tokens，但预算只剩6。因为chunked prefill开启，所以C可以先拿6 tokens：

```text
C: prefill 6 tokens
剩余token_budget = 0
```

D这一轮进不来。

本轮batch：

```text
A decode 1
B decode 1
C prefill 6
总计8 tokens
```

### Step 2

A、B继续decode。

C还差4个prompt tokens。

```text
A: 1
B: 1
C: 4
剩余token_budget = 2
```

这时D可以进来，D prompt 3 tokens，但预算只剩2：

```text
D: prefill 2
```

本轮batch：

```text
A decode 1
B decode 1
C prefill 4
D prefill 2
总计8 tokens
```

### Step 3

C已经完成prefill，可以开始decode。

D还差1个prompt token。

```text
A decode 1
B decode 1
C decode 1
D prefill 1
总计4 tokens
```

这就是continuous batching的直觉：

**每一步都动态混合decode token和prefill chunk。**

## 24. 为什么continuous batching能提升吞吐

decode阶段如果单请求跑，GPU很难吃满。

一个请求每步只有1个新token：

```text
batch = 1 token
```

这太小了。

continuous batching把很多请求的decode token合并：

```text
请求A decode 1
请求B decode 1
请求C decode 1
...
请求N decode 1
```

这样每一步变成：

```text
batch = N tokens
```

GPU利用率更高。

同时，完成的请求会离开，新请求会进来，batch规模更稳定。

吞吐提升主要来自：

1. decode阶段batch更大。
2. 请求完成后不浪费空位。
3. prefill和decode可以混合。
4. chunked prefill避免长prompt独占整轮预算。

但它不是免费午餐。

代价包括：

1. 每步调度复杂。
2. KV cache碎片管理复杂。
3. prefix cache和block table维护复杂。
4. 动态batch对kernel和CUDA graph提出更多要求。

vLLM的核心价值就在于把这些复杂性封装起来。

## 25. 调参时应该看哪些指标

### 25.1 TTFT

TTFT是time to first token。

它主要受：

1. 排队时间。
2. prefill长度。
3. prefix cache命中。
4. chunked prefill切分。
5. preemption。
6. GPU负载。

影响。

如果TTFT很高：

1. 看是否有长prompt堵住。
2. 看prefix cache是否命中。
3. 看`max_num_batched_tokens`是否太小。
4. 看`max_num_partial_prefills`是否限制太严。
5. 看是否频繁preempt。

### 25.2 TPOT

TPOT是time per output token。

它主要受：

1. decode batch大小。
2. KV cache读取量。
3. 上下文长度。
4. attention backend。
5. GPU显存带宽。
6. DCP/TP通信。

影响。

如果TPOT很高：

1. 看decode batch是否太小。
2. 看上下文是否过长。
3. 看KV cache是否重复或显存压力太大。
4. 看GPU utilization和memory bandwidth。

### 25.3 Throughput

吞吐一般看：

```text
tokens/s
requests/s
```

如果吞吐低：

1. batch可能太小。
2. `max_num_seqs`可能太低。
3. `max_num_batched_tokens`可能太低。
4. GPU可能在等CPU scheduler。
5. KV cache不够导致preemption。

### 25.4 Preemption次数

preemption多通常不是好事。

它说明：

1. running请求太多。
2. KV cache不够。
3. 长上下文太多。
4. 调度预算和KV预算不匹配。

解决方向：

1. 降低`max_num_seqs`。
2. 降低`max_model_len`。
3. 增大KV cache可用显存。
4. 使用KV cache quantization。
5. 长上下文模型考虑DCP。

## 26. 常见调参思路

### 26.1 高并发短请求

特点：

1. prompt短。
2. output短。
3. 请求多。

通常关注：

1. `max_num_seqs`
2. decode batch规模
3. CPU scheduler开销

可以尝试：

```text
提高max_num_seqs
适当提高max_num_batched_tokens
开启/保留async_scheduling
```

但如果`max_num_seqs`过高，KV cache和调度开销也会上来。

### 26.2 长prompt短输出

特点：

1. prefill占大头。
2. TTFT敏感。
3. decode不是主要瓶颈。

可以尝试：

```text
提高max_num_batched_tokens
启用chunked prefill
调max_num_partial_prefills
使用prefix cache
```

如果长prompt之间共享前缀，prefix cache非常重要。

### 26.3 短prompt长输出

特点：

1. decode占大头。
2. 要保持较大的decode batch。
3. TPOT更重要。

可以尝试：

```text
提高max_num_seqs
保证KV cache足够
避免频繁preemption
关注TPOT和memory bandwidth
```

### 26.4 长prompt长输出

这是最难的情况。

需要同时关心：

1. TTFT
2. TPOT
3. KV cache显存
4. preemption
5. prefix cache
6. chunked prefill

通常不能只靠一个参数解决。

## 27. 更详细的调度例子

前面讲了概念，这一节专门用数字走几遍。为了容易看懂，例子会故意把参数设得很小。真实线上可能是几千、几万token预算，但逻辑一样。

### 27.1 例子一：只有decode请求时，continuous batch怎么补位

配置：

```text
max_num_scheduled_tokens = 4
max_num_seqs = 4
```

假设当前有4个请求都已经完成prefill，正在decode：

```text
A: 还要生成3个token
B: 还要生成1个token
C: 还要生成2个token
D: 还要生成4个token
```

每个decode请求每轮只需要1个新token。

#### Step 1

running里有A、B、C、D。

本轮调度：

```text
A: decode 1
B: decode 1
C: decode 1
D: decode 1
总tokens = 4
总seqs = 4
```

Step 1结束后：

```text
A: 还要2个
B: 完成
C: 还要1个
D: 还要3个
```

B完成，释放它的KV cache和running位置。

此时新请求E来了，prompt已经很短，假设它也很快进入decode。

#### Step 2

continuous batching不会等A/C/D都结束才接E，而是下一轮就可以重新组batch：

```text
A: decode 1
C: decode 1
D: decode 1
E: prefill或decode 1
总tokens = 4
总seqs = 4
```

这就是continuous batching最直观的收益：

**B结束后留下的位置，下一轮就能被E填上。**

如果是static batching，可能要等A/C/D全部结束后，E才能进入下一批。

### 27.2 例子二：max_num_seqs和max_num_batched_tokens同时限制

配置：

```text
max_num_scheduled_tokens = 8
max_num_seqs = 3
enable_chunked_prefill = True
```

waiting里有5个短请求：

```text
A: prompt 2 tokens
B: prompt 2 tokens
C: prompt 2 tokens
D: prompt 2 tokens
E: prompt 2 tokens
```

从token预算看：

```text
5个请求 * 2 tokens = 10 tokens
```

因为`max_num_scheduled_tokens = 8`，最多只能放8个tokens，所以最多放4个请求。

但还有`max_num_seqs = 3`，所以实际最多只能放3个请求。

Step 1调度：

```text
A: prefill 2
B: prefill 2
C: prefill 2
总tokens = 6
总seqs = 3
剩余token_budget = 2
```

虽然还剩2个token预算，D也刚好只需要2个tokens，但不能加入，因为`max_num_seqs`已经满了。

所以本轮剩余token预算会浪费掉。

这说明：

**token预算够，不代表还能加请求；还要看sequence数量预算。**

如果把配置改成：

```text
max_num_scheduled_tokens = 8
max_num_seqs = 4
```

Step 1就可以变成：

```text
A: prefill 2
B: prefill 2
C: prefill 2
D: prefill 2
总tokens = 8
总seqs = 4
```

### 27.3 例子三：长prefill和decode混合，没有chunked prefill会怎样

配置：

```text
max_num_scheduled_tokens = 8
max_num_seqs = 4
enable_chunked_prefill = False
```

当前running里有两个decode请求：

```text
A: decode 1
B: decode 1
```

waiting里有一个长prompt请求：

```text
C: prompt 10 tokens
```

Step 1先调度running：

```text
A: 1
B: 1
剩余token_budget = 6
```

现在看C。

C需要10 tokens，但预算只剩6。因为chunked prefill关闭，C不能只算前6个tokens，所以C这轮进不来。

本轮batch：

```text
A decode 1
B decode 1
总tokens = 2
```

明明还有6个token预算，却用不上。

如果后续每轮都有A/B decode占用2个tokens，C一直需要10 tokens，就可能迟迟进不来，除非某一轮有足够预算。

这就是不开chunked prefill时，长prompt和decode混合可能出现的问题。

### 27.4 例子四：同样场景，开启chunked prefill

配置只改一个：

```text
enable_chunked_prefill = True
```

还是：

```text
max_num_scheduled_tokens = 8
max_num_seqs = 4

running:
  A: decode 1
  B: decode 1

waiting:
  C: prompt 10 tokens
```

#### Step 1

先调度A/B：

```text
A: decode 1
B: decode 1
剩余token_budget = 6
```

C需要10 tokens，但chunked prefill开启，所以C可以先算6个tokens：

```text
C: prefill 6
```

本轮batch：

```text
A decode 1
B decode 1
C prefill 6
总tokens = 8
```

Step 1结束：

```text
C还差4个prompt tokens
```

#### Step 2

继续：

```text
A: decode 1
B: decode 1
C: prefill 4
总tokens = 6
```

C完成prefill。

#### Step 3

C开始decode：

```text
A: decode 1
B: decode 1
C: decode 1
总tokens = 3
```

这个例子说明：

**chunked prefill让长prompt可以分多轮进入continuous batch，不会因为单次预算不足而完全卡住。**

### 27.5 例子五：long_prefill_token_threshold如何保护decode

配置：

```text
max_num_scheduled_tokens = 16
max_num_seqs = 8
enable_chunked_prefill = True
long_prefill_token_threshold = 4
```

当前：

```text
running:
  A: decode 1
  B: decode 1

waiting:
  C: prompt 100 tokens
  D: prompt 4 tokens
  E: prompt 4 tokens
```

#### 没有long_prefill_token_threshold时

A/B先拿走2个tokens：

```text
剩余token_budget = 14
```

C是长prompt，可能直接拿走14个tokens：

```text
C: prefill 14
```

本轮结束，D/E进不来。

#### 有long_prefill_token_threshold = 4时

A/B先拿走2个tokens：

```text
剩余token_budget = 14
```

C虽然还差100 tokens，但这一轮最多拿4个：

```text
C: prefill 4
剩余token_budget = 10
```

D拿4个：

```text
D: prefill 4
剩余token_budget = 6
```

E拿4个：

```text
E: prefill 4
剩余token_budget = 2
```

本轮batch：

```text
A decode 1
B decode 1
C prefill 4
D prefill 4
E prefill 4
总tokens = 14
```

这个配置让长请求C慢一点，但D/E这种短prefill不用一直等。

所以它的作用不是让单个长请求最快，而是让系统更公平、更平滑。

### 27.6 例子六：prefix cache命中如何减少调度token

假设有两个请求：

```text
A prompt:
  [系统提示100 tokens] + [文档X 900 tokens] + [问题1 20 tokens]

B prompt:
  [系统提示100 tokens] + [文档X 900 tokens] + [问题2 20 tokens]
```

A先执行完成prefill后，vLLM缓存了前缀KV。

B来的时候，如果prefix cache命中：

```text
命中prefix = 系统提示100 + 文档X900 = 1000 tokens
B总prompt = 1020 tokens
```

那么B不需要重新计算全部1020 tokens，只需要计算剩下的20 tokens：

```text
num_computed_tokens = 1000
num_tokens_with_spec = 1020
num_new_tokens = 20
```

如果配置：

```text
max_num_scheduled_tokens = 512
```

没有prefix cache时，B的prefill可能要两轮：

```text
step 1: 512 tokens
step 2: 508 tokens
```

有prefix cache时，只要一轮：

```text
step 1: 20 tokens
```

这就是为什么长前缀场景里，请求路由和prefix cache很重要。

如果是DP部署，还要注意：

```text
A在DP rank 0
B在DP rank 1
```

默认情况下，B不能直接命中rank 0里的prefix cache。除非有外部KV cache/KV transfer机制。

### 27.7 例子七：KV cache不够时为什么会preempt

假设KV cache总共只能放10个blocks。

当前running：

```text
A: 已占4 blocks，还要继续decode
B: 已占4 blocks，还要继续decode
```

剩余：

```text
free blocks = 2
```

现在waiting里来了C。

C的prompt很长，这一轮至少需要分配4个新blocks。

但free blocks只有2个。

scheduler尝试：

```text
allocate_slots(C) -> 失败
```

这时如果策略允许preemption，scheduler可能从running队列里抢占一个请求，比如B：

```text
preempt B
释放B的4 blocks
B放回waiting
B.num_computed_tokens = 0
```

现在free blocks变成：

```text
2 + 4 = 6
```

C可以分配4个blocks并进入running。

这看起来解决了C的问题，但代价是B之后要重新算。

如果频繁发生这种情况，说明系统过载：

1. running请求太多。
2. 长上下文太多。
3. KV cache太小。
4. `max_num_seqs`可能太大。
5. `max_model_len`可能设得过高。

所以preemption不是优化目标，而是压力过大时的兜底机制。

### 27.8 例子八：调大max_num_seqs为什么可能变慢

假设：

```text
max_num_scheduled_tokens = 64
```

情况A：

```text
max_num_seqs = 8
```

每轮最多8个请求。decode阶段每个请求1 token，所以纯decode batch最多8 tokens。

GPU可能吃不满。

情况B：

```text
max_num_seqs = 64
```

纯decode batch最多64 tokens，吞吐可能提高。

但问题是：64个running请求都会占KV cache。

如果每个请求上下文都很长：

```text
64个请求 * 每个请求长上下文KV
```

KV cache可能爆掉，然后出现preemption。

结果可能是：

1. 理论decode batch变大。
2. 但KV cache压力变大。
3. preemption增加。
4. 被抢占请求重算。
5. 端到端延迟反而变差。

所以`max_num_seqs`不是越大越好。它要和KV cache容量、请求长度分布一起调。

### 27.9 例子九：调大max_num_batched_tokens为什么可能影响decode延迟

假设：

```text
max_num_scheduled_tokens = 4096
enable_chunked_prefill = True
long_prefill_token_threshold = 0
```

当前有：

```text
running:
  A/B/C/D: decode请求，各1 token

waiting:
  E: prompt 10000 tokens
```

如果没有额外限制，E可能每轮拿走大量剩余token预算：

```text
A/B/C/D: 4 tokens
E: 4092 tokens
总计4096
```

这对E自己的prefill很快，但会让每轮GPU计算变成“大prefill + 少量decode”的混合。某些场景下decode请求的TPOT可能被影响。

如果设置：

```text
long_prefill_token_threshold = 512
```

调度会更像：

```text
A/B/C/D: 4 tokens
E: 512 tokens
剩余预算给其他请求
```

这样长prefill对decode的冲击更小，但E的TTFT会变高。

这就是一个典型取舍：

1. 想让长prompt尽快首token：给它更大prefill chunk。
2. 想让在线decode更稳定：限制长prefill每轮吃掉的预算。

### 27.10 例子十：一个简化的线上混合负载

假设线上同时有三类请求：

```text
类型S：短问答
  prompt 100 tokens
  output 50 tokens

类型R：RAG问答
  prompt 8000 tokens
  output 200 tokens

类型B：后台批处理
  prompt 20000 tokens
  output 100 tokens
```

如果所有请求同等对待，后台批处理B可能吃掉大量prefill预算，影响S类交互请求。

一个更合理的思路：

```text
max_num_seqs:
  不能太小，否则S类decode batch不够
  不能太大，否则KV cache压力过大

max_num_batched_tokens:
  要足够支持R/B类chunked prefill
  但不能让单轮prefill过大影响decode

long_prefill_token_threshold:
  限制B类长prompt每轮最多吃多少token

priority:
  S类高优先级
  R类中优先级
  B类低优先级
```

调度效果大致是：

```text
每轮先推进正在decode的S/R请求
再给R类prefill分配一部分预算
最后给B类后台请求分配剩余预算
```

这不是一个固定公式，但它说明了vLLM调度参数的本质：

**不是单纯追求最大batch，而是在交互延迟、吞吐、KV cache和公平性之间做平衡。**

## 28. 常见误区

### 28.1 batch size不是只有max_num_seqs

`max_num_seqs`只限制序列数量。

如果每个请求prompt很长，真正限制你的是token预算和KV cache。

### 28.2 max_num_batched_tokens越大越好

不一定。

太大可能让长prefill占用一整轮，影响decode延迟，也会增加显存和编译相关压力。

### 28.3 max_num_seqs越大越好

也不一定。

太大会让更多请求进入running，占用更多KV cache，可能导致preemption。

### 28.4 chunked prefill一定降低TTFT

不一定。

对单个长请求来说，chunked prefill可能让它分多轮完成，TTFT可能变高。

但对混合负载来说，它能避免长请求堵住其他decode请求，让整体延迟更平滑。

### 28.5 preemption只是普通排队

不是。

preemption会释放KV cache，并把请求放回waiting。被抢占请求可能需要重算，代价比普通等待大。

## 29. 一句话总结

最新版vLLM V1 scheduler的核心不是“先prefill再decode”这种固定流程，而是：

**每一轮用有限的token预算和KV cache预算，让所有请求的`num_computed_tokens`尽量追上它们当前需要计算到的位置。**

continuous batching的核心也不是简单“把请求攒成一批”，而是：

**每一步都重新组batch，让已完成的请求离开，让新请求加入，让长prefill拆成chunk，让decode请求持续推进。**

如果只记三个参数：

1. `max_num_batched_tokens`：一轮最多处理多少tokens。
2. `max_num_scheduled_tokens`：scheduler一轮最多发出多少tokens，通常等于前者。
3. `max_num_seqs`：一轮最多处理多少sequences。

如果只记一个调度顺序：

```text
先RUNNING，后WAITING；
先看token预算，再看KV cache；
能chunk就切块，KV不够就可能preempt。
```

## 30. 参考

1. vLLM latest API：SchedulerConfig，https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/
2. vLLM latest API：V1 Scheduler，https://docs.vllm.ai/en/latest/api/vllm/v1/core/sched/scheduler/
3. vLLM latest docs：Engine Arguments，https://docs.vllm.ai/en/latest/configuration/engine_args/
4. vLLM latest docs：Automatic Prefix Caching，https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html
5. vLLM latest docs：Production Metrics，https://docs.vllm.ai/en/latest/usage/metrics.html
