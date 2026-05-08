---
title: DeepSeek DSA、DeepSeek V4与vLLM Hybrid KV Cache详解
category: [笔记]
date: 2026-05-08 15:55
tags: [DeepSeek, VLLM, Attention, KV Cache, Mamba]
---

## 1. 先说结论

版本说明：本文参考的是2026-05-08访问的公开资料，主要包括DeepSeek-V3.2-Exp技术报告、vLLM官方DeepSeek V4支持博文、vLLM latest Hybrid KV Cache Manager设计文档、Mamba论文和HuggingFace Transformers的DeepSeek V4文档。vLLM latest文档是developer preview，Hybrid KV Cache Manager文档也明确说该特性仍在早期阶段，具体行为要以实际安装版本为准。

这篇文章想回答几个问题：

1. DeepSeek Sparse Attention，也就是DSA，到底在干什么。
2. DeepSeek V4的attention为什么比普通attention复杂。
3. linear attention、Mamba这类结构和普通attention有什么区别。
4. 这些结构下，vLLM还要不要KV cache。
5. Hybrid KV Cache Manager到底是什么东西。
6. 为什么普通PagedAttention管理不了所有新模型。

先给结论。

**普通Transformer的KV cache保存所有历史token的K和V。**

如果上下文长度是$T$，decode每一步都要看历史$T$个token：

$$
\mathrm{Attention}(q_t, K_{1:t}, V_{1:t})
$$

所以KV cache随着上下文长度线性增长：

$$
O(T)
$$

而attention计算本身在prefill阶段通常是：

$$
O(T^2)
$$

长上下文下，这两个都很贵：

1. KV cache吃显存。
2. attention计算吃算力和显存带宽。

DeepSeek DSA、DeepSeek V4、linear attention、Mamba都在试图解决这个问题，但方法不一样：

| 方法 | 核心想法 | 是否保存所有K/V | 长上下文收益 |
|---|---|---:|---|
| MLA | 缓存压缩latent KV | 不保存完整K/V，保存latent | 主要省KV cache |
| DSA | 用轻量indexer选Top-k历史token | 候选KV仍要缓存，但主attention只看Top-k | 主要省attention计算 |
| DeepSeek V4压缩attention | 多token压缩成少量cache entry，再配合稀疏选择和短滑窗 | 保存压缩后的cache和少量局部状态 | 同时省KV cache和attention计算 |
| Sliding Window Attention | 只看最近窗口 | 只需要保留最近窗口 | 省KV cache和计算，但牺牲远程依赖 |
| Linear Attention | 把softmax attention改写成可递推状态 | 通常不保存逐token K/V，而保存累积状态 | 理论上线性复杂度 |
| Mamba/SSM | 用状态空间递推表示历史 | 不保存KV，保存state | decode状态固定或低增长 |
| Hybrid模型 | 多种层混在一起 | 有些层存KV，有些层存window，有些层存state | 需要按层分别管理 |

vLLM的Hybrid KV Cache Manager就是为最后一种情况准备的：

**一个模型里不同层需要的“缓存”不一样，不能再用一个统一的普通KV cache逻辑管理所有层。**

例如：

```text
full attention层:
  要保存所有历史token的KV

sliding window层:
  只保存最近W个token的KV

Mamba层:
  不保存K/V，而是保存SSM state

DeepSeek V4压缩层:
  保存压缩KV、indexer cache、compressor state、局部SWA cache
```

这就是Hybrid KV Cache Manager的背景。

## 2. 普通KV Cache到底是什么

先从普通Transformer开始。

一层attention通常会把输入$X$投影成：

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

attention计算是：

$$
\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
$$

在decode时，假设已经有$t-1$个历史token，现在要生成第$t$个token。

新token只需要算一个query：

$$
q_t
$$

但它要和所有历史key/value交互：

$$
K_{1:t-1},\quad V_{1:t-1}
$$

所以历史token的$K,V$必须保存下来。

这就是KV cache。

### 2.1 KV cache为什么大

假设模型有：

1. $L$层
2. $H_{kv}$个KV heads
3. 每个head维度$d$
4. 上下文长度$T$
5. dtype大小$b$ bytes

KV cache大小大致是：

$$
2 \times L \times T \times H_{kv} \times d \times b
$$

这里的$2$表示K和V。

上下文翻倍，KV cache也翻倍。

这就是长上下文推理的核心压力。

### 2.2 PagedAttention解决了什么

vLLM最经典的设计是PagedAttention。

它不是改变模型结构，而是改变KV cache的内存管理方式。

普通做法可能希望每个请求的KV cache连续存放：

```text
请求A: [token0 token1 token2 ... token999]
请求B: [token0 token1 token2 ... token499]
```

但请求长度不同、动态进出，会产生碎片。

PagedAttention把KV cache切成blocks，像操作系统分页一样管理：

```text
请求A:
  block 3 -> block 8 -> block 1

请求B:
  block 4 -> block 7
```

逻辑上连续，物理上可以不连续。

它解决的是：

1. KV cache碎片。
2. 动态batch下的内存复用。
3. prefix cache / block sharing。

但注意：

**PagedAttention默认面对的是普通attention的KV cache。**

如果模型里有Mamba、sliding window、压缩attention、KV sharing，情况就复杂了。

## 3. MLA：先把KV变小

MLA是Multi-head Latent Attention。

它是DeepSeek-V2之后很重要的结构。

普通attention保存完整K/V：

```text
每个token:
  K_t
  V_t
```

MLA保存的是压缩latent：

$$
c_t^{KV} = x_t W^{DKV}
$$

后续再从$c_t^{KV}$恢复或投影出attention需要的信息。

直白理解：

```text
普通KV cache:
  直接存K和V，东西多，但用起来直接。

MLA cache:
  存压缩latent，东西少，但计算时要再加工。
```

MLA主要解决：

1. KV cache太大。
2. 长上下文decode读KV太多。

但MLA不一定解决attention计算复杂度。

即使每个token的cache变小，如果一个query仍然要看全部历史token，那么历史长度是$T$时，仍然要处理很多位置。

所以DeepSeek后续又引入DSA，进一步减少“看多少历史token”。

## 4. DSA：不要看所有历史token，只看重要的Top-k

DSA是DeepSeek Sparse Attention。

它最早在DeepSeek-V3.2-Exp技术报告里公开，目标是提升长上下文效率。

DSA的核心想法：

**先用一个便宜的indexer给历史token打分，再只选Top-k历史token做主attention。**

普通attention是：

```text
当前query看所有历史token
```

DSA是：

```text
当前query先问indexer:
  哪些历史token最可能有用？

然后只对这些Top-k token做attention。
```

### 4.1 Lightning Indexer

DSA里有一个lightning indexer。

它会给当前query token $t$和历史token $s$计算一个index score：

$$
I_{t,s}
$$

可以把它理解成：

```text
query t 对历史token s 的粗略相关性评分
```

DeepSeek报告里的indexer设计很轻量，head数少，还可以用FP8实现，因此比完整MLA attention便宜很多。

### 4.2 Top-k token selection

拿到所有历史token的index score后，DSA选择Top-k：

$$
S_t = \mathrm{TopK}(I_{t,:}, k)
$$

然后主attention只看这些token：

$$
u_t = \mathrm{Attn}(h_t, \{c_s \mid s \in S_t\})
$$

如果上下文长度是$L$，普通attention大致看$L$个历史位置。

DSA只看$k$个位置，其中：

$$
k \ll L
$$

所以主attention复杂度从：

$$
O(L^2)
$$

变成近似：

$$
O(Lk)
$$

当然，indexer本身仍然要打分很多历史token，所以不是完全免费。但indexer比主attention轻很多，整体仍然更省。

### 4.3 DSA和MLA的关系

DeepSeek-V3.2-Exp不是抛弃MLA，而是在MLA基础上做DSA。

报告里提到，DSA在MLA下实例化，并且为了GPU效率，使用类似MQA的模式：每个latent KV entry会被多个query heads共享。

直白理解：

```text
MLA:
  每个历史token的KV表示更小。

DSA:
  每个query不看全部历史token，只看Top-k。
```

二者解决的问题不同：

1. MLA减少每个token存多少。
2. DSA减少每次attention看多少token。

所以它们可以叠加。

### 4.4 DSA对KV cache的影响

DSA不是说“不需要KV cache”。

它仍然需要保存可被选中的历史信息，例如MLA latent KV entry、indexer相关cache等。

区别是：

1. 普通attention：所有历史KV都参与主attention。
2. DSA：历史KV仍在，但每次只gather Top-k参与主attention。

所以在vLLM实现上，DSA难点不只是省计算，还包括：

1. 如何保存indexer需要的cache。
2. 如何高效Top-k选择。
3. 如何从KV cache里gather稀疏token。
4. 如何让稀疏访问不把GPU搞慢。

稀疏attention看起来少算了，但如果gather非常零散、kernel不友好，也可能吃掉收益。

## 5. DeepSeek V4 attention：不是单一DSA，而是混合压缩attention

vLLM官方DeepSeek V4博文把DeepSeek V4的attention拆成几个关键点。

DeepSeek V4要解决两个问题：

1. KV cache memory growth。
2. attention computation cost。

它的设计不是只靠一个技巧，而是组合了多种东西。

### 5.1 Key和Value共享

普通attention里，K和V是两份不同向量。

DeepSeek V4里做了key/value sharing，从存储角度可以理解为：

```text
原来:
  保存K
  保存V

现在:
  K/V共享某种表示
```

这能带来接近2倍的KV cache节省。

但K通常会带RoPE位置信息，V通常不带。如果K/V共享，会带来位置表示问题，所以vLLM博文里提到需要inverse RoPE来修正输出。

初学者可以先记住：

**K/V共享能省cache，但需要额外数学处理保持位置性质正确。**

### 5.2 多token压缩：c4a和c128a

DeepSeek V4进一步把多个token压缩成较少的cache entries。

vLLM博文里提到两种模式：

1. `c4a`
2. `c128a`

直白解释：

```text
c4a:
  大约每4个原始token压缩成1个cache entry。
  具体实现里一个compressed token是8个uncompressed token的加权和，stride是4。

c128a:
  大约每128个原始token压缩成1个cache entry。
  一个compressed token来自128个uncompressed token。
```

因此，如果上下文是1M tokens：

```text
c4a后大约还有250K compressed entries
c128a后大约还有8K compressed entries
```

这大幅降低KV cache数量。

但问题是：压缩太狠可能丢局部信息。

所以DeepSeek V4还保留短滑动窗口。

### 5.3 Short Sliding Window保留局部信息

DeepSeek V4使用短滑动窗口，例如vLLM博文里提到window size 128。

作用是：

```text
压缩KV负责远距离历史信息
短滑窗负责最近局部信息
```

为什么需要它？

因为压缩attention有边界。

例如`c128a`要等128个token形成一个压缩entry。如果当前query还没越过某个压缩边界，它可能不能合法地看未来token形成的compressed entry。短滑窗能保证query至少可以看到最近原始tokens。

直白理解：

```text
远处历史:
  看压缩版本。

最近上下文:
  看原始窗口。
```

### 5.4 DSA在DeepSeek V4里的作用

即使使用`c4a`压缩，1M上下文仍然可能有约250K compressed entries。

如果每个query都对250K entries做完整attention，仍然很贵。

因此还需要DSA：

```text
先用indexer选Top-k compressed entries
再对这些entries做主attention
```

这就是DeepSeek V4 attention复杂的原因：

它不是：

```text
普通attention
```

而是混合了：

```text
K/V sharing
多token压缩
DSA稀疏选择
短滑动窗口
indexer cache
compressor state
```

## 6. DeepSeek V4的KV cache到底存什么

普通模型的KV cache可以简单说：

```text
每层保存K和V blocks
```

DeepSeek V4就没这么简单。

vLLM博文里说DeepSeek V4有多种KV state：

1. `c4a` main KV。
2. `c128a` main KV。
3. `c4a` indexer KV。
4. compressor state。
5. sliding window KV。

可以理解成：

```text
main KV:
  主attention真正读取的压缩KV。

indexer KV:
  lightning indexer用来给历史位置打分。

compressor state:
  用于把连续token逐步压缩成compressed entry的滚动状态。

sliding window KV:
  最近局部token的未压缩信息。
```

所以DeepSeek V4的“KV cache”其实不是单一cache，而是一组cache。

### 6.1 为什么vLLM要固定logical block size

不同压缩率会导致自然block大小不同：

```text
c4a:
  256个原始token -> 64个compressed entries

c128a:
  256个原始token -> 2个compressed entries
```

如果每种attention层都用自己的block大小，allocator会非常复杂。

vLLM的做法是：

**逻辑上统一用256个native token positions作为block单位。**

也就是说，调度器和prefix cache看的是：

```text
这个请求的第0-255个原始token
这个请求的第256-511个原始token
...
```

而每种cache在物理上放多少compressed entries，由对应cache类型自己决定。

这样好处是：

1. scheduler不用理解每种压缩细节。
2. prefix cache边界更统一。
3. block分配逻辑更简单。
4. disaggregated prefill也更容易传输。

### 6.2 compressor state为什么像sliding window

压缩不是瞬间发生的。

例如`c4a`可能需要维护一个8-token的滚动状态，`c128a`可能需要维护一个128-token的滚动状态。

这个状态不是完整历史，而是最近一小段。

这很像sliding window：

```text
只关心最近W个token
超过窗口的状态可以丢掉或已经压缩进main KV
```

所以vLLM把compressor state注册到sliding-window KV cache spec下面，让它复用Hybrid KV Cache Manager已有的窗口语义。

这很重要，因为：

1. prefix caching可以复用block语义。
2. disaggregated prefill可以像传SWA state一样传compressor state。
3. CUDA graph和MTP也能按类似方式处理。

### 6.3 为什么要统一page size

即使逻辑block统一了，不同cache类型的物理大小仍然不同。

例如：

```text
c4a main KV page
c128a main KV page
c4a indexer KV page
c4a compressor state page
```

大小可能都不一样。

如果每一种都单独一个block pool，会产生跨pool碎片：

```text
c4a pool空很多，但c128a pool满了
不能互相借
```

vLLM的做法是把不同cache kind归到少数几个page-size bucket里。

DeepSeek V4实现里，vLLM博文说整个cache stack可以放进三个page sizes：

1. largest bucket
2. middle bucket
3. smallest bucket

这样allocator只需要管理少数几个pool。

直白理解：

**不是每种cache单独开一堆内存池，而是把大小相近/可统一的cache放进同一类池子里。**

这能减少碎片和运行时复杂度。

## 7. Linear Attention是什么

现在换一个方向：linear attention。

普通softmax attention是：

$$
\mathrm{softmax}(QK^T)V
$$

问题是$QK^T$会产生一个$T \times T$矩阵。

prefill复杂度大致是：

$$
O(T^2)
$$

linear attention想把它改写成线性形式。

一个常见思路是用特征映射$\phi$近似softmax kernel：

$$
\mathrm{softmax}(q^T k)
\approx
\phi(q)^T \phi(k)
$$

然后attention可以写成：

$$
\frac{\phi(q_t)^T \sum_{s \le t} \phi(k_s)v_s^T}
{\phi(q_t)^T \sum_{s \le t} \phi(k_s)}
$$

核心是维护两个累积状态：

$$
S_t = \sum_{s \le t} \phi(k_s)v_s^T
$$

$$
z_t = \sum_{s \le t} \phi(k_s)
$$

每来一个新token，更新状态：

$$
S_t = S_{t-1} + \phi(k_t)v_t^T
$$

$$
z_t = z_{t-1} + \phi(k_t)
$$

decode时不需要保存所有历史K/V，只需要保存状态$S_t,z_t$。

### 7.1 Linear Attention的cache是什么

普通attention cache：

```text
token 1: K1, V1
token 2: K2, V2
...
token T: KT, VT
```

linear attention cache：

```text
state S
state z
```

它不是逐token KV cache，而是历史信息的聚合状态。

这就是linear attention长上下文省内存的来源。

但是也有代价：

1. 表达能力和softmax attention不同。
2. 很多任务里需要和full attention混合。
3. kernel实现、数值稳定性、训练配方都很重要。

所以很多实际模型不是纯linear attention，而是hybrid：

```text
一些层用linear attention
一些层用full attention
```

这又回到了Hybrid KV Cache Manager的问题。

## 8. Mamba是什么

Mamba是selective state space model。

它不是attention。

普通attention的历史记忆是：

```text
保存所有历史token的K/V
```

Mamba的历史记忆是：

```text
维护一个递推state
```

可以把它想成RNN式更新：

$$
state_t = f(state_{t-1}, x_t)
$$

输出：

$$
y_t = g(state_t, x_t)
$$

真实Mamba比这个复杂，有selective机制和高效scan实现。但初学者抓住一点就够了：

**Mamba不是每个历史token都存K/V，而是把历史压进一个状态里。**

### 8.1 Mamba的cache是什么

Mamba decode时需要保存state。

它不像普通attention那样保存：

```text
K_1, V_1
K_2, V_2
...
K_T, V_T
```

而是保存类似：

```text
conv_state
ssm_state
```

不同实现名字不同，但本质是：

```text
每层一份状态
```

这个状态大小通常不随上下文长度线性增长，或者增长方式远小于普通KV cache。

所以Mamba类结构对长上下文很有吸引力。

### 8.2 Mamba为什么常和attention混合

纯Mamba不一定在所有任务上都替代attention。

很多模型会混合：

```text
full attention层:
  负责全局信息、精确检索、跨长距离依赖

Mamba层:
  负责高效序列建模、压缩历史状态
```

例如vLLM Hybrid KV Cache Manager文档提到的Bamba、Jamba、Minimax等，就是Mamba + full attention类型。

这类模型里：

1. full attention层需要KV cache。
2. Mamba层需要state cache。
3. 两种cache大小、生命周期、prefix cache规则都不一样。

## 9. Sliding Window Attention是什么

Sliding Window Attention只看最近$W$个tokens。

普通attention：

```text
当前token看所有历史tokens
```

SWA：

```text
当前token只看最近W个tokens
```

公式上可以理解为：

$$
\mathrm{Attn}(q_t, K_{\max(1,t-W):t}, V_{\max(1,t-W):t})
$$

它的KV cache只需要保留最近窗口：

```text
保留 token t-W 到 token t
更老的token可以释放
```

所以SWA能显著降低KV cache。

但代价是：

1. 看不到很远的历史。
2. 需要和full attention层混合，才能保留全局能力。

Gemma 2/3、Ministral、Cohere等模型里常见full attention + sliding window的混合结构。

## 10. Local Chunked Attention是什么

Local chunked attention和sliding window类似，也限制可见范围，但组织方式更像按chunk划分。

直白理解：

```text
不是所有token两两attention
而是token主要看本地chunk或有限范围
```

它也会改变KV cache管理：

1. 某些层不需要完整历史KV。
2. 某些旧block可以释放。
3. prefix cache命中规则和full attention不同。

vLLM Hybrid KV Cache Manager文档把Llama4 local attention + full attention列为hybrid模型例子。

## 11. 为什么这些模型让KV cache管理变复杂

普通模型很简单：

```text
每一层都是full attention
每一层都需要所有历史KV
所有层block大小基本一致
prefix cache规则一致
```

hybrid模型变成：

```text
第0层: full attention
第1层: sliding window
第2层: Mamba
第3层: full attention
第4层: local attention
...
```

不同层需要不同缓存：

| 层类型 | 需要保存什么 | 是否保存所有历史 |
|---|---|---:|
| Full Attention | K/V blocks | 是 |
| MLA | latent KV | 是，但每token更小 |
| DSA | latent KV + indexer cache | 候选信息要保存，主attention稀疏读 |
| DeepSeek V4 c4a/c128a | 压缩KV + indexer + compressor state | 保存压缩历史和局部状态 |
| Sliding Window | 最近窗口K/V | 否 |
| Linear Attention | 累积状态 | 否 |
| Mamba | SSM/conv state | 否 |
| KV sharing层 | 复用别的层KV | 自己可能不用分配 |

如果仍然按普通KV cache管理，就会浪费。

比如sliding window层只需要最近1024 tokens，但普通KV manager可能给它保留全部128K tokens。

这会直接浪费显存。

## 12. Hybrid KV Cache Manager是什么

vLLM的Hybrid KV Cache Manager就是为了管理这些混合模型。

官方文档定义的hybrid model包括：

1. sliding window + full attention
2. Mamba + full attention
3. local chunked attention + full attention

Hybrid KV Cache Manager要做两件事：

1. 对不同layer type分配不同slots。
2. 支持layer-specific prefix-cache规则。

### 12.1 不同层分配不同slots

例如full attention层：

```text
需要保存所有tokens
```

sliding window层：

```text
只需要保存最近sliding_window_size个tokens
```

Mamba层：

```text
需要保存state，不是K/V
```

所以manager不能只说：

```text
给请求A分配N个KV blocks
```

而是要说：

```text
给请求A的full attention group分配N个blocks
给请求A的sliding window group分配M个blocks
给请求A的Mamba group分配state blocks
```

### 12.2 不同层prefix cache规则不同

full attention的prefix cache命中要求：

```text
prefix里的所有token KV都还在
```

sliding window的prefix cache命中要求：

```text
只要最后sliding_window_size个token还在
```

因为更老的token本来就不会被这一层看见。

如果一个模型同时有full attention和sliding window，那么最终一个请求能复用多长prefix，要取各组命中结果的交集。

vLLM文档说，block pool里会类似用：

```text
(block_hash, group_id) -> block
```

也就是说，同一段tokens在不同KV cache group里是分别缓存和淘汰的。

### 12.3 KV cache group是什么

可以把KV cache group理解成：

**一组缓存行为相同的层。**

例如一个模型有：

```text
10层full attention
20层sliding window attention
```

可能被分成：

```text
Group 0: full attention layers
Group 1: sliding window layers part 1
Group 2: sliding window layers part 2
```

每个group有自己的SingleTypeKVCacheManager。

HybridKVCacheCoordinator负责协调多个group：

```text
请求要分配cache
  -> full group分配多少
  -> sw group分配多少
  -> 最终组合成一个allocation result
```

## 13. vLLM Hybrid KV Cache Manager的三层结构

vLLM设计文档里说，KVCacheManager组织成三层：

### 13.1 KVCacheManager

这是scheduler和KV cache系统之间的接口。

scheduler不应该关心底层是full attention、sliding window还是Mamba。

它只问：

```text
这个request还能不能分配slots？
这个request命中了多少prefix？
这个request结束后释放哪些blocks？
```

### 13.2 KVCacheCoordinator

Coordinator协调多个group。

不同情况用不同coordinator：

1. `KVCacheCoordinatorNoPrefixCache`
2. `UnitaryKVCacheCoordinator`
3. `HybridKVCacheCoordinator`

如果只有一种KV cache group，用Unitary。

如果有full attention + 另一种efficient attention group，用Hybrid。

文档里也说明，当前HybridKVCacheCoordinator主要处理“正好两个KV cache groups，且必须包含一个full attention group和另一个efficient attention group”的情况。更多复杂情况还不完整。

### 13.3 SingleTypeKVCacheManager

每个SingleTypeKVCacheManager管理一种cache group。

例如：

```text
FullAttentionManager:
  管full attention blocks

SlidingWindowManager:
  管sliding window blocks

Mamba manager:
  管Mamba state相关缓存
```

它实现这一类attention自己的：

1. allocation
2. prefix caching
3. eviction
4. block lookup

## 14. Hybrid KV Cache的几个典型case

### 14.1 Case 1：纯full attention

这是最简单的情况。

```text
所有层都是full attention
所有层都保存完整历史KV
```

只需要普通KV cache manager。

prefix cache命中也简单：

```text
前缀的所有blocks都在 -> 命中
```

### 14.2 Case 2：full attention + sliding window

假设模型层模式是：

```text
full, sw, sw, full, sw, sw, ...
```

full层需要所有历史。

sw层只需要最近窗口。

如果请求长度是100K，sliding window是4K：

```text
full层:
  需要100K token的KV

sw层:
  只需要最近4K token的KV
```

如果不用hybrid manager，sw层也保存100K，就浪费了：

$$
100K - 4K = 96K
$$

个token的KV。

Hybrid manager会让不同group按自己的规则分配。

### 14.3 Case 3：full attention + Mamba

假设模型有：

```text
full attention层
Mamba层
full attention层
Mamba层
```

full attention层需要KV blocks。

Mamba层需要state。

Mamba state的大小可能和attention层的`kv_hidden_size`完全不同，甚至每层state size更大。

vLLM文档提到，Bamba、Jamba、Minimax这类hybrid mamba模型会遇到这个问题。

当前算法大致是：

1. 增大attention层的`block_size`，让一个attention block的物理大小能容纳Mamba state。
2. 对Mamba state做padding，使它适配统一page size。
3. 再使用grouping策略。

这会带来浪费。

文档也明确说这部分仍是work in progress。

### 14.4 Case 4：KV sharing

有些模型某些层复用其他层的KV cache。

例如Gemma-3n相关结构里有KV sharing。

这种情况下：

```text
layer B 使用 layer A 的KV
```

那就没必要给layer B单独分配KV cache。

vLLM Hybrid KV Cache Manager会忽略这些KV sharing层，只给真正需要KV cache的层分配。

然后model runner侧打补丁，让使用共享KV的层拿到正确的cache。

## 15. DeepSeek V4为什么是更复杂的hybrid cache问题

DeepSeek V4不仅是：

```text
full + sliding window
```

它还包含：

1. c4a压缩KV
2. c128a压缩KV
3. indexer cache
4. compressor state
5. sliding window local cache
6. FP8/FP4 cache dtype策略

不同cache有不同大小和生命周期。

vLLM为DeepSeek V4做了特殊实现：

1. 统一logical block size为256 native token positions。
2. 把compressor state当成sliding-window KV cache spec处理。
3. 把多种cache kind归并到少数page-size buckets。
4. 使用kernel fusion减少HBM读写。
5. 使用multi-stream重叠indexer、compression和SWA insertion。

这说明一个趋势：

**未来模型的attention结构越复杂，KV cache manager越像一个小型内存系统，而不是简单tensor数组。**

## 16. DSA / DeepSeek V4 / Linear Attention / Mamba对比

| 机制 | 解决问题 | cache形态 | 优点 | 代价 |
|---|---|---|---|---|
| MLA | KV per token太大 | latent KV | 显著省KV cache | attention仍可能看很多历史位置 |
| DSA | attention看太多历史token | latent KV + indexer cache | 主attention只看Top-k | 需要indexer和稀疏gather |
| DeepSeek V4 | KV和attention都贵 | 压缩KV + indexer + window state | 1M上下文更可行 | 实现复杂，cache种类多 |
| Sliding Window | 完整历史太贵 | 最近窗口KV | 简单高效 | 远程依赖弱 |
| Linear Attention | softmax attention二次复杂度 | 累积状态 | 理论线性 | 表达和训练更难 |
| Mamba | 用状态建模历史 | SSM/conv state | decode状态小 | 不是attention，长距离检索能力需混合 |
| Hybrid模型 | 兼顾能力和效率 | 多种cache并存 | 折中效果好 | serving系统复杂 |

最直白的区别：

```text
MLA:
  每个历史token存得更小。

DSA:
  每次只读一部分历史token。

DeepSeek V4:
  历史token先压缩，再稀疏读，还保留局部窗口。

Linear Attention:
  不保存所有历史token，保存累积状态。

Mamba:
  不做attention，保存递推state。

Hybrid KV Cache Manager:
  让vLLM同时管理这些不同类型的缓存。
```

## 17. 一个具体例子：普通attention vs sliding window vs Mamba

假设上下文长度：

$$
T = 100000
$$

### 17.1 普通full attention层

每层要保存所有历史KV：

```text
token 1 KV
token 2 KV
...
token 100000 KV
```

decode第100001个token时，要读全部历史KV。

cache规模：

$$
O(T)
$$

### 17.2 Sliding window层

window size：

$$
W = 4096
$$

只保存最近4096个token：

```text
token 95905 KV
...
token 100000 KV
```

更老的KV可以释放。

cache规模：

$$
O(W)
$$

如果$W$固定，不随$T$增长。

### 17.3 Mamba层

保存state：

```text
state
```

不保存100000个KV。

cache规模更像：

$$
O(1)
$$

这里的$O(1)$是相对上下文长度说的，不是说state真的只有一个数字。

### 17.4 Hybrid模型

假设模型有30层：

```text
10层full attention
10层sliding window
10层Mamba
```

缓存需求就是：

```text
10层 * 100000 tokens 的full KV
10层 * 4096 tokens 的window KV
10层 * Mamba state
```

这比30层都full attention省很多。

但vLLM必须知道每一层是哪种类型，否则就会分配错。

## 18. 一个具体例子：DeepSeek V4的压缩cache

假设原始上下文：

$$
T = 1,048,576
$$

也就是约1M tokens。

### 18.1 如果全量保存

普通KV cache要保存约1M个位置。

如果每层每token cache很大，总量会很夸张。

### 18.2 c4a

`c4a`大约每4个token生成1个compressed entry。

所以1M tokens会变成约：

$$
\frac{1,048,576}{4} = 262,144
$$

个compressed entries。

vLLM博文里也用“1M上下文下c4a仍有约250K compressed tokens”来解释为什么还需要DSA。

### 18.3 c128a

`c128a`每128个token生成1个compressed entry。

所以1M tokens会变成：

$$
\frac{1,048,576}{128} = 8192
$$

个compressed entries。

8192已经不算大，所以`c128a`层可以更接近full attention over compressed entries。

### 18.4 short sliding window

如果window size是128，那么最近128个原始tokens仍然可被直接访问。

整体就变成：

```text
远处历史:
  compressed entries

近处历史:
  sliding window原始KV

注意力计算:
  对c4a可能用DSA选Top-k
  对c128a可以看较少compressed entries
```

这就是DeepSeek V4能把1M上下文做得更可行的原因。

## 19. vLLM在这种情况下怎么做KV cache

对普通模型：

```text
KVCacheManager:
  管每层K/V blocks
```

对hybrid模型：

```text
KVCacheManager:
  作为统一入口

KVCacheCoordinator:
  协调不同KV cache groups

SingleTypeKVCacheManager:
  各自管理full / sliding window / mamba / compressed cache
```

对DeepSeek V4这种模型，vLLM还需要更定制的布局：

```text
logical block:
  统一按native token positions算

physical page:
  按cache kind放进不同page-size bucket

compressor state:
  当成sliding-window状态管理

indexer cache:
  单独保存，供DSA打分

main compressed KV:
  供主attention读取
```

这比普通KV cache多了很多层。

但目标很简单：

1. 少浪费显存。
2. prefix cache还能工作。
3. disaggregated prefill还能传KV。
4. CUDA graph和kernel fusion还能用。
5. scheduler不需要理解所有attention细节。

## 20. 初学者最容易误解的点

### 20.1 “Mamba没有KV cache，所以vLLM不用管cache”

不对。

Mamba没有普通attention的K/V cache，但它有state。

serving系统仍然要管理这些state：

1. 分配。
2. 保存。
3. 释放。
4. prefix cache相关处理。
5. batch中不同请求的状态索引。

只是它不叫普通KV cache。

### 20.2 “Sliding window就是普通KV cache少存一点”

不完全。

除了少存，它的prefix cache和eviction规则也不同。

full attention要保证整个prefix都还在。

sliding window只关心窗口内的tokens。

### 20.3 “DSA不需要保存历史KV”

不对。

DSA只是主attention不看全部历史token。

但它仍然需要历史token的可选表示，以及indexer需要的cache。

否则Top-k选出来以后没东西可读。

### 20.4 “DeepSeek V4就是DSA”

不准确。

DeepSeek V4包含DSA相关思想，但它的attention机制还包括：

1. K/V sharing
2. c4a/c128a压缩
3. short sliding window
4. compressor state
5. indexer cache
6. custom KV cache layout

只说“DeepSeek V4 = DSA”会漏掉KV cache压缩这条主线。

### 20.5 “Hybrid KV Cache Manager是KV压缩算法”

不是。

Hybrid KV Cache Manager不是模型算法，也不是压缩算法。

它是vLLM里的缓存管理系统。

它负责：

```text
不同层需要不同缓存时，
怎么分配、复用、查prefix、释放、避免浪费。
```

## 21. 对vLLM开发/使用的启发

### 21.1 看模型结构时要看attention type

以后部署模型，不能只看：

```text
参数量
上下文长度
量化格式
```

还要看：

```text
attention类型
是否MLA
是否DSA
是否sliding window
是否Mamba
是否linear attention
是否KV sharing
```

因为这些决定KV cache怎么分配。

### 21.2 长上下文性能不只看max_model_len

一个模型支持1M context，不代表普通KV cache能轻松撑住1M。

要看：

1. 每token cache多大。
2. 是否压缩KV。
3. 是否稀疏attention。
4. 是否滑窗。
5. vLLM是否有对应kernel。
6. Hybrid KV Cache Manager是否支持该结构。

### 21.3 新attention结构会把复杂度转移到系统层

模型论文里说：

```text
KV cache减少
attention计算减少
```

但推理系统还要解决：

1. cache布局。
2. block分配。
3. prefix cache。
4. 稀疏gather。
5. kernel fusion。
6. 多stream并行。
7. disaggregated serving。

DeepSeek V4就是典型例子。结构上省了很多，但vLLM需要专门做复杂实现，才能把理论收益落到实际吞吐上。

## 22. 总结

这篇文章可以用几句话总结。

第一，普通Transformer的KV cache保存所有历史K/V，长上下文下显存和计算都会变贵。

第二，MLA通过latent KV减少“每个token存多少”，DSA通过Top-k sparse attention减少“每个query看多少token”。

第三，DeepSeek V4不是单一DSA，而是把K/V sharing、多token压缩、DSA、short sliding window、indexer cache、compressor state组合起来，目标是在1M上下文下同时降低KV cache和attention计算。

第四，linear attention和Mamba走的是另一条路：不保存逐token K/V，而是维护递推或累积状态。

第五，现代模型越来越hybrid：full attention、sliding window、Mamba、local attention、compressed attention可能混在一个模型里。

第六，vLLM Hybrid KV Cache Manager就是为这种情况服务的。它不是压缩算法，而是一个缓存管理系统，用来按layer type分组管理不同cache，并处理prefix cache、allocation、eviction和memory layout。

一句话概括：

**未来长上下文模型的难点不只是“attention怎么算”，而是“每一层到底需要记住什么、记多久、放在哪里、怎么复用”。Hybrid KV Cache Manager就是vLLM为这个问题做的系统抽象。**

## 23. 参考

1. DeepSeek-V3.2-Exp技术报告：DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention，https://paper.arxivhub.com/DeepSeek_V3_2.pdf
2. vLLM Blog：DeepSeek V4 in vLLM: Efficient Long-context Attention，https://vllm.ai/blog/deepseek-v4
3. vLLM Design Docs：Hybrid KV Cache Manager，https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/
4. HuggingFace Transformers：DeepSeek-V4 model documentation，https://huggingface.co/docs/transformers/main/model_doc/deepseek_v4
5. Mamba论文：Mamba: Linear-Time Sequence Modeling with Selective State Spaces，https://arxiv.org/abs/2312.00752
6. DeepSeek-V2论文页：DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model，https://huggingface.co/papers/2405.04434
