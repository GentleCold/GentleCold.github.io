---
title: ASL论文调研
category: [笔记]
date: 2026-05-08 11:54
tags: [LLM, KV Cache, Long Context]
---

论文：Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference

版本：arXiv:2601.07667v2, 2026-04-16

## 1. 背景

ASL讨论的是长上下文LLM推理中的layer-wise token pruning问题。它的直接上下文是FastKV、GemFilter、PyramidInfer这类方法：在prefill阶段的某一层选择重要token，之后更深层只处理被选中的token，从而减少prefill计算；同时再压缩KV cache，降低decode阶段显存和延迟。

这类方法的关键难点是：**到底应该在哪一层选token？**

如果selection layer太早，模型还没有充分整合上下文信息，attention关注的token集合还不稳定，容易把后续层需要的信息删掉。如果selection layer太晚，精度更安全，但前面大部分层都已经处理了完整上下文，prefill加速收益变小。

FastKV使用固定TSP layer，例如Llama-3.1-8B上取layer 15。ASL论文认为这个固定层对不同任务不鲁棒：简单任务可以早裁，困难任务需要晚裁。固定一个层会导致要么浪费计算，要么困难任务掉精度。

## 2. 论文发现的问题

论文用FastKV在不同selection layer上的表现说明：最佳selection layer和任务强相关。

例如：

- number string、简单NIAH任务中，较早层已经可以确定重要token。
- KV retrieval、multi-key NIAH、带干扰段落的QA任务中，早期attention还不能可靠区分关键token，过早裁剪会严重掉精度。

原因是困难任务里，query和上下文之间可能存在高语义相似干扰项。模型需要更多层推理才能把真正有用的token和干扰token分开。

因此ASL的核心判断是：**selection layer不应该是人工固定的，而应该在推理过程中根据当前样本的attention模式自适应决定。**

## 3. 核心思想

ASL观察的是attention score排序的稳定性。

如果连续若干层里，高attention token的排名变化很大，说明模型还没有稳定地聚焦到某个token子集，此时不适合裁剪。

如果连续若干层里，高attention token的排名基本固定，说明attention已经稳定集中到一小批token上，此时做token selection更可靠。

ASL用一个指标量化这个稳定性：**variance of token ranks**。

直观地说，它不是只看attention score大小，而是看“哪些token排在前面”以及“这些token的名次跨层是否稳定”。当rank variance下降到阈值以下，就认为可以裁token了。

## 4. 方法细节

ASL在prefill阶段运行，从一个起始层$L_{min}$开始观察attention模式。主实验设置中：

- Llama-3.1-8B-UL：$L_{min} = 10$
- Qwen2.5-7B：$L_{min} = 9$
- $L_{obs} = 8$
- 默认阈值$\tau = 0.3$
- 默认KV budget = 2048

### 4.1 attention score预处理

ASL使用recent window tokens作为query，统计它们对context tokens的attention。

为了降低噪声，论文沿用了类似SnapKV的做法，对attention score做1D average pooling。这样不仅关注单个token，也能捕捉上下文中连续区域的重要性。

attention score会在head维度上聚合。对于GQA模型，也会在KV group相关维度上做聚合。

### 4.2 构造top token集合

在当前层$L$，ASL回看最近$L_{obs}$层，也就是从$L - L_{obs}$到$L$。

对每一层，取attention score最高的top-k token。然后把这些层的top-k token做union，得到一个候选集合$\mathrm{top}(L)$。

只在这个候选集合上计算rank variance，原因是attention本身是稀疏的，绝大多数低分token不需要参与稳定性判断。

### 4.3 计算rank variance

对候选集合里的每个token，ASL记录它在最近`Lobs`层中的attention rank。然后计算这个token rank序列的方差，再对候选集合求平均，得到：

$$
\mathrm{var}_{top}(L) = \mathrm{mean}\left(\mathrm{var}(\mathrm{rank}_t \ \text{over recent } L_{obs} \text{ layers})\right)
$$

如果一个token在连续层里排名忽上忽下，方差就大；如果排名稳定，方差就小。

### 4.4 relative variance

不同任务的raw variance尺度不一定可比，因此ASL用$L_{min}$处的初始variance做归一化：

$$
\mathrm{relative\_variance}(L) = \frac{\mathrm{var}_{top}(L)}{\mathrm{var}_{top}(L_{min})}
$$

当：

$$
\mathrm{relative\_variance}(L) < \tau
$$

就把当前层作为selection layer。

这一步让ASL变成task-aware：简单任务的relative variance会更早下降到阈值以下，因此早裁；困难任务下降较慢，因此晚裁。

### 4.5 one-shot token selection

确定selection layer后，ASL只做一次token selection，然后后续所有层只传播这些被选中的token。

这和逐层反复裁剪不同，ASL是one-shot方式。这样实现更简单，也避免每层都重新做复杂的选择逻辑。

## 5. 如何满足KV budget

ASL本身决定的是“在哪一层裁token”和“从这一层之后传播哪些token”。但用户通常还会给定decode阶段的KV cache budget，例如每层只保留2048个token的KV。

论文给了两种组合方式。

### 5.1 ASL + SnapKV

这是论文中称为ASL的一遍方案。

selection layer之后，ASL已经只传播选中的token。selection layer之前的层仍然有完整上下文产生的KV cache。为了让所有层都满足同样的KV budget，可以对selection layer之前的层使用SnapKV做KV cache reduction。

如果ASL和SnapKV都用同一个top-k，那么decode时每层KV cache大小可以被控制在k。

### 5.2 ASL + GemFilter

这是论文中的ASL_2pass。

第一遍运行时，ASL根据rank variance确定selection layer并选出token。第二遍从layer 0开始，只用选出的token重新prefill。

这更接近GemFilter的两阶段流程，可能进一步降低prefill计算，但也可能因为重跑碎片化输入而损失部分上下文整合能力。论文实验也显示ASL_2pass并不总是优于GemFilter或ASL one-pass。

## 6. 成本分析

ASL额外需要维护两类信息：

1. pooled attention scores
2. rank cache

但只保留最近$L_{obs}$层，而且attention scores已经在head维度聚合，所以额外显存较小。

论文给出的例子是：当$L_{obs} = 8$时，对Llama-3.1-8B这类32层、8个KV heads的模型，pooled scores额外开销相比attention计算约为$1/32$级别。

时间开销主要来自：

- attention score pooling：$O(n)$
- rank计算：$O(n \log n)$
- variance计算：$O(L_{obs} \cdot m)$，其中$m$是最近$L_{obs}$层top-k union后的大小

整体上，ASL会比固定层FastKV多一些在线统计开销，并且因为困难任务通常选择更深层，所以TTFT可能比FastKV更高。但它换来的是更稳的精度。

## 7. 实验设置

模型：

- Llama-3.1-Nemotron-8B-UltraLong-1M-Instruct，简称Llama-3.1-8B-UL，32层
- Qwen2.5-7B-Instruct-1M，简称Qwen2.5-7B，28层

Benchmark：

- InfiniteBench，平均上下文长度约214K
- RULER，4K到128K
- Needle-in-a-Haystack，1K到256K

Baseline：

- FastKV
- GemFilter
- PyramidInfer
- SnapKV
- Full KV

默认设置：

- KV budget = 2048
- $\tau = 0.3$
- $L_{obs} = 8$
- $L_{min} = \lfloor \mathrm{num\_layers} / 3 \rfloor$附近，例如32层模型取10

## 8. 主要结果

### 8.1 InfiniteBench

InfiniteBench包含多种长上下文任务，例如英文摘要、英文QA、多选、对话、中文QA、代码debug、数学查找、passkey、number retrieval、KV retrieval等。

论文结论：

- Llama-3.1-8B-UL、KV budget 2048下，ASL_2pass平均分最高。
- Qwen2.5-7B下，ASL平均分最高，并且和FastKV持平或更优。
- 当selection layer之前允许Full KV，而selection layer之后限制到2048时，ASL相对FastKV的优势更明显，尤其是KV retrieval这类困难任务。

这个结果符合ASL的动机：困难任务需要更晚选择token，固定selection layer的FastKV容易太早裁剪。

### 8.2 RULER

RULER上，论文报告了4K、8K、16K、32K、64K、128K上下文长度的平均分。

Llama-3.1-8B-UL、KV budget 2048：

- 4K：FastKV 93.4，ASL 93.7
- 8K：FastKV 85.8，ASL 87.0
- 16K：FastKV 79.3，ASL 79.7
- 32K：FastKV 69.5，ASL 73.0
- 64K：FastKV 63.2，ASL 71.0
- 128K：FastKV 56.1，ASL 66.7

上下文越长，ASL优势越明显。128K时差距达到10.6分。

在Full KV before selection、selection后2048的设置中，差距更大：

- 128K：FastKV 60.6，ASL 69.2

Qwen2.5-7B也有类似趋势。在KV budget 2048下：

- 32K：FastKV 69.9，ASL 75.1
- 64K：FastKV 63.2，ASL 71.7
- 128K：FastKV 59.1，ASL 66.4

在Full KV before selection设置下：

- 128K：FastKV 64.2，ASL 80.9

说明ASL最核心的收益不是来自更强的KV压缩器，而是来自更合适的selection layer。

### 8.3 NIAH

NIAH上，Qwen2.5-7B、上下文长度从1K到256K。

论文结论：

- ASL和ASL_2pass在所有上下文长度上都达到full score，和Full KV一致。
- SnapKV和FastKV基本能找回大部分needle，但在148K长度附近有失败。
- GemFilter在长上下文上表现一般。

这说明ASL在needle retrieval类任务上能更稳地保留关键token。

### 8.4 效率

TTFT方面：

- SnapKV不优化prefill，所以TTFT接近Full KV。
- FastKV、GemFilter、ASL都能降低TTFT。
- GemFilter通常最快，因为它selection layer更早。
- ASL通常比FastKV慢，因为ASL在困难任务上会选择更深层，prefill中full-context部分更多。

TPOT方面：

- 所有KV cache reduction方法都比Full KV快。
- Llama-3.1-8B-UL上，FastKV、GemFilter、ASL在长上下文下TPOT接近。
- Qwen2.5-7B上，SnapKV、FastKV、ASL的TPOT类似，GemFilter最快，ASL_2pass在64K以上也有竞争力。

吞吐方面，128K RULER、2048 KV budget下：

- ASL平均吞吐约为FastKV的74%（Llama-3.1-8B-UL）
- ASL平均吞吐约为FastKV的69%（Qwen2.5-7B）

也就是说ASL不是无代价地全面赢。它牺牲了一部分TTFT/吞吐，换来长上下文困难任务上的精度提升。

显存方面，ASL额外统计信息开销很小，整体内存使用和其他KV cache reduction方法几乎一致。

## 9. 阈值和超参

### 9.1 tau

$\tau$越大，relative variance越容易提前低于阈值，所以selection layer越早，TTFT越低，但可能更容易丢精度。

$\tau$越小，ASL会等到rank更稳定才裁剪，selection layer更晚，精度更稳，但TTFT更高。

论文在RULER 128K上扫了$\tau = 0.2$到$0.6$，最后建议使用$\tau = 0.3$作为精度和TTFT折中。

### 9.2 Lmin

$L_{min}$太小会让ASL从过早层开始观察，relative variance趋势可能更稳定下降，但早期attention噪声也更大。$L_{min}$太大则可能错过可加速空间。

论文建议取$\lfloor \mathrm{num\_layers} / 3 \rfloor$附近，例如32层模型取10。

### 9.3 Lobs

$L_{obs}$控制观察窗口。窗口越小，对层间变化越敏感，但也更容易受噪声影响；窗口越大，稳定性判断更平滑，但反应更慢。

论文选择$L_{obs} = 8$。

## 10. 和FastKV的关系

ASL可以看作是在FastKV思想上的一层自适应扩展。

FastKV已经证明了：前面层full-context，后面层只传播重要token，同时解耦TSP rate和KV retention rate，是有效的长上下文加速方式。

ASL进一步指出：FastKV的固定TSP layer不是所有任务都合适。于是它把“固定TSP layer”替换成“根据attention rank variance动态选择selection layer”。

两者的差异可以概括为：

| 对比项 | FastKV | ASL |
|---|---|---|
| selection layer | 固定，例如layer 15 | 动态，根据relative rank variance判断 |
| 主要目标 | 同时降低prefill和decode成本 | 提升layer-wise pruning在不同任务上的鲁棒性 |
| 速度 | 通常更快 | 困难任务可能更慢 |
| 精度 | 简单任务好，困难任务可能过早裁剪 | 长上下文困难任务更稳 |
| 额外开销 | 较低 | 需要维护pooled score和rank cache |

## 11. 局限

ASL仍然依赖若干超参：

- $\tau$
- $L_{min}$
- $L_{obs}$
- top-k / KV budget
- pooling kernel size

虽然论文给了默认值，但不同模型、不同任务分布下可能还需要调。

ASL的收益主要来自更晚、更稳地裁剪，因此如果任务本身很简单，或者延迟比精度更重要，FastKV这种固定层方法可能更划算。

此外，ASL通过attention rank稳定性判断是否可裁剪，这是一种启发式指标。rank稳定不一定严格等价于“被删token对最终输出无影响”，只能说它在实验上和token重要性稳定有较强相关。

## 12. 总结

ASL的贡献是把layer-wise token pruning里的selection layer从静态超参变成动态决策。

它的推理逻辑是：

1. 从$L_{min}$开始观察最近$L_{obs}$层attention排序。
2. 对top-k token union计算rank variance。
3. 用$L_{min}$处variance归一化成relative variance。
4. 当relative variance低于$\tau$，说明重要token集合稳定，开始one-shot token selection。
5. 后续层只传播选中的token，并结合SnapKV或GemFilter满足KV budget。

一句话概括：**ASL不是发明新的KV压缩目标，而是解决“什么时候裁token”这个问题；它让简单任务早裁以提速，让困难任务晚裁以保精度。**
