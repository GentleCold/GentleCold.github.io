---
title: FastKV论文调研
category: [笔记]
date: 2026-05-08 11:54
tags: [LLM, KV Cache, Long Context]
---

论文：FastKV: Decoupling of Context Reduction and KV Cache Compression for Prefill-Decoding Acceleration

版本：arXiv:2502.01068v7, 2026-04-20

代码：https://github.com/dongwonjo/FastKV

## 1. 背景

长上下文LLM推理的成本主要来自两个阶段：

1. prefill阶段：模型需要一次性处理完整prompt，attention复杂度随上下文长度增长很快，长上下文下TTFT会明显变差。
2. decoding阶段：每生成一个新token，都要访问历史token的KV cache，KV cache越大，显存占用和每步解码开销越高。

已有KV cache压缩方法大致可以分成两类：

1. decoding-only方法，例如StreamingLLM、H2O、SnapKV。这类方法主要压缩decode阶段使用的KV cache，因此能降低TPOT和显存，但prefill还是要处理完整上下文，TTFT收益有限。
2. prefill-aware方法，例如GemFilter、PyramidInfer。这类方法在prefill阶段就减少后续层处理的token，因此能降低TTFT，但通常把prefill中的context reduction和decode阶段的KV budget绑在一起。KV budget越小，prefill时丢掉的token也越多，容易伤精度。

FastKV的核心问题定义是：**prefill阶段应该传播多少上下文，和decoding阶段应该保留多少KV cache，不应该被同一个压缩比例绑定。**

## 2. 核心观察

论文的关键观察是：不同层对上下文的依赖是不一样的。

早期层的attention模式还不稳定，不同token之间的依赖关系仍在形成。如果太早删token，后面层可能再也拿不到某些必要信息，错误会沿层传播。

但到中后层之后，重要token集合会逐渐稳定。也就是说，并不是所有层都必须处理完整上下文：前面层可以 full-context 处理，等重要token相对稳定之后，后面的层只传播高saliency token。

同时，decode阶段真正被新token频繁attend到的prefill token也只是少数。因此，prefill中“让哪些hidden state继续往后层传播”和decode中“每层KV cache保留多少”可以分开设计。

## 3. 方法

FastKV由两个部分组成：

1. Two-stage prefill with Token-Selective Propagation
2. Layer-wise KV retention

### 3.1 Token-Selective Propagation

FastKV在模型中选择一个固定的TSP layer。以论文主实验为例：

- LLaMA-3.1-8B-Instruct：TSP layer = 15
- Ministral-8B-Instruct：TSP layer = 17
- observation window size = 8
- pooling kernel size = 7
- TSP rate = 20%

prefill分成两个阶段：

1. 从第0层到TSP layer，所有层仍处理完整上下文。
2. 到TSP layer时，根据attention分数选出salient tokens。后续层只处理这些被选中的token以及recent window tokens。

token saliency的计算方式大致是：用最近窗口token作为query，观察它们对上下文token的attention权重；在head维度上聚合，再做pooling，得到每个上下文token的重要性分数。之后取top-ranked tokens，数量由TSP rate控制。

recent window tokens会被强制保留，因为新近token对生成通常很重要，而且它们也承担局部上下文连续性的作用。

这个设计和GemFilter的重要差别是：GemFilter会先选token，然后用选出来的碎片化输入重新prefill；FastKV则让早期层先看完整上下文，到中间层之后再减少传播的hidden states。因此它减少了“早期层信息还没整合就被删掉”的风险。

### 3.2 Layer-wise KV retention

FastKV还引入独立的KV retention rate，用于控制每层保留多少KV cache给decode阶段使用。

这和TSP rate是两个独立超参：

- TSP rate：控制TSP layer之后有多少token继续参与prefill后续层计算，主要影响TTFT。
- KV retention rate：控制每层KV cache保留比例，主要影响显存和TPOT。

例如可以使用相对保守的TSP rate保护prefill精度，同时使用更小的KV retention rate压低decode成本。反过来，如果显存压力特别大，也可以同时降低二者。

这就是论文标题里所谓的decoupling：**context reduction和KV cache compression解耦。**

## 4. 和已有方法的区别

### 4.1 对比SnapKV

SnapKV压缩的是decode阶段使用的KV cache，但prefill仍然需要完整执行。长上下文下，如果输入非常长、输出不长，那么prefill占主导，SnapKV的端到端延迟收益会被限制。

FastKV在prefill中也减少后续层token数，因此TTFT会下降。

### 4.2 对比GemFilter

GemFilter可以同时降低prefill和decode成本，但它的prefill reduction和KV budget耦合更强。为了得到小KV cache，它会更激进地减少输入token，精度容易掉。

FastKV保留早期full-context计算，并且让TSP rate和KV retention rate独立，精度-效率曲线更灵活。

### 4.3 对比PyramidInfer

PyramidInfer从较早层开始逐层减少KV/cache，容易在token依赖还不稳定时丢信息。论文实验里PyramidInfer在多个长上下文设置下精度和工程可运行性都不如FastKV，部分设置还会OOM。

## 5. 实验设置

模型：

- LLaMA-3.1-8B-Instruct，32层，GQA，128K context
- Ministral-8B-Instruct，36层，GQA，128K context
- 附录中还报告了Mistral-Nemo-12B-Instruct

Benchmark：

- LongBench：单文档QA、多文档QA、摘要、few-shot、synthetic、code等长上下文任务
- RULER：多种可控长度和检索难度的长上下文benchmark
- Needle-in-a-Haystack：测试长上下文中的needle retrieval能力

实现：

- 基于HuggingFace Transformers self-attention
- 使用FlashAttention-2
- 主实验TSP rate = 20%
- KV retention rate通常测10%和20%

## 6. 主要结果

### 6.1 LongBench

在LongBench上，FastKV在降低prefill compute和KV cache的同时，平均精度接近full-context。

以LLaMA-3.1-8B-Instruct为例：

- Full-context平均分：50.19
- SnapKV 10% KV：48.73
- SnapKV 20% KV：49.43
- GemFilter 10% KV：48.47
- GemFilter 20% KV：49.07
- FastKV 10% KV：48.47
- FastKV 20% KV：49.07

论文文本强调：GemFilter在某些设置下会因为重跑碎片化prefill导致明显精度损失，最高平均drop可到11.58%；而FastKV通过早期full-context和后期selective propagation，在效率接近GemFilter的同时保持更接近full-context的精度。

在Ministral-8B-Instruct上，FastKV也保持了相对稳定的平均表现，20% KV retention下平均分为51.07，接近full-context的51.87。

### 6.2 RULER

RULER上，LLaMA-3.1-8B-Instruct、10% KV retention、上下文长度到128K：

- Full-context平均：86.0
- StreamingLLM：18.6
- SnapKV：73.6
- GemFilter：69.6
- FastKV：75.6

FastKV相比GemFilter和SnapKV有更好的平均精度，尤其是在长上下文下还能保持较好表现。

### 6.3 Needle-in-a-Haystack

Needle-in-a-Haystack中，LLaMA-3.1-8B-Instruct、10% KV retention，平均16K到128K长度：

- Full-context：99.0
- StreamingLLM：33.5
- SnapKV：99.0
- GemFilter：95.8
- FastKV：99.9

这里FastKV甚至略高于full-context。论文解释是，TSP有时能让模型更集中在全局关键token上。

### 6.4 延迟

论文在单张A100 SXM上测端到端延迟，固定生成256个token，变化输入长度。

结论：

- full-context在长输入下延迟快速增长，128K时prefill占主导。
- SnapKV等decode-only方法可以降低decode成本，但prefill仍然完整，所以长prompt下端到端收益受限。
- GemFilter和FastKV都能同时降低prefill和decode延迟。
- 128K上下文下，FastKV和GemFilter相对full-context都有超过2倍端到端speedup。
- 论文摘要报告FastKV最高可达1.82x prefill speedup和2.87x decoding speedup。

GemFilter有时更快，因为它的filter layer更早、prefill compute更低；但FastKV的精度更稳。

## 7. 消融实验

### 7.1 TSP rate

TSP rate太低时，后续层能看到的token太少，精度下降明显。随着TSP rate增大，精度提升，但到20%左右开始趋于饱和。因此论文主实验选20%作为速度和精度的折中。

### 7.2 TSP layer

TSP layer太早会显著掉精度，因为早期层的attention模式不稳定，重要token集合还没收敛。TSP layer太晚虽然更稳，但prefill加速收益变小。

论文认为LLaMA-3.1-8B-Instruct的layer 15是较好的平衡点：LongBench分数基本饱和，同时还有明显prefill加速。

## 8. 局限

FastKV最大的问题是TSP layer仍然是固定的。不同任务需要裁剪的最佳层不一定一样：简单检索任务可能中间层就能确定关键token，但困难KV retrieval任务可能需要更深层才能稳定。如果固定在layer 15，可能对某些任务太早，对另一些任务又偏晚。

这正是ASL论文进一步要解决的问题：**不要固定selection layer，而是在推理时根据attention rank稳定性自适应决定什么时候裁token。**

另一个工程限制是，FastKV仍然需要修改attention执行路径和KV cache管理逻辑。它不是纯prompt侧技巧，而是推理系统级优化。

## 9. 总结

FastKV的贡献不是简单提出一个新的top-k token selection规则，而是提出了一个比较清晰的系统分解：

1. 早期层保留full-context，避免过早丢失信息。
2. 中后层使用Token-Selective Propagation减少prefill计算。
3. decode阶段使用独立的KV retention rate控制cache大小。
4. 将prefill context reduction和decode KV compression解耦，获得更灵活的精度-效率折中。

一句话概括：**FastKV是在固定中间层之后只传播重要token，并且把prefill传播比例和decode KV保留比例拆成两个独立控制项，从而同时优化TTFT、TPOT和显存。**
