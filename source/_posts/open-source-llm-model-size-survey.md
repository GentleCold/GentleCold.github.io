---
title: 主流开源大模型参数规模调研
category: [笔记]
date: 2026-05-09 16:49
tags: [LLM, OpenSource, Model]
---

本文整理截至 2026-05-09 的主流开源/开放权重大模型参数规模、模型大小、上下文长度、许可证和部署取舍。因为大模型发布节奏很快，下面的表格更适合作为“选型地图”，真正部署前还需要回到模型卡确认最新 checkpoint、量化版本和推理框架支持情况。

## 先说结论

现在的开放模型大致分成三类：

- 小模型：3B 到 14B，适合单卡、边缘设备、本地助手、低成本批处理。代表是 Phi-4-mini、Gemma 3 4B/12B、Qwen3 8B/14B。
- 中等模型：24B 到 80B，质量明显上一个台阶，很多能在 24GB 到 80GB 显存范围内通过 4bit/8bit 跑起来。代表是 Mistral Small 3.2 24B、Qwen3 32B、Qwen3-Next-80B-A3B、Hunyuan-A13B。
- 大模型/MoE：100B 到 1T+，磁盘和内存压力很大，但每 token 激活参数可能只有几 B 到几十 B。代表是 DeepSeek-V3/R1、Qwen3-235B-A22B、GLM-4.5、Kimi K2、Llama 4 Maverick、gpt-oss-120b。

一个容易误解的点是：MoE 模型的“总参数”决定权重加载和存储成本，“激活参数”更接近每 token 的计算成本。例如 Qwen3-235B-A22B 需要加载约 235B 参数，但每 token 只激活约 22B；gpt-oss-120b 总参数约 116.8B，但每 token 激活约 5.1B。

## 开源、开放权重和许可证

严格说，很多模型不是 OSI 意义上的“完整开源”，因为训练数据、训练代码、完整训练配方通常没有全部开放。日常社区里说“开源大模型”时，常常指的是至少开放了权重，可以下载、自托管、微调或量化。

需要区分几种情况：

- Apache 2.0 / MIT：相对宽松，商业使用友好，例如 Qwen3、GLM-4.5、Hunyuan-A13B、gpt-oss。
- 自定义开放模型许可证：能下载权重，但可能有额外限制，例如 Llama 系列、Gemma、Mistral Large 2。
- 研究/非商用限制：不适合直接商用，需要逐条看 license。

选型时不要只看 benchmark，许可证是第一道门槛。尤其是公司内部部署、SaaS 产品、再分发量化权重、用输出训练另一个模型，都可能触发额外条款。

## 参数规模怎么看

大模型常见指标如下：

- Total Params：总参数量。决定模型权重文件大小，也是加载模型时主要显存/内存压力来源。
- Active Params：MoE 模型每个 token 实际参与前向计算的参数量。Dense 模型没有这个差异，可以近似认为 active = total。
- Context Length：上下文窗口。长上下文会显著增加 KV Cache，不是模型权重大小本身。
- Experts / Activated Experts：MoE 专家数量和每 token 选择的专家数量。
- Precision：BF16/FP16/FP8/INT8/INT4/GGUF/AWQ/GPTQ 等。精度越低，权重越小，但质量和速度不一定线性变好。

权重大小可以粗略估算：

$$
\mathrm{WeightMemory} \approx \mathrm{Params} \times \frac{\mathrm{bits}}{8}
$$

例如 70B dense 模型：

- BF16/FP16：约 $70B \times 2 = 140GB$，再加框架开销和 KV Cache。
- INT8：约 70GB。
- INT4：约 35GB，实际 GGUF/AWQ 文件会因为 scale、metadata、embedding 等略有差异。

KV Cache 不能忽略。长上下文下，KV Cache 可能比量化后的权重还大。它大致和层数、KV head 数、head dim、上下文长度、batch size、精度成正比：

$$
\mathrm{KVCache} \propto L \times T \times H_{kv} \times D \times \mathrm{bytes}
$$

所以“支持 128K/256K/1M context”不等于你本地可以无代价跑满这个长度。很多本地部署实际只开 8K、16K、32K，是为了吞吐和显存。

## 主流模型参数表

下面优先列目前社区和工程部署中比较常见的开放权重模型。参数可能因 base、instruct、thinking、FP8、量化版略有差异，表格采用官方模型卡或技术报告里的主版本数据。

| 系列/模型 | 架构 | 总参数 | 激活参数 | 上下文 | 许可证/开放性 | 主要特点 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Qwen3 Dense | Dense | 0.6B/1.7B/4B/8B/14B/32B | 同总参数 | 通常 32K 起，部分可扩展 | Apache 2.0 | 中文、代码、多语言和工具调用都均衡，尺寸覆盖完整 |
| Qwen3-30B-A3B | MoE | 30B | 3B | 32K 起 | Apache 2.0 | 低激活 MoE，适合单机中小显存尝试 |
| Qwen3-235B-A22B | MoE | 235B | 22B | 32K 原生，YaRN 到 131K | Apache 2.0 | Qwen3 旗舰 MoE，128 experts、8 active experts |
| Qwen3-Next-80B-A3B | Hybrid MoE | 80B | 3B | 262K 原生，部分说明可扩展到约 1M | Apache 2.0 | 极低激活参数，长上下文和推理吞吐友好 |
| DeepSeek-V3 / V3.1 | MoE | 671B | 37B | 128K 左右 | DeepSeek License / open weights | 大规模 MoE，代码、数学、中文和英文都强 |
| DeepSeek-R1 | MoE reasoning | 671B | 37B | 128K 左右 | open weights | 强推理模型，另有大量 distill 版本 |
| Llama 3.1 405B | Dense | 405B | 405B | 128K | Llama License | 大型 dense 开放权重模型，部署成本高 |
| Llama 3.3 70B | Dense | 70B | 70B | 128K | Llama License | 70B 级别经典通用模型，生态成熟 |
| Llama 4 Scout | MoE + multimodal | 109B | 17B | 10M 标称 | Llama License | 超长上下文，16 experts，多模态输入 |
| Llama 4 Maverick | MoE + multimodal | 400B | 17B | 1M 标称 | Llama License | 128 experts，质量更偏旗舰，激活成本相对低 |
| GLM-4.5 | MoE | 355B | 32B | 128K | MIT | agent、代码、推理三合一，开放 base、hybrid reasoning、FP8 版本 |
| GLM-4.5-Air | MoE | 106B | 12B | 128K | MIT | 更易部署的 GLM-4.5 轻量版 |
| Kimi K2 | MoE | 1T | 32B | 128K 到 256K，取决于版本 | Modified MIT / open weights | agentic/coding 强，1T 总参数但每 token 32B active |
| MiniMax-M1 | MoE + hybrid attention | 456B | 45.9B | 1M | open weights | Lightning Attention，长上下文和长推理导向 |
| Hunyuan-A13B | Fine-grained MoE | 80B | 13B | 256K | 开放权重，GitHub license | 腾讯混元，双模式推理，80B 总参数但 13B active |
| Hunyuan-Large | MoE | 389B | 52B | 256K | open weights | 更大的混元 MoE，激活参数较高 |
| Mistral Small 3.2 | Dense + vision | 24B | 24B | 128K | Apache 2.0 | 24B 档实用模型，函数调用和视觉输入友好 |
| Mistral Large 2 | Dense | 123B | 123B | 128K | Mistral Research License | 开放权重但非 Apache/MIT，单节点大模型定位 |
| Mixtral 8x22B | MoE | 141B 左右 | 39B 左右 | 64K | Apache 2.0 | 经典 MoE，8 experts、每 token 2 experts |
| Gemma 3 | Dense + vision | 1B/4B/12B/27B | 同总参数 | 1B 为 32K；4B/12B/27B 为 128K | Gemma Terms | Google 开放权重，多模态，小到中尺寸覆盖好 |
| Gemma 3n | Dense/移动优化 | E2B/E4B 有效规模 | 约 2B/4B 级 | 移动端导向 | Gemma Terms | 面向端侧，参数跳过和 PLE cache 降低有效内存 |
| Phi-4 | Dense | 14B | 14B | 16K，reasoning 训练常见 32K | MIT | 小尺寸推理、数学、代码强 |
| Phi-4-mini | Dense | 3.8B | 3.8B | 128K | MIT | 端侧/低成本模型，长上下文，小显存友好 |
| gpt-oss-120b | MoE | 116.8B | 5.1B | 128K | Apache 2.0 | OpenAI 开放权重推理模型，原生 MXFP4，单 H100 级部署 |
| gpt-oss-20b | MoE | 20.9B | 3.6B | 128K | Apache 2.0 | 面向 16GB 级设备的开放权重推理模型 |

## 按模型家族看

### Qwen

Qwen 现在是开放模型里最完整的家族之一。它的优势不是单个模型，而是尺寸、任务类型和许可证都比较齐：

- Dense：0.6B、1.7B、4B、8B、14B、32B，适合从本地轻量助手到单机服务。
- MoE：30B-A3B、235B-A22B，分别覆盖低成本 MoE 和旗舰 MoE。
- Next：80B-A3B 这种极低 active 参数模型，重点是长上下文和推理效率。
- Coder/VL/Omni 等变体也多，生态里 vLLM、SGLang、Transformers、llama.cpp、Ollama 支持都比较积极。

如果你想要“能商用、中文好、生态强、尺寸选择多”，Qwen 通常是第一梯队候选。235B-A22B 的质量强，但加载成本仍然是 235B 级；80B-A3B 则更偏工程效率。

### DeepSeek

DeepSeek-V3/R1 的特点是大规模 MoE 和强推理能力。V3 系列偏通用、代码、数学和中文英文混合任务；R1 是 reasoning 模型，社区里也有大量蒸馏版本，例如 Distill-Qwen、Distill-Llama。

DeepSeek 的关键数字是 671B total、37B active。也就是说它的每 token 计算量不是 671B dense 那么夸张，但权重加载、通信、量化和多卡切分仍然很重。部署全量模型通常要多卡服务器；个人设备更现实的是跑 distill 或较激进量化。

### Llama

Llama 的优势是生态和工具链。Llama 3.1/3.3 时代，70B 和 405B 是典型 dense 模型；Llama 4 开始转向 MoE，并引入原生多模态和超长上下文。

- Llama 3.3 70B：生态成熟，很多微调、量化和工具链围绕它优化。
- Llama 3.1 405B：dense 巨型模型，质量强但部署成本高。
- Llama 4 Scout：109B total、17B active、10M context 标称，长上下文是最大卖点。
- Llama 4 Maverick：400B total、17B active、1M context，质量更偏旗舰。

实际工程上，Llama License 需要单独审查，尤其是大规模商业产品和用模型输出训练竞争模型的场景。

### GLM / Z.ai

GLM-4.5 系列很适合关注 agent、代码和推理任务的人。官方模型卡给出的核心配置是：

- GLM-4.5：355B total、32B active。
- GLM-4.5-Air：106B total、12B active。
- 上下文：128K。
- 许可证：MIT。

它的定位很清楚：用 MoE 把总容量拉大，同时把每 token 计算量压到 12B/32B 级别。GLM-4.5-Air 尤其适合作为“中大模型但不想上 200B+ 权重”的选项。

### Kimi K2

Kimi K2 是 1T 总参数 MoE，激活参数约 32B。这个结构的意义是：模型容量非常大，但每 token 只走一小部分专家。它的强项偏 agentic workflow、代码、工具调用和长任务。

需要注意的是，1T total 仍然意味着权重加载和存储极重，即便 INT4 也不是普通单卡轻松处理的量级。它更像是多卡服务器、云端推理或高度优化引擎的对象。

### Hunyuan

腾讯混元开放模型里，Hunyuan-A13B 很值得关注：

- 80B total、13B active。
- 256K context。
- fine-grained MoE。
- 支持快/慢两种推理模式。

它的工程定位和 GLM-4.5-Air、Qwen3-Next 有些相似：总参数不小，但 active 参数控制在中等范围，希望在质量和成本之间取得更好平衡。

### Mistral

Mistral 早期 Mixtral 系列是开放 MoE 的代表。Mixtral 8x22B 大约 141B total、39B active、64K context，是一个经典但部署成本并不低的 MoE。

Mistral Small 3.2 24B 更适合实际使用：24B dense、128K context、Apache 2.0，并支持多模态输入。它在 24B 档很有吸引力，尤其适合不想承担 70B 以上成本的服务。

Mistral Large 2 是 123B dense、128K context，开放权重但许可证是 Mistral Research License，不等同于 Apache/MIT。部署前需要看清商用条款。

### Gemma

Gemma 3 是 Google 的开放权重轻中量级模型，尺寸是 1B、4B、12B、27B。4B/12B/27B 支持 128K 输入上下文，1B 支持 32K；4B 以上还支持图文输入、文本输出。

Gemma 的优势在端侧和单卡：4B、12B、27B 的梯度很自然。缺点是 Gemma Terms 不是 Apache/MIT，商用和再分发也需要看条款。

### Phi

Phi 系列的核心是“小但强”，尤其偏数学、代码和结构化推理：

- Phi-4：14B dense，常见上下文 16K，reasoning 相关版本训练上下文可到 32K。
- Phi-4-mini：3.8B dense，128K context。

Phi-4-mini 很适合本地助手、低成本文档处理、边缘设备和教学实验。它不会替代 32B/70B 通用模型，但在 4B 档非常实用。

### gpt-oss

OpenAI 的 gpt-oss 系列是开放权重 MoE 推理模型：

- gpt-oss-120b：116.8B total、5.1B active、36 layers、128 experts、top-4 routing。
- gpt-oss-20b：20.9B total、3.6B active、24 layers、32 experts、top-4 routing。
- 两者都是 Apache 2.0。
- 官方强调 120b 可在单张 80GB GPU 级别运行，20b 面向 16GB 级设备。

它的 active 参数很低，推理成本有吸引力；但质量是否适合具体任务，需要用自己的 eval 验证，不能只看参数量。

## 按显存预算选模型

下面是更工程化的粗略建议，默认使用 4bit/8bit 量化，不考虑跑满超长上下文。实际显存还会受 batch size、KV Cache、推理框架、并行策略影响。

| 机器条件 | 更现实的模型范围 | 推荐候选 |
| --- | --- | --- |
| CPU / 8GB 内存 | 0.5B 到 3B，低上下文 | Qwen3 0.6B/1.7B、Gemma 3 1B、Phi 小模型 |
| 8GB 显存 | 3B 到 8B 量化 | Qwen3 4B/8B、Gemma 3 4B、Phi-4-mini |
| 12GB 到 16GB 显存 | 7B 到 14B 量化，或 20B 极限低 bit | Qwen3 8B/14B、Gemma 3 12B、Phi-4、gpt-oss-20b |
| 24GB 显存 | 14B 到 32B 量化，部分 MoE offload | Qwen3 32B、Mistral Small 24B、Gemma 3 27B、Qwen3-30B-A3B |
| 48GB 到 80GB 显存 | 32B 到 80B，或 120B 低精度 | Qwen3-Next-80B-A3B、Llama 70B、gpt-oss-120b、Mistral Large 2 量化 |
| 多卡 80GB | 100B+ 或 200B+ MoE | Qwen3-235B-A22B、GLM-4.5、DeepSeek-V3/R1、Llama 4 Maverick |
| 集群/云推理 | 400B 到 1T+ | DeepSeek、Kimi K2、MiniMax-M1、Hunyuan-Large |

注意：长上下文是显存杀手。如果你真的要 128K/256K/1M context，显存预算要重新计算，不能只按权重文件大小估算。

## Dense 和 MoE 怎么选

Dense 模型的优点：

- 实现简单，推理框架支持成熟。
- 小 batch 和本地部署更稳定。
- 量化、LoRA、合并权重、KV Cache 优化都更直接。

Dense 模型的缺点：

- 参数越大，每 token 计算量越线性增长。
- 70B 以上对单机部署压力很大。

MoE 模型的优点：

- 总容量大，active 参数小，质量/计算成本比可能更好。
- 更适合大规模服务和多卡并行。
- 对代码、推理、工具调用等复杂任务，旗舰 MoE 往往很强。

MoE 模型的缺点：

- 权重仍然要加载，总参数决定显存/内存下限。
- 专家路由带来负载均衡、通信和 kernel 支持问题。
- 本地 CPU/GPU offload 容易出现“能跑但很慢”。
- 量化后质量波动更难预测，需要实测。

简单说：个人本地优先 dense 或小 MoE；生产服务如果有多卡和成熟推理栈，可以考虑大 MoE。

## 常见误区

### 误区 1：active 参数小就一定省显存

active 参数影响计算，不等于只加载 active 专家。大多数部署需要加载完整权重，除非你做专家分页、CPU/NVMe offload 或专门的 MoE serving 优化。

### 误区 2：128K context 可以免费使用

上下文越长，prefill 越慢，KV Cache 越大。很多模型卡写 128K、256K、1M，但真实任务中可能 32K 就已经是成本和质量的折中点。

### 误区 3：模型越大越适合所有任务

小模型在分类、抽取、格式化、短文档处理、边缘部署上可能更划算。70B/235B/671B 的优势通常出现在复杂推理、长链工具调用、代码代理、多轮规划、困难问答上。

### 误区 4：开源模型都可以随便商用

Apache 2.0/MIT 相对宽松，但 Llama、Gemma、Mistral Research License、Modified MIT 等都需要读条款。模型权重、代码、输出、再分发、微调模型发布，可能分别有不同限制。

## 我的选型建议

如果是中文通用助手或 RAG：

- 小成本：Qwen3 8B/14B、Gemma 3 12B。
- 单卡质量优先：Qwen3 32B、Mistral Small 3.2 24B。
- 多卡质量优先：Qwen3-235B-A22B、DeepSeek-V3、GLM-4.5。

如果是代码和 agent：

- 轻量：Qwen3-Coder 系列、Phi-4、Phi-4-mini。
- 中等：Qwen3-Next-80B-A3B、GLM-4.5-Air、Hunyuan-A13B。
- 高质量：Kimi K2、DeepSeek-R1/V3、GLM-4.5、Qwen3-235B-A22B。

如果是本地个人机器：

- 8GB 显存：Gemma 3 4B、Phi-4-mini、Qwen3 4B/8B。
- 16GB 显存：Qwen3 14B、Gemma 3 12B、Phi-4、gpt-oss-20b。
- 24GB 显存：Qwen3 32B、Mistral Small 24B、Gemma 3 27B、Qwen3-30B-A3B。

如果是研究 MoE serving：

- 小一点的 MoE：Qwen3-30B-A3B、Hunyuan-A13B、Qwen3-Next-80B-A3B。
- 大 MoE：Qwen3-235B-A22B、DeepSeek-V3/R1、GLM-4.5、Kimi K2。
- 重点看：expert parallelism、routing balance、KV Cache、FP8/INT4 kernel、prefill/decode 分离。

## 总结

现在开放模型的竞争已经不只是“参数更大”。真正重要的是：许可证是否可用、上下文是否真实有效、active 参数和总参数的成本差异、推理框架是否支持、量化后质量是否稳定。

最稳妥的选型路径是：

1. 先按许可证筛掉不能用的模型。
2. 再按硬件预算筛掉权重和 KV Cache 放不下的模型。
3. 然后用自己的任务集做 eval，不要只看公开 benchmark。
4. 最后再决定是否微调、蒸馏、量化或上多卡 MoE serving。

对于大多数工程场景，Qwen3 14B/32B、Mistral Small 24B、Gemma 3 12B/27B、Phi-4-mini、Qwen3-Next-80B-A3B 是比较实际的起点。对于追求开放权重最高质量的服务端场景，再考虑 DeepSeek、Qwen3-235B、GLM-4.5、Kimi K2、Llama 4 Maverick 这类大模型。

## 参考

- Qwen3-235B-A22B 模型卡：https://huggingface.co/Qwen/Qwen3-235B-A22B
- Qwen3-Next-80B-A3B-Instruct 模型卡：https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct
- DeepSeek-V3 技术报告：https://arxiv.org/abs/2412.19437
- DeepSeek-V3.1 模型页：https://huggingface.co/deepseek-ai/DeepSeek-V3.1
- Meta llama-models 仓库：https://github.com/meta-llama/llama-models
- Llama 3 技术报告：https://arxiv.org/abs/2407.21783
- GLM-4.5 模型卡：https://huggingface.co/zai-org/GLM-4.5
- GLM-4.5 技术报告：https://arxiv.org/abs/2508.06471
- Kimi K2 技术报告：https://arxiv.org/abs/2507.20534
- Hunyuan-A13B 仓库：https://github.com/Tencent-Hunyuan/Hunyuan-A13B
- Hunyuan-Large 技术报告：https://arxiv.org/abs/2411.02265
- MiniMax-M1 技术报告：https://arxiv.org/abs/2506.13585
- Mistral Large 2 模型卡：https://docs.mistral.ai/models/model-cards/mistral-large-2-0-24-07
- Gemma 3 模型卡：https://huggingface.co/google/gemma-3-27b-it
- Gemma 3n 文档：https://ai.google.dev/gemma/docs/gemma-3n
- Phi-4-reasoning 技术报告：https://arxiv.org/abs/2504.21318
- Phi-4-mini 技术报告：https://arxiv.org/abs/2503.01743
- gpt-oss 发布说明：https://openai.com/index/introducing-gpt-oss/
- gpt-oss 模型卡：https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf
