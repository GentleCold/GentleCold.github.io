---
title: 主流开源大模型参数规模调研
category: [笔记]
date: 2026-05-09 17:38
tags: [LLM, OpenSource, Model]
---

本文重新整理截至 2026-05-09 的主流开源/开放权重大模型参数规模、模型大小、上下文长度、许可证和部署取舍。本文只把能在官方博客、官方 API 文档、官方 Hugging Face 组织或技术报告中核到的模型写进主表；社区传闻、路由商别名和第三方镜像不作为主依据。

## 先说结论

2026 年上半年的开放模型变化很明显：大厂开放模型已经从 7B/70B dense 时代，进入“超大总参数 + 较小 active 参数 + 长上下文 + agent/coding 定向优化”的阶段。

- 小模型：3B 到 14B，适合单卡、本地助手、端侧和低成本批处理。代表是 Ministral 3 3B/8B/14B、Gemma 4 E2B/E4B、Phi-4-mini、Qwen3/Qwen3.6 小模型。
- 中模型：24B 到 40B，很多是 coding/agent 实用甜点位。代表是 Qwen3.6-35B-A3B、Gemma 4 31B、Gemma 4 26B-A4B、Mistral Small/Ministral 方向模型。
- 大 MoE：100B 到 1T+，总参数决定权重加载成本，active 参数决定单 token 计算量。代表是 DeepSeek-V4-Pro、Kimi K2.6、GLM-5.1、Qwen3.5-397B-A17B、Mistral Large 3、Hy3 preview。

一个容易误解的点是：MoE 模型的“总参数”决定权重文件和加载成本，“激活参数”更接近每 token 的计算成本。例如 DeepSeek-V4-Pro 是 1.6T total / 49B active；Kimi K2.6 是 1T total / 32B active；Qwen3.5-397B-A17B 是 397B total / 17B active。

## 开源、开放权重和许可证

严格说，很多模型不是 OSI 意义上的完整开源，因为训练数据、完整训练代码、完整训练配方通常没有全部开放。社区里说“开源大模型”时，常常指的是至少开放了权重，可以下载、自托管、微调或量化。

需要区分几种情况：

- Apache 2.0 / MIT：相对宽松，商业使用友好。当前代表包括 Qwen3.5/3.6、GLM-5.1、Gemma 4、Mistral 3、DeepSeek-V4、gpt-oss。
- 自定义社区许可证：能下载权重，但可能有额外限制，例如 Llama、Kimi、Tencent Hy3 preview。
- 研究/非商用限制：不适合直接商用，需要逐条看 license。

选型时不要只看 benchmark。公司内部部署、SaaS 产品、再分发量化权重、用输出训练另一个模型，都可能触发额外条款。

## 参数规模怎么看

大模型常见指标如下：

- Total Params：总参数量。决定模型权重文件大小，也是加载模型时主要显存/内存压力来源。
- Active Params：MoE 模型每个 token 实际参与前向计算的参数量。Dense 模型没有这个差异，可以近似认为 active = total。
- Context Length：上下文窗口。长上下文会显著增加 KV Cache，不是模型权重大小本身。
- Experts / Activated Experts：MoE 专家数量和每 token 选择的专家数量。
- Precision：BF16/FP16/FP8/NVFP4/MXFP4/INT8/INT4/GGUF/AWQ/GPTQ 等。精度越低，权重越小，但质量和速度不一定线性变好。

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

所以“支持 256K/1M context”不等于你本地可以无代价跑满这个长度。很多本地部署实际只开 8K、16K、32K，是为了吞吐和显存。

## 主流模型参数表

下面优先列目前社区和工程部署中比较常见、且能从官方来源核到参数的开放权重模型。参数可能因 base、instruct、thinking、FP8/INT4/NVFP4 量化版略有差异，表格采用官方模型卡或官方发布页里的主版本数据。

| 系列/模型 | 架构 | 总参数 | 激活参数 | 上下文 | 许可证/开放性 | 主要特点 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| DeepSeek-V4-Pro | MoE + DSA | 1.6T | 49B | 1M | MIT / open weights | DeepSeek 当前 V4 旗舰，agentic coding、推理、长上下文主打 |
| DeepSeek-V4-Flash | MoE + DSA | 284B | 13B | 1M | MIT / open weights | V4 轻量高性价比版本，官方 API 同步可用 |
| Kimi K2.6 | MoE + MLA + multimodal | 1T | 32B | 256K | Modified MIT / open weights | 原生多模态 agentic 模型，支持图像/视频输入，原生 INT4 部署导向 |
| GLM-5.1 | MoE + DSA | 754B | 约 40B | 200K | MIT | Z.ai 最新开放权重主线，长程 agentic engineering 和 coding 导向 |
| Qwen3.5-397B-A17B | Hybrid MoE + Gated DeltaNet + multimodal | 397B | 17B | 262K 原生，托管 Plus 默认 1M | Apache 2.0 | Qwen3.5 旗舰开放权重，图文/视频、多语言和 agent 工具调用 |
| Qwen3.6-35B-A3B | Hybrid MoE + multimodal | 35B | 3B | 262K 原生，可扩展到约 1.01M | Apache 2.0 | Qwen3.6 首个开放权重版本，coding/agent 小 active 参数 |
| Tencent Hy3 preview | MoE | 295B | 21B | 256K | Tencent Hy Community License | 腾讯 Hunyuan 新主线预览，192 experts、top-8 激活 |
| Mistral Large 3 | Sparse MoE + multimodal | 675B | 41B | 长上下文，官方强调 vLLM/NVFP4 部署 | Apache 2.0 | Mistral 3 旗舰开放模型，Base/Instruct 开放 |
| Ministral 3 | Dense + multimodal | 3B/8B/14B | 同总参数 | 取决于版本 | Apache 2.0 | Mistral 3 边缘/本地模型，base/instruct/reasoning 变体 |
| Gemma 4 31B | Dense + multimodal | 30.7B | 30.7B | 256K | Apache 2.0 | Google DeepMind 开放模型，图文输入、thinking 模式 |
| Gemma 4 26B-A4B | MoE + multimodal | 25.2B | 3.8B | 256K | Apache 2.0 | 26B 总参数但接近 4B active，适合中小显存推理 |
| Gemma 4 E2B/E4B | Dense/PLE + multimodal | 5.1B/8B 含 embedding；2.3B/4.5B effective | 约 2B/4B 级 | 128K | Apache 2.0 | 端侧优化，支持文本、图像和音频输入 |
| Llama 4 Scout | MoE + multimodal | 109B | 17B | 10M 标称 | Llama License | 超长上下文，16 experts，多模态输入 |
| Llama 4 Maverick | MoE + multimodal | 400B | 17B | 1M 标称 | Llama License | 128 experts，质量更偏旗舰，激活成本相对低 |
| gpt-oss-120b | MoE | 116.8B | 5.1B | 128K | Apache 2.0 | OpenAI 开放权重推理模型，原生 MXFP4，单 H100 级部署 |
| gpt-oss-20b | MoE | 20.9B | 3.6B | 128K | Apache 2.0 | 面向 16GB 级设备的开放权重推理模型 |
| Phi-4 / Phi-4-mini | Dense | 14B / 3.8B | 同总参数 | 16K 到 128K，依版本 | MIT | 小尺寸数学、代码、结构化推理模型 |

## 按模型家族看

### DeepSeek

DeepSeek-V4 已经替代 V3/R1 成为 DeepSeek 当前最值得关注的开放权重主线。官方 API 文档在 2026-04-24 发布 V4 Preview，并给出两档模型：

- DeepSeek-V4-Pro：1.6T total / 49B active。
- DeepSeek-V4-Flash：284B total / 13B active。
- 两者上下文都是 1M。
- 官方强调 token-wise compression + DSA，也就是 DeepSeek Sparse Attention，用来降低长上下文计算和内存成本。

和 V3 的 671B total / 37B active / 128K 相比，V4-Pro 总容量明显增大，active 参数也更高；V4-Flash 则更像成本可控的日常服务版本。真正部署时，V4-Pro 是多卡/集群对象，V4-Flash 才更接近工程上可高频调用的版本。

### Kimi

Kimi K2.6 是 Moonshot 当前开放权重主线。官方 Hugging Face 模型卡给出的关键配置是：

- 架构：MoE。
- Total Parameters：1T。
- Activated Parameters：32B。
- Layers：61，其中 1 个 dense layer。
- Experts：384 routed experts，每 token 8 个 selected experts，另有 1 个 shared expert。
- Context Length：256K。
- Attention：MLA。
- Vision Encoder：MoonViT。

相比早期 Kimi K2/K2.5，K2.6 的重点已经不只是长上下文，而是原生多模态 agent：图像、视频、长程 coding、coding-driven design、工具调用和多 agent 编排。需要注意的是，1T total 仍然意味着权重加载和存储极重，即便 native INT4 能降低部署门槛，也不是普通单卡模型。

### GLM / Z.ai

GLM 当前官方开放权重主线是 GLM-5.1。官方模型卡和 Z.ai 文档给出的关键信息是：

- GLM-5.1：754B total，社区和部署资料通常标注约 40B active。
- Context Length：200K。
- Maximum Output Tokens：128K。
- Hugging Face 官方模型卡标注模型大小 754B，license 为 MIT。
- 官方文档把 GLM-5.1 定位为 long-horizon task / agentic engineering 模型，可以在单任务上持续执行到 8 小时级别。
- 许可证：MIT。

GLM-5.1 的定位是复杂系统工程、长程 agentic tasks、代码和推理。它比 GLM-4.5-Air 这类 100B 级模型更重，部署上更接近 DeepSeek-V4-Pro、Kimi K2.6、Qwen3.5-397B 这一档。

### Qwen

Qwen 现在有两条很重要的新线：

- Qwen3.5-397B-A17B：397B total / 17B active，默认 262K，上云托管 Plus 版本默认 1M context。
- Qwen3.6-35B-A3B：35B total / 3B active，262K 原生，可用 YaRN 扩展到约 1.01M。

Qwen3.5-397B-A17B 是旗舰开放权重，采用 vision encoder、Gated DeltaNet、sparse MoE 等混合结构，适合多模态、RAG、agent、工具调用和视频/图文理解。Qwen3.6-35B-A3B 则非常工程化：总参数 35B，但 active 只有 3B，官方定位也更偏 coding agent 和真实开发工作流。

如果你想要“能商用、中文好、生态强、尺寸选择多”，Qwen 仍然是第一梯队候选。和旧的 Qwen3-235B-A22B 相比，Qwen3.5-397B-A17B 更像新旗舰；Qwen3.6-35B-A3B 则更适合单机/低成本实验。

### Tencent Hy / Hunyuan

腾讯新的 Hy3 preview 是 Hunyuan 系列里更值得替代 Hunyuan-A13B/Hunyuan-Large 的新主线预览。官方 Hugging Face 模型卡给出的配置是：

- Total Parameters：295B。
- Activated Parameters：21B。
- MTP Layer Parameters：3.8B。
- Layers：80，不含 MTP layer。
- Experts：192 experts，top-8 activated。
- Context Length：256K。
- 许可证：Tencent Hy Community License Agreement。

它的定位是 reasoning、instruction following、context learning、coding 和 agent。和老的 Hunyuan-A13B 80B/13B active 相比，Hy3 preview 更大、更偏旗舰；和 Hunyuan-Large 389B/52B active 相比，它的 active 参数更低，成本结构更友好。

### Mistral

Mistral 3 已经比旧的 Mistral Small 3.2 / Mistral Large 2 更适合作为当前参考。官方发布页给出的核心信息是：

- Mistral Large 3：675B total / 41B active，sparse MoE，多模态，Apache 2.0。
- Ministral 3：3B、8B、14B 三个小 dense 模型，base、instruct、reasoning 变体均开放，Apache 2.0。
- 官方强调和 vLLM、NVIDIA、Red Hat 合作，提供 NVFP4 压缩 checkpoint，并支持单 8xA100/8xH100 节点部署 Large 3。

Mistral 最大变化是许可证和工程部署更友好：Large 3 不再像旧的 Mistral Large 2 那样卡在研究许可证，而是 Apache 2.0。

### Gemma

Gemma 4 已经替代 Gemma 3 成为 Google DeepMind 开放权重主线。官方模型卡给出的配置是：

- Gemma 4 E2B：2.3B effective，5.1B with embeddings，128K。
- Gemma 4 E4B：4.5B effective，8B with embeddings，128K。
- Gemma 4 31B Dense：30.7B，256K。
- Gemma 4 26B-A4B MoE：25.2B total / 3.8B active，256K。
- 许可证：Apache 2.0。

Gemma 4 的优势是小到中尺寸完整覆盖，并且多模态能力更完整。E2B/E4B 面向端侧和移动设备；26B-A4B 是低 active 参数 MoE；31B dense 更适合追求质量且不想处理 MoE serving 复杂度的场景。

### Llama

Llama 4 仍然值得保留在表里，因为生态很强，但要注意许可证不是 Apache/MIT。两个主要开放权重模型是：

- Llama 4 Scout：109B total / 17B active，10M context 标称。
- Llama 4 Maverick：400B total / 17B active，1M context 标称。

Llama 的优势是生态和工具链，缺点是商业使用和再分发需要认真读 Llama License。

### gpt-oss

OpenAI 的 gpt-oss 系列是开放权重 MoE 推理模型：

- gpt-oss-120b：116.8B total / 5.1B active，36 layers、128 experts、top-4 routing。
- gpt-oss-20b：20.9B total / 3.6B active，24 layers、32 experts、top-4 routing。
- 两者都是 Apache 2.0。
- 官方强调 120b 可在单张 80GB GPU 级别运行，20b 面向 16GB 级设备。

它的 active 参数很低，推理成本有吸引力；但质量是否适合具体任务，需要用自己的 eval 验证，不能只看参数量。

## 按显存预算选模型

下面是更工程化的粗略建议，默认使用 4bit/8bit/NVFP4/MXFP4 量化，不考虑跑满超长上下文。实际显存还会受 batch size、KV Cache、推理框架、并行策略影响。

| 机器条件 | 更现实的模型范围 | 推荐候选 |
| --- | --- | --- |
| CPU / 8GB 内存 | 0.5B 到 3B，低上下文 | Gemma 4 E2B、Ministral 3 3B、Phi 小模型 |
| 8GB 显存 | 3B 到 8B 量化 | Gemma 4 E4B、Ministral 3 8B、Phi-4-mini |
| 12GB 到 16GB 显存 | 7B 到 14B，或 20B 极限低 bit | Ministral 3 14B、Phi-4、gpt-oss-20b |
| 24GB 显存 | 24B 到 35B 量化，小 active MoE | Qwen3.6-35B-A3B、Gemma 4 26B-A4B、Gemma 4 31B |
| 48GB 到 80GB 显存 | 35B 到 120B，低精度优先 | Qwen3.6-35B-A3B、gpt-oss-120b、部分 70B/120B 量化模型 |
| 单节点 8x80GB | 200B 到 700B MoE | Qwen3.5-397B-A17B、Hy3 preview、Mistral Large 3、DeepSeek-V4-Flash |
| 多节点/云推理 | 700B 到 1.6T | GLM-5.1、Kimi K2.6、DeepSeek-V4-Pro |

注意：长上下文是显存杀手。如果你真的要 256K/1M context，显存预算要重新计算，不能只按权重文件大小估算。

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

简单说：个人本地优先 dense 或小 active MoE；生产服务如果有多卡和成熟推理栈，可以考虑大 MoE。

## 常见误区

### 误区 1：active 参数小就一定省显存

active 参数影响计算，不等于只加载 active 专家。大多数部署需要加载完整权重，除非你做专家分页、CPU/NVMe offload 或专门的 MoE serving 优化。

### 误区 2：1M context 可以免费使用

上下文越长，prefill 越慢，KV Cache 越大。DeepSeek-V4、Qwen3.5 Plus、Qwen3.6 YaRN 都在讲 1M context，但真实任务中可能 32K/128K 就已经是成本和质量的折中点。

### 误区 3：模型越大越适合所有任务

小模型在分类、抽取、格式化、短文档处理、边缘部署上可能更划算。700B/1T/1.6T 的优势通常出现在复杂推理、长链工具调用、代码代理、多轮规划、困难问答上。

### 误区 4：开源模型都可以随便商用

Apache 2.0/MIT 相对宽松，但 Llama、Kimi、Tencent Hy Community License 等都需要读条款。模型权重、代码、输出、再分发、微调模型发布，可能分别有不同限制。

## 我的选型建议

如果是中文通用助手或 RAG：

- 小成本：Gemma 4 E4B、Ministral 3 8B/14B、Qwen 小模型。
- 单卡质量优先：Qwen3.6-35B-A3B、Gemma 4 31B、Gemma 4 26B-A4B。
- 多卡质量优先：Qwen3.5-397B-A17B、DeepSeek-V4-Flash、Hy3 preview。

如果是代码和 agent：

- 轻量：Qwen3.6-35B-A3B、Ministral 3 14B reasoning、Phi-4。
- 中高端：Qwen3.5-397B-A17B、Mistral Large 3、Hy3 preview。
- 旗舰：Kimi K2.6、GLM-5.1、DeepSeek-V4-Pro。

如果是本地个人机器：

- 8GB 显存：Gemma 4 E4B、Phi-4-mini、Ministral 3 8B。
- 16GB 显存：Ministral 3 14B、Phi-4、gpt-oss-20b。
- 24GB 显存：Qwen3.6-35B-A3B、Gemma 4 26B-A4B、Gemma 4 31B 量化。

如果是研究 MoE serving：

- 小一点的 MoE：Qwen3.6-35B-A3B、Gemma 4 26B-A4B、gpt-oss-20b/120b。
- 中大型 MoE：Hy3 preview、Qwen3.5-397B-A17B、Mistral Large 3。
- 超大型 MoE：Kimi K2.6、GLM-5.1、DeepSeek-V4-Pro。
- 重点看：expert parallelism、routing balance、MTP/speculative decoding、KV Cache、FP8/INT4/NVFP4 kernel、prefill/decode 分离。

## 总结

现在开放模型的竞争已经不只是“参数更大”。真正重要的是：许可证是否可用、上下文是否真实有效、active 参数和总参数的成本差异、推理框架是否支持、量化后质量是否稳定。

最稳妥的选型路径是：

1. 先按许可证筛掉不能用的模型。
2. 再按硬件预算筛掉权重和 KV Cache 放不下的模型。
3. 然后用自己的任务集做 eval，不要只看公开 benchmark。
4. 最后再决定是否微调、蒸馏、量化或上多卡 MoE serving。

对于大多数工程场景，Qwen3.6-35B-A3B、Gemma 4 26B-A4B/31B、Ministral 3、gpt-oss-20b/120b 是比较实际的起点。对于追求开放权重最高质量的服务端场景，再考虑 DeepSeek-V4、Kimi K2.6、GLM-5.1、Qwen3.5-397B-A17B、Mistral Large 3、Hy3 preview 这类大模型。

## 参考

- DeepSeek-V4 Preview 官方发布：https://api-docs.deepseek.com/news/news260424
- DeepSeek-V4 开放权重集合：https://huggingface.co/collections/deepseek-ai/deepseek-v4
- Kimi K2.6 模型卡：https://huggingface.co/moonshotai/Kimi-K2.6
- GLM-5.1 模型卡：https://huggingface.co/zai-org/GLM-5.1
- GLM-5.1 官方文档：https://docs.z.ai/guides/llm/glm-5.1
- Qwen3.5-397B-A17B 模型卡：https://huggingface.co/Qwen/Qwen3.5-397B-A17B
- Qwen3.6-35B-A3B 模型卡：https://huggingface.co/Qwen/Qwen3.6-35B-A3B
- Tencent Hy3 preview 模型卡：https://huggingface.co/tencent/Hy3-preview
- Mistral 3 官方发布：https://mistral.ai/news/mistral-3
- Gemma 4 31B Instruct 模型卡：https://huggingface.co/google/gemma-4-31B-it
- Meta llama-models 仓库：https://github.com/meta-llama/llama-models
- gpt-oss 发布说明：https://openai.com/index/introducing-gpt-oss/
- gpt-oss 模型卡：https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf
- Phi-4-reasoning 技术报告：https://arxiv.org/abs/2504.21318
- Phi-4-mini 技术报告：https://arxiv.org/abs/2503.01743
