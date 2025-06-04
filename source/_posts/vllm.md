---
title: VLLM与大模型推理框架
category: [笔记]
date: 2025-05-21 19:38
tags: [VLLM]
---

## VLLM

VLLM v1 代码整体流程，代码版本v0.8.5

<p align="center">
    <img src="/imgs/image-20250521194042.png"/>
</p>

调度部分，先调度running队列，再调度waiting队列。

https://zhuanlan.zhihu.com/p/1908153627639551302

<p align="center">
    <img src="/imgs/image-20250521201303.png"/>
</p>

关于抢占，抢占只是释放block不再进行运算，实际等到根据LRU策略去替换block时才会真正抢占。

## KV Cache

当前Q乘缓存的K，再乘缓存的V，得到第三行的Attention

<p align="center">
    <img src="/imgs/image-20250521194442.png"/>
</p>

## 调度优化

Continuous batching: https://mp.weixin.qq.com/s/Se4lzaTLNZF29BXLRjw0xw?poc_token=HC64LWijE2pWprmvhuJr1WCCkTO0yYe_G-WfBe6d

VLLM调度策略修改：https://github.com/vllm-project/vllm/issues/16969

VLLM V1调度添加优先级：https://github.com/vllm-project/vllm/issues/14002

## AlayaDB: The Data Foundation for Efficient and Effective Long-context LLM Inference

核心在于将注意力计算和缓存管理抽象为一种查询处理流程，并通过本地查询优化器提升性能。

从总体上看，AlayaDB在LLM推理中的角色类似于传统数据库在Web应用中的作用。具体而言，LLM应用开发者只需关注应用逻辑，而AlayaDB则提供高效的长上下文管理能力，支持开发完成的LLM应用。这类似于Web开发者专注于应用逻辑，将高效的数据管理交给传统关系型数据库。

长上下文计算成本高，现有结构：

- Coupled Architecture: vllm, sglang 大量GPU内存用于存储KV缓存，以粗略的方式重用KV缓存
- KVCache Disaggregation: LMCache, MoonCake 将KV缓存拆分到外部设备，需要对引擎进行大量侵入式修改
- Retrieval-based Sparse Attention: InfLLM, Retrieval Attention 从分离的KV缓存中检索部分缓存使用，在内存消耗、推理延迟和生成质量之间进行权衡

AlayaDB将KV缓存和稀疏注意力计算与LLM推理引擎解耦，提供了：

- 用户界面
- 查询处理引擎
- 向量存储引擎

动态内积范围查询

相比于top-k，固定的k无法满足关键token数量不同的场景，使用DIPR去按比例搜寻

查询优化

- 粗粒度索引，将token向量分块并使用代表向量
- 细粒度索引，利用图索引
- 扁平索引，将keys连续存储然后顺序扫描一遍

然后使用基于规则的优化，根据prompt的上下文长度、GPU显存大小来决定使用哪种方式

## CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion

sparse attention类似于让Q只和部分K做计算，例如RetrievalAttention通过对KVCache做检索来做到这一点，来减少QKV点乘的开销

而CacheBlend则扩展了对KVCache的复用机制，对于相同的Token提高KVCache复用率，在第一层比对然后只重算差距大的KVCache，来减少KV计算的开销

有没有可能结合这两点

<p align="center">
    <img src="/imgs/image-20250604224204.png"/>
</p>

对于大规模数据处理任务，实际这种类型感觉和RAG检索很像，多了不需要上下文关联的prompt类型：

<p align="center">
    <img src="/imgs/image-20250605214722.png"/>
</p>
