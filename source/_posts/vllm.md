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
