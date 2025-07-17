---
title: CS336-Spring2025课程笔记
category: [笔记]
date: 2025-06-19 16:06
tags: [笔记, LLM]
---

# 课程笔记

## overview

- prefill: compute-bound / decode: memory bound
- scaling laws:

<p align="center">
    <img src="/imgs/image-20250714194646.png"/>
</p>

- tokenizer: https://tiktokenizer.vercel.app/
- byte pair encoding(BPE)

## resource counting

- float32 / float16 / bfloat16 / fp8
- mixed precision training
- model FLOPs utilization (MFU) (actual FLOP/s / promised FLOP/s)
- 前向传播的浮点数计算是参数量2倍（一次乘法+加法），反向传播是4倍（两次乘法+加法）

## architecture

- pre norm / post norm
- layernorm / rmsnorm(更少的计算量和计算时间)
- relu / swiglu
- parallel layer
- rope
- feedforward ratio(d_ff/d_model)
- softmax stability: zloss / qk norm
- GQA / MQA
- sparse / sliding window attention

## MOE

- router func
- multihead latent attention

## GPU

- sm(streaming multiprocessors) --contain--> sp(streaming processor)
- tpu
- conditionals lead to the overhead
- low precision / operator fusion to minimize memory access / recompute activations / memory coalescing / tiling
- flashattention

## kernels / tritons

# 作业

## 作业一

- 通过utf-8编码将词汇表的0-154997的数值范围转换到0-255，但是会增大序列长度；
- 词级分词器（word-level tokenizers）面临词汇表外（out-of-vocabulary）问题，字节级分词器需要更长的长度，所以使用Subword tokenization（BPE）
