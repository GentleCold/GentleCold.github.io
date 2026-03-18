---
title: CUDA笔记
category: [笔记]
date: 2025-10-09 16:50
tags: [CUDA]
---

Cuda Mode

- 使用load_inline函数可以bind cpp to python
- ncu profiler
- torch -> triton: `TORCH_LOGS = "OUTPUT_CODE" python square_compile.py` with torch.compile

- 修饰符：
<p align="center">
    <img src="/imgs/image-20251010155350.png"/>
</p>

- thread divergence / roofline model
