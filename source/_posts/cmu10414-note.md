---
title: CMU15445-Fall2022课程笔记
category: [笔记]
date: 2025-06-19 16:06
tags: [笔记, CMU, 深度学习]
---

# 课程笔记

## softmax

- 监督学习/无监督学习
- 假设函数/损失函数/优化方法
- 有些err函数是不可微分的，所以用softmax(激活函数，引入非线性层)->交叉熵(-log)作为损失函数
- 转换为优化问题，使用梯度下降/随机梯度下降

<p align="center">
    <img src="/imgs/image-20250619232440.png"/>
</p>

<p align="center">
    <img src="/imgs/image-20250619232801.png"/>
</p>

## neural networks

## automatic differentiation

- forward mode AD
- reserve mode AD
- reserve mode AD by extending computational graph
