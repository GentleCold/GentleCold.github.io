---
title: CMU10414-Fall2022课程笔记
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

设$m$为样本数，$n$为特征数，$k$为分类数

$h(x)$为假设函数，$h_y(x)$为在$label=y$上的分量

softmax为：

$$
z_i = e^{h_i(x)}/ \sum_{k=1}^n e^{h_k(x)} \\
$$

softmax存在数值爆炸问题，可以对$h_k(x)-max(h(x))$，映射范围为0~1，每个分量和为1

softmax_loss为：

$$
\begin{align*}
l(h, y) &= -log(e^{h_y(x)} / \sum_{k=1}^n e^{h_k(x)}) \\
&= -h_y(x) + log(\sum_{k=1}^n e^{h_k(x)})
\end{align*}
$$

对$h_i(x)$求偏导：

$$
\frac{\partial l(h,y)}{\partial h_i}=-1\{i=y\}+z_i
$$

如果$h(x)=\Theta^T \cdot x$，$\Theta \in R^{n\times k}$，则偏导为：

$$\frac{\partial l(h,y)}{\partial \theta_{i,j}} =(-1\{i=y\}+z_i)\cdot x_{i,j}$$

批量梯度下降，选择$B$个样本然后算出梯度平均，然后对参数立即更新

## neural networks

- 引入非线性层
- 有几个W（权重）就是几层网络

<p align="center">
    <img src="/imgs/image-20250624234051.png"/>
</p>

- 对于一个两层网络的梯度推导：

<p align="center">
    <img src="/imgs/image-20250624234202.png"/>
</p>

<p align="center">
    <img src="/imgs/image-20250624234830.png"/>
</p>

- 对于多层：

<p align="center">
    <img src="/imgs/image-20250625142326.png"/>
</p>

<p align="center">
    <img src="/imgs/image-20250625142722.png"/>
</p>

通过$G_{L+1}$反向传播计算到$G_i$，从而可以算出梯度（需要保留前向传播算出的$Z_i$）

## automatic differentiation

- forward mode AD
<p align="center">
    <img src="/imgs/image-20250625142955.png"/>
</p>

这种方法一次只能计算出$\frac{\partial v_7}{\partial x_1}$，对于$\frac{\partial v_7}{\partial x_2}$还要再传播一次

- reserve mode AD
<p align="center">
    <img src="/imgs/image-20250625143246.png"/>
</p>

使用反向的方法可以一次推导出所需要的所有偏导数

- reserve mode AD by extending computational graph

<p align="center">
    <img src="/imgs/image-20250625143352.png"/>
</p>

$$
\overline{v_{i->j}}=\overline{v_i}\frac{\partial v_i}{\partial v_j}
$$

使用扩展计算图，可以方便计算梯度的梯度

## fully connected networks

- matrix broadcasting（does not copy any data）
