---
title: 刷题笔记
category: [笔记]
date: 2024-02-19 01:09
tags: [笔记, leetcode, 算法]
---

本文用以记录在leetcode上的刷题思路，以作复习(重刷/刷类似题)之用

笔记仅记录思路，源码见：https://github.com/GentleCold/leetcode

参考：

- https://programmercarl.com
- https://leetcode.cn

```cpp
#include <bits/stdc++.h>
using namespace std;
```

# 第一章 基础

## 数组

### 704 二分查找(2024-02-19)

> time: O(log n), space: O(1)

Solution:

- left，right --> mid
- compare mid and change left/right

Think:

- 终止条件？边界？

### 27 移除元素(2024-02-19)

> time: O(n), space: O(1)

Solution:

- 暴力双层循环
- 双指针，右指针挑到非target给左指针(全换了)
- 优化：左指针发现target后让右指针用非target替换(只针对target，target少时有帮助)

### 977 有序数组的平方(2024-02-19)

> time: O(n), space: O(n) // store the ans vector

Solution:

- 本质就是归并排序
- 头尾往中间逆序可以减少边界条件判断

### 209 长度最小的子数组(2024-02-19)

> time: O(n), space: O(1)

Solution:

- 滑动窗口，左指针去掉数字，右指针增加数字
