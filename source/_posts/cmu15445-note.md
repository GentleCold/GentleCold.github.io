---
title: CMU15445-Fall2023课程笔记
category: [笔记]
date: 2024-02-22 17:37
tags: [笔记, CMU15445, 数据库]
---

# 课程笔记

- algebra
- vector database for ai
- relational languages
  - dml/ddl/dcl

# 项目思路

## 项目准备

- 判分：

[gradescope](https://www.gradescope.com): KK5DVJ

- 讨论：

https://discord.gg/YF7dMCg

- 使用clang：

```cmake
set(CMAKE_C_COMPILER "/usr/bin/clang")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
```

nmd这玩意儿要写在project后面谁跟我说要写在前面的？？

## P0. C++ Primer

### Think basic

- what is a trie? https://zhuanlan.zhihu.com/p/28891541
- what is copy on write? 即修改的时候再复制而不是直接复制：https://zh.wikipedia.org/zh-cn/%E5%AF%AB%E5%85%A5%E6%99%82%E8%A4%87%E8%A3%BD
- why copy on write? 每次修改都会产生一个新的trie，如果不能影响到原来的trie，需要尽量复用节点
- 智能指针可通过get获取原始指针，可用dynamic_cast判断派生类指针

### Task1

- Get: 找到后用dynamic_cast把指针改成指向TrieNodeWithValue
- Put: 所有经过的节点都需要克隆
- 因为Put是一个const函数，所以tmd不能修改root!
- nmd所有sharedptr都tm是const TrieNode，只有uniqueptr是non-const的，That's why the return of Clone is unique!
- 所以本质就是non-const unique->const shared，因为shared是const，所以只能从后往前构建，那就用栈！
- why use unique? 如果是non-const shared，把它改了那所有的引用都改了
- 还要注意空字符串/root\_初始为nullptr!
- wtmd，map的insert方法不会自动替换相同的键的值！要么用[]要么用insert_or_assign!!

### Task2
