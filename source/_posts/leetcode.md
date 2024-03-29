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

- left，right --> mid
- compare mid and change left/right
- 终止条件？边界？

### 27 移除元素(2024-02-19)

> time: O(n), space: O(1)

- 暴力双层循环
- 双指针，右指针挑到非target给左指针(全换了)
- 优化：左指针发现target后让右指针用非target替换(只针对target，target少时有帮助)

### 977 有序数组的平方(2024-02-19)

> time: O(n), space: O(n) // store the ans vector

- 本质就是归并排序
- 头尾往中间逆序可以减少边界条件判断

### 209 长度最小的子数组(2024-02-19)

> time: O(n), space: O(1)

- 滑动窗口，左指针去掉数字，右指针增加数字

### 59 螺旋矩阵II(2024-02-19)

> time: O(n^2), space: O(1)

- 模拟法，类似贪吃蛇的移动

## 链表

### 203 移除链表元素(2024-02-20)

> time: O(n), space: O(1)

- 注意内存管理和head条件即可
- 可以用dummyhead来优化head的条件判断

### 707 设计链表(2024-02-20)

- 自己实现一个单向链表/双向链表
- 谨防未定义行为

### 206 反转链表(2024-02-21)

> time: O(n), space: O(1)

- 递归/迭代两种方式（迭代更好写

### 24 两两交换链表中的节点(2024-02-21)

> time: O(n), space: O(1)

- dummyhead大法好

### 19 删除链表的倒数第N个结点(2024-02-21)

- 考虑一次遍历，栈/双指针法，后者的空间复杂度更低

### 160 相交链表(2024-02-21)

- 哈希记录扫描过的节点
- 双指针遍历(有点类似倒水问题，关键在于长度差距)

### 142 环形链表II(2024-02-21)

- 同样使用哈希记录节点
- 如果要O(1)空间，用快慢指针(一个走一步，一个走两步，如果有环肯定相遇，并且可以根据相遇处推出环入口)

## 哈希表

- 若限制了数据大小且大小不大时即可用数组作答
- 需要判断是否出现过、找数字等，用哈希记录

### 242 有效的字母异位词(2024-02-21)

### 349 两个数组的交集(2024-02-21)

### 202 快乐数(2024-02-22)

> time: O(logn), space: O(1)

- 哈希记录找到循环。还要考虑无穷大的情况，不过这种不会发生
- 既然是查环结构，那自然可以用快慢指针(参见142)

### 1 两数之和(2024-02-22)

- 暴力枚举，无法双指针因为排序会打乱下标
- 如何降到线性复杂度？考虑如何快速获取需要的数
- 一次遍历就行了，注意同一个数字不能重复出现

### 454 四数相加II(2024-02-22)

- 根据之前的经验，当然是哈希，如果只哈希一组数据为O(n^3)，空间为O(n)
- 分组可以再减少一波复杂度O(n^2+n^2)，空间为O(n^2)

### 15 三数之和(2024-02-22)

- 不用哈希，因为去重麻烦
- 排序+双指针相对更高效，确定一个位置，再双指针确定，同时也要注意去重，三个指针的去重！

### 18 四数之和(2024-02-22)

- 类似于三数之和，上升一个时间复杂度，为O(n^3)
- 注意下溢出，用long解决

## 字符串

### 344 反转字符串(2024-02-25)

- use swap

### 151 反转字符串中的单词(2024-02-25)

- 需要O(1)空间
- 去除多余空格（双指针）->反转所有单词->反转单个单词
- 别随便在循环条件里用j++!

### 28 找出字符串中第一个匹配项的下标(2024-02-25)

> time: O(n+m), space: O(m)

- KMP算法！
- 前缀/后缀/next数组！
- Great answer! https://www.zhihu.com/question/21923021/answer/281346746
- 注意j跑到0时的情况，i和j都增加(注意j为-1或j为0，前者即不匹配的情况)

### 459 重复的子字符串(2024-02-26)

- 暴力枚举子串(n^2)
- s+s再匹配s

## 栈与队列

- 在SGI STL中，栈和队列并不算容器，被归类为container adapter，底层使用deque，不提供迭代器，在内存中不连续分布

### 232. 用栈实现队列(2024-02-28)

- 双栈，倒来倒去（push不倒，pop倒出来）

### 225. 用队列实现栈(2024-02-28)

- 单队列，队列加到尾部

### 1047. 删除字符串中的所有相邻重复项(2024-02-28)

- 用栈的消消乐

### 150. 逆波兰表达式求值(2024-02-29)

- 后缀表达式：遇到数字就入栈，遇到运算符就取出两个数字运算
- 判断是否为数字时注意负数（也可以直接判断是否为运算符

### 239. 滑动窗口最大值(2024-02-29)

- 优先队列，时间为O(nlogn)，同时记录下标，将所有不在滑动窗口中的最大值弹出
- 单调队列，即队列中元素是单调的，线性复杂度(每个元素进入和弹出各一次)

### 347. 前K个高频元素(2024-02-29)

- 哈希记录次数+优先队列，O(nlogk) (可以用快排优化)

## 二叉树

- 满二叉树/完全二叉树/搜索二叉树/平衡二叉树
- 前序/中序/后序/层序遍历(递归/迭代)
- 树的度/高度/深度

### 102. 二叉树的层序遍历(2024-03-01)

- 队列（广度优先搜索
- 固定size

### 226. 翻转二叉树(2024-03-01)

- 前序/后序反转

### 101. 对称二叉树(2024-03-01)

- 递归比较，注意是轴对称

### 222. 完全二叉树的节点数(2024-03-01)

- 利用完全二叉树的性质
- 通过深度判断节点个数
- 通过二分查找和位运算判断（查找底层节点个数）

### 110. 平衡二叉树(2024-03-02)

- 注意要求是每个节点高度差都不超过1
- 使用自低向上的递归（后序）可以降低复杂度，返回-1表示false

### 257. 二叉树的所有路径(2024-03-03)

- 遍历+回溯

### 404. 左叶子之和(2024-03-05)

### 513. 找树左下角的值(2024-03-05)

### 112. 路径总和(2024-03-05)

- 注意只有到叶子节点才能算(没有孩子)

### 106. 从中序与后序遍历序列构造二叉树(2024-03-06)

- 中序的顺序是左/中/右
- 后序的顺序是左/右/中
- 我们根据后序遍历来切割中序遍历
- 后序遍历也要跟着切割
- 后序数组切割根据中序的大小切，因为两者的长度肯定一样
- 后序数组也需要记录开始和结束的位置

### 654. 最大二叉树(2024-03-06)

## 回溯算法

### 77. 组合(2024-03-07)

- 回溯！push了再pop
- 减枝！如果之后的数字不够就结束循环

### 216. 组合总和III(2024-03-09)

- 通过降序来去重
- 循环也可以剪枝（通过剩余选择数

### 17. 电话号码的字母组合(2024-03-14)

### 39. 组合总和(2024-03-14)

### 40. 组合总和II(2024-03-14)

- 要求去重，经过排序即可
- 如果是重复数字就跳了(放后面)

### 131. 分割回文串(2024-03-14)

- 回溯递归分割点
- 动态规划记录是否为回文

### 93. 复原IP地址(2024-03-15)

### 78. 子集(2024-03-15）

- 组合和分割问题是找叶子节点，子集问题是找所有节点

### 90. 子集II(2024-03-15)

- 排序去重

### 491. 非递减子序列(2024-03-18)

- 使用back获取vector最后的元素时要注意是否为空
- 注意[1,2,1,1]的情况

### 46. 全排列(2024-03-18)

- 数组位置记录优于哈希记录

### 47. 全排列II(2024-03-18)

- 排序去重/哈希表

### 332. 重新安排行程(2024-03-24)

- 构建哈希表，记录机场间行程（图的边）
- 回溯构造行程
- 按照字典序排: map/priority queue
- 终止条件为结果长度达到边的数量+1

### 51. N皇后(2024-03-25)

- 直接记录皇后的位置
- 只需要记录列/两个斜角的一半

### 37. 解数独(2024-03-25)

## 贪心算法

### 455. 分发饼干(2024-03-26)

- 小饼干喂小胃口/大饼干喂大胃口
- 排序

# 第二章 LeetCode HOT100

### 3 无重复字符的最长子串(2024-02-21)

> time: O(n), space: O(x) // 需要记录字符种类

- 区分字串和子序列
- 滑动窗口，注意长度为1的情况
- 优化：用map记录字符出现位置(右指针发现的重复字符不一定是最开始的那个)
