---
title: Raft论文浅读
category: [论文阅读]
date: 2024-04-14 17:13
tags: [Raft, 数据库, 分布式]
---

> 论文为：In Search of an Understandable Consensus Algorithm (Extended Version)

https://pdos.csail.mit.edu/6.824/papers/raft-extended.pdf

# 1 Raft 介绍

作为共识算法，Raft有三个特性：

- 只有一个Leader
- 使用随机定时器来选举
- 允许运行时改变配置

# 2 复制状态机

<p align="center">
<img src="/imgs/image-20240414175838.png"/>
</p>

共识算法处理Client的命令，负责Log的同步，Log应用到状态机上，然后通知Client

# 3 Paxos的问题

- 难以理解
- 实践较少

# 4 可理解性设计

- 将问题分割
  - leader election
  - log replication
  - safety
  - membership changes
- 减少状态空间

# 5 Raft算法

首先选举一个Leader，只有Leader在线时才能处理

Leader需要处理来自客户端的Log Entries，复制这些Log，通知服务器在合适的时候应用这些Log

此节将介绍Raft的三个方面：

- 当一个Leader寄了要进行新的选举
- Leader需要将Log复制给所有节点
- 当一个状态机应用Log时，保证所有的状态机在这个index下都应用这个Log

## 5.1 Raft basics

- 节点都有三个状态：Leader、Follower、Candidate

<p align="center">
<img src="/imgs/image-20240414180058.png"/>
</p>

- Terms充当逻辑时间，一次Term由选举产生，选举可能出现Split vote导致没有Leader选出，此时重新产生选举
- 当Server间通信时，会更改Term和状态，对于两个通信的Server：
  - 旧的Term会同步到新的Term
  - 如果Candidate或Leader是旧的Term，则转为Follower
  - 拒绝旧Term的请求
- Terms保证了不会产生脑裂

<p align="center">
<img src="/imgs/image-20240414180443.png"/>
</p>

- Server间通过RPC通信，目前只考虑两种RPC: RequestVote(candidate)和AppendEntries(leader)，也就是说Follower不会主动发起通信
- PRC并行发起，会被重试

## 5.2 Leader election

刚开始所有节点为Follower。Follower在一段时间（election timeout）内未收到RPC通信（将空的AppendEntries RPC当作来自Leader的心跳）则变为Candidate开始选举，一个节点只能投一票(一个任期中)

开始选举后：

- 增加Terms
- 给自己投一票
- 给所有节点发送RequestVote RPC

什么时候结束选举：

- 赢得选举
- 已经选出Leader
- 超时而仍然没有Leader

当获得Majority票数后成为Leader，开始发送心跳

可能出现Split votes，即多个Candidate但是没有Majority票数，此时等待超时然后重新发起选举。同时，使用随机化的timeouts来减少split vote出现的可能

结果表明这种方式也能快速完成选举，同时减少了逻辑复杂度(本来打算引入ranking system)

## 5.3 Log replication

选举出Leader后，Leader处理来自客户端的Log，向所有节点发送AppendEntries RPC。如果follower没有给出合适的应答，会无期限重试。如果收到了RPC，需要进行一致性检查（前一个Log是否有相同的index和term）

当Log被大多数server复制后，这个Log被认为是committed，可以应用到状态机上。Raft保证committed的log最终一定会被执行

不一致发生在Leader寄了的时候，Leader中的Log可能没有被committed。Raft通过覆盖Follower的Log为Leader的Log来处理不一致。Leader为每个Follower维护一个nextIndex（初始化为自己的nextIndex），当RPC被拒绝后会减少nextIndex直到一致性检查成功，然后插入新的Log

## 5.4 Safety

此节对Raft如何保证一致性进行进一步的解释

### 5.4.1 Election restriction

Raft算法通过给选举添加限制来保证被选举的Leader包含之前所有committed的Log。即发起选举时，如果follower的Log比candidate新，则拒绝投票。

如此一来，若Candidate缺失了committed的Log，则不会获得majority的票数。但是如果Candidate通过选举，仍可能包含未被committed的Log

### 5.4.2 Committing entries from previous terms

所以Leader选举后要完成之前未完成的工作。如前所述，Leader仍然只是覆盖所有不一致的地方，意味着可能覆盖掉旧的但是仍被majority记录的Log。为解决这个问题，Leader只有当前任期下的Log被Majority记录后才考虑commit

<p align="center">
<img src="/imgs/image-20240415011431.png"/>
</p>

如图所示，\(c\)中的2并未被commit，只有(e)才能被commit，此时S5无法被选举成功。当然也有别的办法判断2是否应该被commit，或者直接更新这个Log的Term，但Raft中Leader不会更改旧的Log，只是Append。Raft为保持简单使用了保守的策略，虽然这样可能会导致发出2命令的客户端等待更长的时间

### 5.4.3
