---
title: MIT6824-Spring2024课程笔记
category: [笔记]
date: 2024-04-04 23:54
tags: [笔记, MIT, 数据库, 分布式]
---

> 刷完CMU15445，继续开坑6824

# 课程笔记

公开课最新只放了2021版的，所以笔记也是基于此，但是Lab做的是2024版的。

不同于CMU15445，课程没有PPT，但是官网放了官方笔记

和学习CMU15445时一样，本笔记只记录关键词

## Introduction

- why distributed
  - connect(sharing)
  - increase parallelism
  - tolerate faults
  - achieve security(isolate)
- history
  - dns/email
  - datacenters(web)
  - cloud computing

## RPC and threads

- using channels/using mutex+cond
- sync.Mutex / sync.NewCond / cond.Broadcast / cond.Wait
- remote procedure call
- client -> stub -> stub -> server

## GFS

About GFS: https://zhuanlan.zhihu.com/p/354450124

- fault tolerance -> replication -> consistency
- ideal consistency
- gfs: big/fast/global
- gfs client/master/chunkserver
- master
  - filename -> array of chunk handles
  - chunkhandle -> version number/list of chunk servers
  - log + checkpoints
- relax consistency model
  - if some success, but some is error. Retry, but the success will write twice

## Primary/backup replication

About VM-FT: https://www.cnblogs.com/brianleelxt/p/13245754.html

- state transfer / replicated state machine(physical/logical)

## Raft

About Raft:

- https://www.cnblogs.com/brianleelxt/p/13251540.html
- https://www.cnblogs.com/xybaby/p/10124083.html
- https://docs.qq.com/doc/DY0VxSkVGWHFYSlZJ

- prev pattern: single point of failure
- majority rule
- no split brain(using term) / maybe split vote(using election timeouts)
- logs may cliverge

# 项目思路

## MapReduce

Master为Worker分配任务，输入数据(来自GFS)，Map将中间数据存储在本地，Reducer通过remote的方式获取，最后输出到GFS

当worker在一定时间内没有反应，就认为出了故障，此时重新执行，并且这个worker上执行成功的所有map都得重新执行(认为这个worker无法被访问了)

Map和Reduce可能重复运行，但是结果是不变的

Map过程为生成一组KeyValue对，然后对Key做partition将结果分到nReduce个桶中

Reduce读取这个桶下的所有中间文件，将所有KeyValue对按照Key排序，然后得到每组`Key[Values]`传入用户reduce函数得到结果

### 文件结构

给出的`main/mrsequential.go`为单线程mapreduce参考文件

需要实现伪分布式mapreduce，主线程为`main/mrcoordinator.go`和`main/mrworker.go`

在本次实验中，map的结果存储为文件，不用考虑reducer对文件的remote调用

另外在`mrapps`目录下放置了很多mapreduce的应用文件(例如`wc.go`)，以插件的形式加载

需要修改的是`mr`目录下文件

### 整体思路

- 一个任务包括任务类型、任务编号、输入文件名
- Coordinator初始化时即分配好Map任务
- Worker初始化后先向Coordinator获得一个worker_id，然后向Coordinator请求任务，拿到任务并执行成功后，通知Coordinator
- 若任务已完成，不再接受其他此任务的成功完成，同时将记录Map任务完成后生成的中间文件名
- Worker生成中间文件命名为`mr-X-Y-W`，分别为Map任务编号，Reduce任务编号，Worker编号
- 每次分配任务都开一个后台线程，睡眠10s然后检查任务是否完成，否则将任务重新添加
- 通过Channel添加和分配任务、通过WaitGroup判断任务完成阶段（Map->Reduce->Finish）

### 结果

任务的容错处理比较简单，只要10s内没有结果就再安排一次任务，这样可能多个Worker同时执行同一个任务，由于Map中间文件名已经通过Worker编号标识，并且Coordinator只认首次完成成功的结果，所以不会造成冲突，而对于Reduce来说，由于每次Reduce的结果都一样，可能有多个Reduce同时写一个输出文件，但是结果不会受到影响

`sh test-mr-many.sh 500` 运行500次没出现故障

## Raft

实验仅需修改`raft.go`文件，测试代码位于`raft/test_test.go`

### Part A: leader election

阅读测试代码，首先根据论文图2完善raft相关结构，然后在Make函数中进行初始化。

另外由于测试要求1s不超过几十次心跳发送，所以设置心跳发送间隔为100ms，选举超时设置为300\~600ms，同时设置ticker每10ms检查是否需要进行选举。

具体步骤为，ticker每10s检查超时，如果超时则变为candidate，重置时间，给自己投票，然后广播发起选举，为每个节点启动一个goroutine，发起RequestVote RPC调用，见函数handleRequestVote

在handleRequestVote中，处理RPC调用的reply，如果得票超过半数，则变为leader，否则如果遇到新的term，则从candidate变为follower。变为leader后启动heartbeat goroutine。

在heartbeat中，每100s为每个节点启动一个goroutine，发送不包括entries的AppendEntries RPC调用，见函数handleAppendEntries。

在handleAppendEntries中，处理RPC调用的reply，如果遇到新的term，则从leader变为follower。

函数调用关系为：

```mermaid
flowchart LR
    ticker --> handleRequestVote --> heartbeat --> handleAppendEntries
```

然后就是RPC函数过程，对于RequestVote来说，如果request的term更大，则更新状态（follower，term），同时重置超时时间，返回voteGranted为true，否则只返回false

对于AppendEntries来说，如果request的term更大或相等，则更新状态（follower，term），同时重置超时时间，返回success为true，否则只返回false。

测试结果如下：

<p align="center">
    <img src="/imgs/image-20240829230750.png"/>
</p>

### Part B: log

首先考虑raft算法本身，一个index和term唯一确定一个log，已经commit的log不会丢失，已经apply的log不会在此处被其他节点apply到不一致的log，rpc的调用是幂等性的

在代码编写过程中可以多考虑一下是否满足了这些条件，考虑一下leader/follower/candidate的宕机或者网络分区是否会破坏这些要求

本部分的重点是nextIndex和matchIndex，leader将根据这两个属性来进行节点共识处理

#### rpc struct

首先根据论文完善rpc相关结构

其中对于属性nextIndex[]和matchIndex[]：这两个属性由leader节点管理，前者记录每个节点下一个需要的log的index，后者记录每个节点与leader匹配的最大的log的index，在理想情况下，这两个数组实际是一样的，问题发生在重新选出leader时，前者将初始化为leader的最大的logIndex，而后者将从0开始往后递增。从作用上来看，leader根据前者找到prevLog进行一致性检验，根据后者判断majority的log复制，来更新commitIndex

#### basic process

客户端将从Start函数处提出请求，然后又raft来达成共识，如果当前节点是leader，此函数会在本地增加log，并立即返回。之后将进行共识，不保证一定达成共识。

修改heartbeat函数，原先发送的AppendEntries的entries默认为空，但此时会根据对应节点的nextIndex的值发送从此处开始之后所有的log，同时根据此值确定prevLogIndex和prevLogTerm，以进行一致性检查。如果nextIndex和leader的log长度一致，则不需要发送log，保持发送心跳，但仍要进行一致性检查。

修改AppendEntries函数，添加一致性检查的过程，分为三种情况：

- prevLogIndex处没有log，返回false
- log为空或者一致性检查成功，进行append操作，返回true
- 一致性检查失败，返回false

需要在进行append时确保覆盖操作，即覆盖PrevLogIndex之后所有的log为leader的log

对于leader，根据reply的success判断日志复制是否成功，如果成功，更新nextIndex和matchIndex，如果失败，更新nextIndex为不一致的位置

另外修改投票逻辑，选出的新leader需要重置nextIndex和matchIndex

#### commit and apply

对于leader，在每一次日志复制成功后将检查matchIndex，即找到最大的index，使得log[index].Term为当前term（对应论文5.4.2要求commit当前任期下的log才能commit之前的log），并且matchIndex数组中的大多数大于等于index（注意这里不用考虑leader本身的matchIndex），则可更新commitIndex为index

对于follower，在AppendEntries中，当且仅当成功进行日志复制，即完成同步，才更新commitIndex为min（len(log), LeaderCommit)

对于所有节点，apply的过程放在ticker函数中，即每10ms检查一次LastApplied和CommitIndex，将尚未应用但已提交的log通过applyCh通道传递来表示应用到状态机

#### election restriction

为避免论文5.4.1所说的问题，添加选举限制，即选举时多传一份LastLogIndex和LastLogTerm，当且仅当candidate的log比follower的新才能获得选票（term大的或者term相同时，index大的），来确保不让log不一致的节点当选leader覆盖掉已经commit的log

#### fast rollback

在原先leader处理日志复制失败的情况时，是一个log一个log回退的，如果出现大量不一致log会导致效率下降（尽管这种情况实际很少发生），为提高效率可以根据不一致log的term来回退

为此需要在AppendEntries的返回中增添两个属性ConflictIndex和ConflictTerm，考虑复制失败中的两种情况：

- prevLogIndex处没有log，更新ConflictIndex为本地最后一个log位置
- 一致性检查失败，更新ConflictIndex为冲突log的任期下的第一个log位置

leader处理失败时，可以直接更新nextIndex为ConflictIndex，也可以确认此处log的term是否为ConflictTerm，如果是则表示这里已经一致了，就nextIndex++，直到找到不一致的地方，这样可以避免多传输不必要的log

测试结果如下：

<p align="center">
    <img src="/imgs/image-20240901181858.png"/>
</p>
