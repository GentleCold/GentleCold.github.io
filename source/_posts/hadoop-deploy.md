---
title: Hadoop部署实验报告
category: [实验报告, 分布式系统]
date: 2024-03-07 17:56
tags: [Hadoop, 分布式系统]
---

## set java home

```shell
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk
```

## 运行map reduce

<p align="center">
    <img src="/imgs/image-20240307182429.png"/>
</p>

## 运行word count

<p align="center">
    <img src="/imgs/image-20240307183128.png"/>
</p>

<p align="center">
    <img src="/imgs/image-20240307200446.png"/>
</p>

## 启动hdfs

<p align="center">
    <img src="/imgs/image-20240307214653.png"/>
</p>

<p align="center">
    <img src="/imgs/image-20240307214829.png"/>
</p>

## hdfs shell

### 目录操作

<p align="center">
    <img src="/imgs/image-20240307215732.png"/>
</p>

### 文件操作

#### 上传文件

<p align="center">
    <img src="/imgs/image-20240307215916.png"/>
</p>

<p align="center">
    <img src="/imgs/image-20240307220149.png"/>
</p>

#### 下载文件

<p align="center">
    <img src="/imgs/image-20240307220343.png"/>
</p>

#### 拷贝文件

<p align="center">
    <img src="/imgs/image-20240307220639.png"/>
</p>

## 启动Map reduce

<p align="center">
    <img src="/imgs/image-20240307221111.png"/>
</p>

<p align="center">
    <img src="/imgs/image-20240307221407.png"/>
</p>

<p align="center">
    <img src="/imgs/image-20240307221619.png"/>
</p>

### wordcount 运行结果

<p align="center">
    <img src="/imgs/image-20240307221904.png"/>
</p>

### 运行历史

<p align="center">
    <img src="/imgs/image-20240307222029.png"/>
</p>

### 停止所有服务

<p align="center">
    <img src="/imgs/image-20240307222303.png"/>
</p>

## 思考题

1. 前者设置的是每个子进程的最大内存，后者设置的是整个hadoop的最大内存
2. FsShell进程执行文件操作，在此事例中，其负责将本地文件上传至Hdfs文件系统中
3. HDFS: SecondaryNameNode/NameNode/DataNode; MapReduce: JobTracker/RunJar/TaskTracker/Child
4. 通过fsck命令查看(`./bin/hadoop fsck /user/gentle/input -files`):

<p align="center">
    <img src="/imgs/image-20240307230355.png"/>
</p>

如图，input文件夹占据了18个块

添加-files参数，可以查看每个文件的详细块占用:

<p align="center">
    <img src="/imgs/image-20240307230551.png"/>
</p>

5. 无论是map还是reduce都没有改变，因为 Map 任务的数量通常与输入数据的分片数量相关。如果输入数据被划分为较少的分片，那么可能无法实现指定的 Map 任务数量
6. 同理
