---
title: Minichain实验报告
category: [实验报告, 区块链]
date: 2024-02-29 15:49
tags: [区块链]
---

# 挖矿模拟

## Minichain 代码结构

```mermaid
flowchart
    subgraph our works
    MinerNode --+Transaction--> BlockBody
    MinerNode --+previous hash--> BlockHead
    BlockBody --> Block
    BlockHead --> Block
    end
    Block --satisfy difficulty--> BlockChain

    TransactionProducer --> Transaction --add to--> TransactionPool

    Network --start thread--> MinerNode
    Network --start thread--> TransactionProducer
```

需要实现的部分即MinerNode生成Block的部分

## 生成Merkle树根

此部分借助 Queue 实现

## 习题

### Q1

1. 分别对D01/D02/D03/D04作哈希得到哈希值
2. 拼接D01和D02的哈希值，再作哈希得到Hash11，对D03和D04拼接，再作哈希得到Hash12
3. 最后拼接Hash11和Hash12，做哈希得到最终的TopHash

### Q2

构造过程如下

```mermaid
flowchart BT
A -->Ah[Hash A = 4]
B -->Bh[Hash B = 3]
C -->Ch[Hash C = 1]
D -->Dh[Hash D = 0]
E -->Eh[Hash E = 3]

Ah --> ABh[Hash AB = 2]
Bh --> ABh

Ch --> CDh[Hash CD = 1]
Dh --> CDh

Eh --> EEh[Hash EE = 1]
Eh1[Hash E = 3] --> EEh

ABh --> ABCDh[Hash ABCD = 3]
CDh --> ABCDh

EEh --> EEEh[Hash EEEE = 2]
EEh1[Hash EE = 1] --> EEEh

ABCDh --> TopHash[TopHash = 0]
EEEh --> TopHash
```

### Q3

TransactionProducer 作为生产者，每次生产一个Transaction，然后放入TransactionPool

当TransactionPool达到最大值，通知挖矿进程MinerNode，会将所有的Transaction取出，然后计算Merkle树根的值，加上前一个区块的哈希值，以及一个随机数nonce得到区块头，并与所有的Transaction组成的区块体结合构成一个区块

计算整个区块体的哈希值，挖矿进程通过不断随机nonce值使得哈希值满足难度系数（前n位为0），则将其加入区块链（前一个区块的后面）

# 交易模拟

## 签名，验签，UTXO
