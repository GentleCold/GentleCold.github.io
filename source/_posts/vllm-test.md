---
title: VLLM测试
category: [实验]
date: 2025-06-19 14:23
tags: [VLLM]
---

## 1. 数据集

imdb影评情感分析数据集：http://ai.stanford.edu/~amaas/data/sentiment/

csv文件，格式类似如下：

| review  | sentiment |
| ------- | --------- |
| text... | postive   |
| text... | negtive   |

## 2. 测试

使用模型：NousResearch/Hermes-3-Llama-3.1-8B

使用显卡：单张H800

模型最大上下文限制为(prompt tokens + output tokens)：131072

KV Cache计算器：https://lmcache.ai/kv_cache_calculator.html

### 2.1 离线吞吐量测试

#### 2.1.1 测试1

为使用1条prompt，格式为n个text按换行符拼接+要求：

`"text1\n
text2\n
...
For each line of the above comments, determine whether it is a positive or negative comment. Answer only postive or negative:\n"`

同时限制output token为n

#### 2.1.2 测试2

为使用n条prompt，格式为1个text+要求：

`"text1\n
For the above comment, determine whether it is a positive or negative comment. Answer only postive or negative:\n"`

`"text2\n
For the above comment, determine whether it is a positive or negative comment. Answer only postive or negative:\n"`

`...`

同时限制每条output token为1

#### 2.1.3 测试结果

n = 400（此时对于单条prompt来说可以认为几乎跑满了模型最大tokens限制）

```bash
测试1:
Throughput: 0.04 requests/s, 5058.22 total tokens/s, 16.80 output tokens/s
Total num prompt tokens: 120006
Total num output tokens: 400

测试2:
Throughput: 65.75 requests/s, 21299.52 total tokens/s, 65.75 output tokens/s
Total num prompt tokens: 129180
Total num output tokens: 400
```

n = 40

```bash
测试1:
Throughput: 1.02 requests/s, 11443.23 total tokens/s, 40.65 output tokens/s
Total num prompt tokens: 11220
Total num output tokens: 40

测试2:
Throughput: 73.27 requests/s, 22262.43 total tokens/s, 73.27 output tokens/s
Total num prompt tokens: 12114
Total num output tokens: 40
```

n = 4

```bash
测试1:
Throughput: 8.01 requests/s, 8118.47 total tokens/s, 32.03 output tokens/s
Total num prompt tokens: 1010
Total num output tokens: 4

测试2:
Throughput: 32.03 requests/s, 8648.41 total tokens/s, 32.03 output tokens/s
Total num prompt tokens: 1076
Total num output tokens: 4
```

测试2的吞吐均大于测试1

考虑prefill阶段，测试2的batchsize更大，不考虑prefix/kv cache复用的话，长prompt的prefill（n^2）肯定是没有多个短prompt（n）快的，测试1是没有优势的

如果使用对文本的kvcache复用（可以重复交两个一样的请求，然后利用prefix机制来复用），此时在计算量上才能显现测试1的优势，因为测试1的prompt tokens数是小于测试2的（basic prompt内部的交叉注意力会多次计算）

但是考虑decode阶段，prompt越长qkv点乘计算越慢，所以测试1 decode阶段还是没有优势的

#### 2.1.4 验证

对于测试1和测试2，同时设置一个duplicate=100，表示一个请求重复提交100次，那么后99次都会使用prefix cache进行缓存复用（同时避免basic prompt的复用，即只考虑文本块的复用），可以近似认为算出的吞吐量为完全复用时的吞吐量：

```bash
100x duplication
测试1:
Throughput: 2.47 requests/s, 297019.33 total tokens/s, 986.72 output tokens/s
Total num prompt tokens:  12000700
Total num output tokens:  40000

测试2:
Throughput: 703.70 requests/s, 229369.89 total tokens/s, 703.70 output tokens/s
Total num prompt tokens:  12998000
Total num output tokens:  40000
```

说明如果能完全复用kv cache的话，测试1的吞吐量是更有优势的，但是decode没办法看出来，这里的output tokens计算方式是`output tokens / elapsed time`

如果是完全复用的话：

```bash
100x duplication
测试1:
Throughput: 2.50 requests/s, 301121.59 total tokens/s, 1000.35 output tokens/s
Total num prompt tokens:  12000600
Total num output tokens:  40000

测试2:
Throughput: 1465.77 requests/s, 476303.48 total tokens/s, 1465.77 output tokens/s
Total num prompt tokens:  12958000
Total num output tokens:  40000
```

此时又是测试2快

### 2.2 在线吞吐量测试

n = 400

```bash
测试1:
============ Serving Benchmark Result ============
Successful requests:                     1
Benchmark duration (s):                  7.29
Total input tokens:                      120006
Total generated tokens:                  400
Request throughput (req/s):              0.14
Output token throughput (tok/s):         54.89
Total Token throughput (tok/s):          16524.00
---------------Time to First Token----------------
Mean TTFT (ms):                          293.37
Median TTFT (ms):                        293.37
P99 TTFT (ms):                           293.37
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          17.52
Median TPOT (ms):                        17.52
P99 TPOT (ms):                           17.52
---------------Inter-token Latency----------------
Mean ITL (ms):                           17.52
Median ITL (ms):                         17.40
P99 ITL (ms):                            19.57
==================================================

测试2:
============ Serving Benchmark Result ============
Successful requests:                     400
Benchmark duration (s):                  4.84
Total input tokens:                      129180
Total generated tokens:                  400
Request throughput (req/s):              82.71
Output token throughput (tok/s):         82.71
Total Token throughput (tok/s):          26794.76
---------------Time to First Token----------------
Mean TTFT (ms):                          2706.23
Median TTFT (ms):                        2815.29
P99 TTFT (ms):                           4828.05
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          0.00
Median TPOT (ms):                        0.00
P99 TPOT (ms):                           0.00
---------------Inter-token Latency----------------
Mean ITL (ms):                           0.00
Median ITL (ms):                         0.00
P99 ITL (ms):                            0.00
==================================================
```
