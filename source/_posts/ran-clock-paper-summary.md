---
title: Ran-CLOCK论文调研
category: [笔记]
date: 2026-05-08 14:54
tags: [Cache, CLOCK, SIEVE, Mean Field]
---

论文：Performance Analysis of the Randomized SIEVE/CLOCK Cache Replacement Algorithm

作者：Yirong Wang, Peter Desnoyers, Benny Van Houdt

发表：Proc. ACM Meas. Anal. Comput. Syst., Vol. 10, No. 2, Article 49。PDF元数据标注为2026年6月。

链接：https://effywn.com/assets/pdf/Ran-CLOCK.pdf

DOI：https://doi.org/10.1145/3805647

## 1. 背景

这篇论文讨论的是缓存替换算法，尤其是CLOCK和SIEVE这一类“小状态、低开销”的算法。

CLOCK的基本做法是：每个缓存对象维护一个access bit，命中时把bit置为1；发生miss需要驱逐时，clock hand顺序扫描缓存对象，遇到bit为1的对象就清零并跳过，遇到bit为0的对象就驱逐。

SIEVE和CLOCK很像，也维护每个对象的access bit，也是在驱逐时跳过被访问过的对象。区别是SIEVE把缓存对象放在FIFO顺序的链表里，新对象固定插入到head，而不是像CLOCK那样原地替换被驱逐的位置。

这类算法的优点很明确：

1. 每个对象只需要很少状态。
2. 驱逐逻辑简单。
3. 不需要维护完整LRU链表的复杂更新。
4. 工程实现成本比精确LRU低。

但它们有一个重要弱点：**遇到长扫描序列时容易出现performance cliff。**

所谓performance cliff，就是缓存大小略低于某个临界点时命中率很差；一旦超过临界点，命中率突然大幅上升。对系统来说，这很危险：如果缓存容量刚好配在 cliff 左侧，增加一点容量也没什么收益，但再增加一点又突然变好，容量规划会变得非常不稳定。

论文用Alibaba block I/O trace volume28说明这个现象：SIEVE、CLOCK、LRU、FIFO等策略在某些cache size区间都会出现明显 cliff，而RANDOM反而能让曲线更平滑，平均命中率还比SIEVE高约10.22%。

这就是论文的动机：**能不能保留CLOCK/SIEVE的小状态优点，同时用随机化缓解扫描负载带来的同步驱逐问题？**

## 2. 核心问题

传统CLOCK/SIEVE的驱逐路径是确定性的。

在miss时，clock hand或SIEVE hand会沿着固定顺序扫描对象。长扫描负载会让一批对象按照相近节奏被插入、清bit、驱逐，形成同步化的aging frontier。结果是：

1. 扫描序列中的对象成批进入缓存。
2. 旧对象被顺序挤出。
3. 当扫描长度略大于cache size时，之前加载的对象很可能在下一轮扫描前刚好被挤掉。
4. 命中率在某些cache size附近突然跳变。

这个问题不是简单调大access bit就能完全解决，因为根因之一是**顺序扫描产生了结构化、同步化的驱逐压力**。

论文的判断是：如果把驱逐候选从“顺序扫描”改成“随机探测”，就可以把驱逐压力分散到不同对象上。即使负载本身是长扫描，缓存对象也不会按同一个确定顺序一起老化和淘汰。

## 3. 算法设计

论文提出的是Ran-CLOCK(K)和Ran-SIEVE(K)。

这里的$K$是每个缓存对象的access counter上限。对象的计数器取值范围是：

$$
0, 1, \ldots, K
$$

需要的状态位数是：

$$
\left\lceil \log_2(K + 1) \right\rceil
$$

例如：

1. $K = 1$时，只需要1 bit，退化为普通CLOCK/SIEVE风格的access bit。
2. $K = 15$时，需要4 bits。
3. $K = 0$时，没有保护计数，Ran-CLOCK/Ran-SIEVE退化为RANDOM。

### 3.1 CLOCK(K) / SIEVE(K)

论文先把CLOCK和SIEVE自然推广到多计数器版本。

命中时：

$$
\mathrm{counter} = \min(\mathrm{counter} + 1, K)
$$

miss且缓存满时：

1. 按原来的顺序扫描候选对象。
2. 如果候选对象counter为0，就驱逐它。
3. 如果候选对象counter大于0，就把counter减1，然后继续扫描。

对CLOCK(K)来说，新对象插入到被驱逐的位置，counter初始化为0。

对SIEVE(K)来说，被驱逐对象从链表中删除，新对象固定插入到head，counter初始化为0。

### 3.2 Ran-CLOCK(K) / Ran-SIEVE(K)

随机化版本只改一个关键点：**miss时不再顺序扫描，而是从当前缓存对象中均匀随机采样候选。**

流程是：

1. miss发生，缓存已满。
2. 从缓存中uniform random选择一个对象。
3. 如果它的counter为0，就驱逐它并插入新对象。
4. 如果它的counter大于0，就把counter减1，然后继续随机选择对象。

采样是with replacement，也就是同一次miss处理中，一个对象理论上可能被抽到多次。不过在缓存规模很大时，同一轮探测里重复抽中同一对象的概率趋近于0，因此均场模型里可以忽略这个影响。

因为随机探测不依赖list order，所以Ran-CLOCK(K)和Ran-SIEVE(K)在命中率意义上是等价的。二者只在内部数据结构和插入位置上不同，但对象计数器的随机演化过程相同。

## 4. 直觉：为什么随机化抗扫描

顺序CLOCK/SIEVE的核心问题是驱逐路径会跟扫描负载发生同步。

如果请求流是反复扫描一段长度为$s$的对象序列，而cache size是$C < s$，那么顺序策略很容易出现“上一轮刚放进去的对象，在下一轮回来之前已经按顺序被挤掉”的情况。于是cache size只要略小于扫描长度，命中率就可能接近0。

RANDOM或Ran-CLOCK不同。每次miss的驱逐位置是随机的，一个对象是否被淘汰更像独立的生灭过程：

1. 请求命中会增加counter，给对象更多保护。
2. miss时随机探测会给某些对象施加counter decrement或eviction压力。
3. 驱逐压力被分散，而不是沿着固定frontier推进。

论文在第7节给了一个简单模型。对重复扫描长度`s`、cache size `C` 的场景，RANDOM替换下，一个对象在一轮扫描中存活的概率可近似写成：

$$
\left(1 - \frac{1}{C}\right)^{s P_{miss}} = 1 - P_{miss}
$$

其中$P_{miss}$是长期miss概率。用指数近似后，可以得到一个关于$C/s$的平滑命中率曲线。

关键结论是：**随机替换不会在某个临界cache size突然跳变，而是随着$C/s$增大逐步提高命中率。**

这解释了为什么Ran-CLOCK/Ran-SIEVE能缓解scan-induced performance cliff。

## 5. 均场模型

论文的另一大贡献是给Ran-CLOCK/Ran-SIEVE建立了可计算的heterogeneous mean-field model。

之所以能做得比较干净，是因为随机探测把复杂的顺序扫描耦合变成了近似的对象级随机过程。每个对象可以被看成一个有限容量队列：

1. 请求命中相当于arrival，会让counter上升。
2. 随机探测相当于service，会让counter下降。
3. counter降到0后再被探测，就会被驱逐。

### 5.1 状态变量

对对象`k`，定义：

$$
x_{k,-1}(t): \text{对象 } k \text{ 不在缓存中的概率}
$$

$$
x_{k,j}(t): \text{对象 } k \text{ 在缓存中且 counter}=j \text{ 的概率}, \quad j = 0,\ldots,K
$$

缓存容量为`C`，对象总数为`n`。如果缓存一直是满的，那么：

$$
\sum_k x_{k,-1}(t) = n - C
$$

再定义：

$$
m(t) = \sum_k p_k x_{k,-1}(t)
$$

表示当前miss rate，其中$p_k$是对象$k$的请求概率。

还定义：

$$
x_0(t) = \sum_k x_{k,0}(t)
$$

表示缓存中counter为0的对象期望数量。

在mean-field视角下，每次miss最终会驱逐一个counter为0的对象；为了找到它，随机探测平均会对counter大于0的对象做若干次decrement。有效的service rate可以写成：

$$
z = \frac{m}{x_0}
$$

这就是每个对象counter下降的全局压力。

### 5.2 IRM下的固定点

在Independent Reference Model下，对象$k$的请求是速率为$p_k$的Poisson过程。

论文证明固定点唯一，并且有非常简洁的形式：

$$
x_{k,j} =
\frac{(p_k / z_K)^{j+1}}
{\sum_{i=0}^{K+1} (p_k / z_K)^i}
$$

其中：

$$
j = -1, 0, \ldots, K
$$

$z_K$由缓存容量约束唯一确定：

$$
n - C =
\sum_k
\frac{1}
{\sum_{i=0}^{K+1} (p_k / z)^i}
$$

这个方程对`z`单调，可以在`(0, 1)`上用二分法求解。

固定点下的miss probability是：

$$
P_{miss}(C, K) =
\sum_k
\frac{p_k}
{\sum_{i=0}^{K+1} (p_k / z_K)^i}
$$

这个结果很有用：Ran-CLOCK/Ran-SIEVE的命中率预测不需要模拟完整缓存轨迹，只需要解一个标量$z_K$。

### 5.3 队列解释

固定点分布等价于一个$M/M/1/K+1$队列的平稳分布：

1. arrival rate是对象请求率$p_k$。
2. service rate是全局随机探测压力$z_K$。
3. 队列长度对应对象状态：空队列对应不在缓存，非空队列对应counter状态。

这个解释很直观：热门对象arrival rate高，counter更容易被推高，因此更不容易被驱逐；冷对象arrival rate低，被随机探测慢慢降到0后更容易离开缓存。

### 5.4 K趋于无穷时的表达式

论文还给了$K \to \infty$时的显式表达式。

直观上，当counter上限无限大，足够热门的对象会被永久保护，几乎不产生miss；剩余不够热门的对象共享剩余缓存容量。

论文定义一个阈值位置`ell`，表示最热门的一批对象。`ell`个最热门对象在极限中被稳定缓存；剩余对象的miss probability由剩余请求质量和collision probability决定。

这部分结果的意义不只是数学形式漂亮，它说明Ran-CLOCK(K)随着`K`增大，会逐渐接近一种“按频率稳定保护热门对象”的策略。

## 6. PH renewal模型

IRM假设请求是memoryless的，但真实负载往往有recency结构：同样的对象流行度下，不同inter-request time分布会导致完全不同的缓存表现。

因此论文还分析了phase-type renewal model。每个对象的请求间隔服从PH分布，用`(alpha_k, T_k)`表示。

在这个模型下，对象状态不再只是counter，还要包含当前renewal phase。论文把：

$$
x_{k,j}(t)
$$

扩展成向量，表示对象`k`处于counter `j`且处于不同PH phase的概率。

固定点结果仍然成立，但对象级队列从：

$$
M/M/1/K+1
$$

变成：

$$
PH/M/1/K+1
$$

服务率仍然是一个全局标量`z`，并且仍然可以通过缓存容量约束用二分法求解。

这一节的价值在于：论文不是只在简单IRM假设下分析算法，而是把recency结构也纳入模型。实验也显示，保持相同popularity但改变inter-request time分布，策略排名会变化。这对缓存算法很关键，因为真实trace里frequency和recency往往不能混为一谈。

## 7. 模型准确性

论文用synthetic trace验证均场模型。

设置包括：

1. IRM Zipf popularity，$\theta = 0.5, 0.8, 1.1$。
2. PH renewal workload，保持Zipf popularity，但请求间隔改成两相hyperexponential分布。
3. $K = 1, 15, 63$等不同counter上限。
4. 对象数量`n`从30到960。
5. trace length为$10^7$。

结果是：

1. 点估计上，mean-field miss probability和simulation非常接近。
2. `n`越大，误差越小。
3. 在$n = 960$时，miss ratio curve的MAE通常低于$0.0005$。
4. 误差较大的情况主要出现在`n`较小、Zipf skew较大时。

论文还在附录给了理论支持：对一个稍微修改过的Ran-CLOCK/Ran-SIEVE$(K, d^*)$，把每次miss最多探测的对象数截断到$d^*$，可以套用heterogeneous mean-field理论，得到$O(1/n)$级别的误差解释。

这个理论支持依赖两个条件：

1. 每次状态转移影响的对象数有界。
2. 交互率满足合适的规模化形式。

原始算法一次miss的探测次数理论上无界，所以论文用截断版本来建立理论支撑。这个处理是合理的：工程上探测次数通常也不会无限增长，实验中平均probe数很小。

## 8. 实验结果

### 8.1 Markovian synthetic workloads

在IRM和PH synthetic workloads上，论文比较了：

1. SIEVE(K)
2. CLOCK(K)
3. Ran-SIEVE(K)
4. Ran-CLOCK(K)
5. LRU
6. FIFO
7. RANDOM-style list policies

主要结论：

1. 在IRM下，SIEVE(K=15)表现最好，SIEVE(K=1)第二，Ran-CLOCK/Ran-SIEVE(K=15)第三。
2. 在PH下，策略差距变小，CLOCK(K=15)表现最好。
3. Ran-CLOCK/Ran-SIEVE在$K=1$时表现较弱，但$K=15$后进入前几名。
4. 同样的popularity下，只改变inter-request time分布，就可能改变策略排名。

这说明随机化不是在所有理论负载下都支配传统CLOCK/SIEVE。它牺牲了一部分顺序扫描策略在Markovian环境下对recency的利用能力，换来更强的scan robustness。

### 8.2 K的影响

论文重点分析了counter上限$K$。

在$K = 1, 15, 255, 65535$的实验中，命中率随$K$增加而提高，但边际收益很快递减：

1. 从$K=1$到$K=15$收益明显。
2. 从$K=15$继续增加到$255$或$65535$，曲线几乎重合。

这给出一个很实用的结论：**4 bits左右的counter就足够拿到大部分收益。**

平均probe次数也随$K$增加而增加，但同样会很快饱和。论文给出的一个例子是：

1. $K=15, C=24, \theta=0.8, n=120$
2. IRM下mean-field预测$x_0 \approx 14.19$，平均probe约$24 / 14.19 \approx 1.69$
3. PH下$x_0 \approx 10.96$，平均probe约$24 / 10.96 \approx 2.19$

也就是说，即使用4-bit counter，miss时平均只需要约2次随机探测，不是很重的开销。

### 8.3 生产trace

论文在四个生产block I/O trace上测试：

1. Alibaba volume28
2. Alibaba volume766
3. CloudPhysics w11
4. CloudPhysics w44

结果更能体现随机化的价值。

SIEVE在一些trace上表现很好，例如w44和volume766平均命中率最高；但在volume28上有明显cliff，在w11上表现很差。

CLOCK在w11上最好，但同样不具备稳定的scan resistance。

Ran-CLOCK/Ran-SIEVE表现更稳：

1. 在volume28上平均命中率最高，论文报告为0.4170，而第二名SIEVE(K=15)为0.3614。
2. 在w44上排第二，接近SIEVE(K=1)，平均命中率约0.3986 vs 0.4019。
3. 在w11上明显优于SIEVE，排在CLOCK和LRU之后。
4. 在volume766上不如SIEVE，但差距不大，平均命中率约0.7471 vs 0.7558。

最重要的是，在w44和volume28这类cache-unfriendly trace上，Ran-CLOCK/Ran-SIEVE的命中率随cache size增长更平滑，没有传统策略那种明显cliff。

## 9. 工程视角

这篇论文的算法实现并不复杂。

如果已有CLOCK或SIEVE实现，Ran-CLOCK/Ran-SIEVE需要的变化大致是：

1. 每个对象维护一个小counter，而不是单bit。
2. miss时用随机采样选择候选对象，而不是沿hand顺序扫描。
3. counter为0则驱逐，counter大于0则减1。
4. 命中时counter饱和加1。

从工程角度看，主要问题有几个。

### 9.1 随机采样结构

Ran-CLOCK需要能从当前缓存对象中均匀随机取样。

如果缓存对象放在数组或slot table里，这比较自然；如果是链表结构，均匀随机采样可能需要额外索引。论文从算法行为上说明Ran-CLOCK和Ran-SIEVE等价，但真正实现时，CLOCK式数组结构可能更适合随机探测。

### 9.2 并发和锁

顺序CLOCK通常只移动一个hand，虽然也有并发争用，但访问模式比较局部。随机探测会触碰随机slot，可能影响cache locality，也可能让并发控制方式不同。

如果缓存系统本身已经是分片结构，比较自然的做法是在每个shard内部随机探测，而不是全局随机探测。

### 9.3 探测次数尾部

平均probe次数不高，但理论上一次miss可能探测很多次，特别是当多数对象counter都大于0时。

工程实现可以考虑设置最大探测次数：

1. 前`d`次按Ran-CLOCK逻辑随机探测。
2. 如果仍没有找到counter为0的对象，就随机驱逐或退化到其他策略。

这也和论文附录中的截断版本Ran-CLOCK/Ran-SIEVE(K, d*)一致。

### 9.4 K的选择

论文实验给出的建议很清楚：不要盲目把$K$设得很大。

$K=15$已经基本拿到主要收益，对应4 bits per object。再增大counter不仅收益小，还可能增加状态维护、探测次数和实现复杂度。

## 10. 和其他策略的关系

### 10.1 相比LRU

LRU维护精确recency，对局部性强的负载很有效。但它在长扫描场景下可能把热点挤出去，也有较高维护成本。

Ran-CLOCK不是要精确近似LRU，而是选择了另一种折中：用小counter表达“近期被命中过”，用随机探测避免顺序扫描带来的同步淘汰。

### 10.2 相比CLOCK/SIEVE

普通CLOCK/SIEVE的优势是简单和高效，但会被长扫描打出cliff。

Ran-CLOCK/Ran-SIEVE保留小状态优势，同时把确定性扫描路径改成随机路径。代价是在纯Markovian负载下，有时不如SIEVE(K)或CLOCK(K)。

### 10.3 相比RANDOM

RANDOM天然抗扫描，但完全不区分热门对象和冷对象。

Ran-CLOCK可以看成RANDOM加上小counter保护：驱逐候选是随机的，但被命中过的对象不是马上牺牲，而是先消耗counter。这样它比RANDOM更能利用频率和recency信号。

### 10.4 相比复杂自适应策略

像CACHEUS、Talus等方案也能处理复杂负载或performance cliff，但需要更多结构、分区、在线学习或控制逻辑。

Ran-CLOCK的定位更朴素：它不是最强的自适应缓存策略，而是在极低元数据和简单实现下，显著改善扫描鲁棒性。

## 11. 局限

这篇论文也有一些限制。

第一，随机探测可能降低cache locality。对CPU cache友好性、NUMA、锁竞争等工程因素，论文没有深入展开。

第二，均场模型对Ran-CLOCK/Ran-SIEVE很漂亮，但对原始CLOCK(K)这种顺序扫描策略还没有同样完整的模型。论文也把这列为future work。

第三，实验主要是block I/O trace和synthetic workload。Web cache、KV cache、数据库buffer pool等场景是否同样受益，还需要结合对象大小、写入、TTL、代价权重等因素重新评估。

第四，算法默认对象等大小。真实缓存里对象大小可能差异很大，驱逐策略通常还要考虑byte hit rate、cost、expiry、重建代价等目标。

第五，随机化缓解的是长扫描和cliff问题，不保证在所有负载下最优。论文实验里也能看到，SIEVE或CLOCK在某些trace上仍然更好。

## 12. 总结

这篇论文的贡献可以概括为三点：

1. 提出Ran-CLOCK/Ran-SIEVE(K)：把CLOCK/SIEVE的顺序驱逐扫描改为随机探测，同时把access bit推广为小counter。
2. 建立heterogeneous mean-field模型：在IRM和PH renewal workload下，固定点都可以由一个全局标量`z`刻画，并用二分法高效求解。
3. 用仿真和真实trace说明随机化能显著缓解scan-induced performance cliff，在生产trace上表现更稳定。

最关键的系统启发是：**CLOCK/SIEVE的问题不只是“状态太少”，而是顺序hand在扫描负载下会制造同步化淘汰路径。随机探测打散了这个路径，再配合少量counter，就能用很低的元数据成本换来更平滑、更鲁棒的命中率曲线。**

一句话概括：**Ran-CLOCK是给CLOCK/SIEVE加了一点随机性和一点计数器，用接近RANDOM的抗扫描能力弥补顺序扫描策略的performance cliff，同时保留对热点对象的轻量保护。**
