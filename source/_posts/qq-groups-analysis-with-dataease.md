---
title: 基于DataEase的QQ群数据分析
category: [实验报告, 云计算系统]
date: 2024-1-21 14:27
tags: [gocqhttp, dataease, qq, 数据分析]
---

<h1 align="center">基于DataEase的QQ群数据分析</h1>

<div style="text-align: center;">
  <a href="https://dataease.io" style="display: inline-block;">
    <img src="https://img.shields.io/badge/DataEase-DATAEASE-brightgreen.svg" alt="DataEase">
  </a>
  <a href="https://www.ucloud.cn/" style="display: inline-block;">
    <img src="https://img.shields.io/badge/uCloud-UCLOUD-blue.svg" alt="uCloud">
  </a>
  <a href="https://docs.go-cqhttp.org/" style="display: inline-block;">
    <img src="https://img.shields.io/badge/gocqhttp-GOCQHTTP-purple.svg" alt="uCloud">
  </a>
</div>

> 云计算系统期末设计

## 项目介绍

本次项目所选赛道为 数据分析与可视化

使用从个人QQ群采集的数据进行分析，包括聊天记录、群成员信息等，将数据存储于云数据库服务，并使用DataEase网页版进行数据分析可视化呈现

分享的知乎链接：

https://zhuanlan.zhihu.com/p/679053445

考虑到本次项目数据的隐私性，并未录制视频，也并未将源码上传至Github

## 项目文件

```shell
.
├── cut_word.py    // 用以分词
├── db.py          // 用以创建数据表
├── gocqhttp       // 仿QQ客户端
├── imgs
├── README.md      // 项目报告
├── README.pdf     // 项目报告
├── record.py      // 用以记录QQ群数据
└── stopwords.txt  // 中文停用词，用以分词
```

## 项目过程

### 一、准备数据集

#### 1. 如何获取QQ群数据

由于QQ官方并未提供API（个人开发认证仅提供频道API），因此需要考虑获取QQ群数据的方式

经过多次探索，可能的获取方式有以下几种：

- 通过模拟点击获取聊天记录
- 通过QQ自带的聊天记录导出功能获取
- 使用HOOK技术拦截QQ内部函数
- 使用第三方QQ客户端

考虑到获取消息的方便性，我们使用最后一种方法，这里选择了[go-cqhttp](https://docs.go-cqhttp.org/)

go-cqhttp原本的目的是为QQ机器人服务的，它通过模拟一个QQ客户端，接受QQ服务器发来的消息，同时将API引出用以服务机器人，而对于我们项目来说，我们可以利用它的API来获取消息

在阅读相关文档后，我们使用反向websocket的方式，即通过python创建一个websocket服务端，然后通过go-cqhttp登陆个人QQ，登陆后它会作为客户端连接我们创建的服务端，并且实时上报接受到的消息，这样，我们就可以通过python代码处理我们接受到的消息

```python
async def server(websocket, _):
    print("Server is connected...")

    # Get group list and members
    await websocket.send(json.dumps({"action": "get_group_list", "echo": "groups"}))

    while True:
        raw = await websocket.recv()
        await handle_response(websocket, raw)

start_server = websockets.serve(server, ip, 2024)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
```

除了消息之外，我们可以通过go-cqhttp提供的API获取其他信息，如群成员信息等。另外由于本次项目基本使用云的环境，所以仅仅考虑群消息的获取而不考虑私聊消息

> 令人遗憾的是，作者已经放弃了对go-cqhttp的维护（由于QQ官方针对协议库的围追堵截, 不断更新加密方案, 我们已无力继续维护此项目.），虽然目前此方案仍然可行，但是已经不能保证未来的稳定使用。但是，QQNT的出现(Electron版QQ)让第三种方案更易于实现

#### 2. 连接云数据库

在解决获取消息的问题后，我们要考虑如何存储消息。最常用的办法当然是使用数据库，另外考虑到DataEase接入数据源需要云数据库，因此我们使用Ucloud来提供云数据服务，并选择Mysql数据库。

我们首先租用一个Mysql数据库，但是其IP是内网隔离的，因此我们还需要租用一个弹性网络。我们通过端口转发的方式将SQl服务暴露给外网：

<p align="center">
<img src="/imgs/image-20240121204845.png"/>
</p>

#### 3. 创建数据表

之后，我们便可以组织数据表的结构。

我们使用四张数据表，结构定义如下（python代码中使用ORM的方式操纵数据库）

```python
class User(Base):
    __tablename__ = "user"

    user_id = Column(BigInteger, primary_key=True)  # QQ ID
    group_id = Column(Integer, primary_key=True)
    name = Column(String(100))
    card = Column(String(100))
    sex = Column(String(10))
    age = Column(Integer)
    level = Column(Integer)
    join_time = Column(Integer)
    last_sent_time = Column(Integer)


class Group(Base):
    __tablename__ = "group"

    group_id = Column(Integer, primary_key=True)  # Group ID
    group_name = Column(String(100))
    group_create_time = Column(Integer)
    group_level = Column(Integer)
    member_count = Column(Integer)


class Message(Base):
    __tablename__ = "message"

    message_id = Column(Integer, primary_key=True)
    time = Column(Integer)
    user_id = Column(BigInteger)
    group_id = Column(Integer)
    raw_message = Column(String(5000))
    anonymous = Column(Boolean)
    emotion = Column(Float)


class Word(Base):
    __tablename__ = "word"

    user_id = Column(BigInteger, primary_key=True)
    group_id = Column(Integer, primary_key=True)
    word = Column(String(50), primary_key=True)
    count = Column(Integer)
```

分别对应群成员信息、群信息、群消息和群消息分词词频统计

#### 4. 记录数据

由于go-cqhttp并未提供获取群所有历史消息的API，因此我们需要在我们的程序中实时记录群消息。

在创建的websocket服务端中，当客户端连接后，我们即向其发送获取所有群信息，以及所有群成员信息的API，之后，我们循环处理收到的每一条上报信息，根据上报类型，将数据存储于不同的数据表，另外，我们还针对消息利用SnowNLP获取其情感值，一并存储于数据表中（代码详见record.py）

另外为了后续方便进行词云图展示，我们将所有消息利用jieba库进行分词，并将分词结果进行词频统计，然后存储于word数据表中（代码详见cut_word.py）

最后还需注意，我们需要将数据库的编码格式改为utf8mb4，以适应群成员昵称

#### 5. 部署

我们首先运行record.py代码，然后运行go-cqhttp客户端，即可实时将消息记录到云数据库中，记录过程如下：

<p align="center">
<img src="/imgs/image-20240121210511.png"/>
</p>

当然我们也可以将其部署到云主机上，从而能够持续不断稳定的记录消息。由于本人使用linux操作系统，将整个过程编写为服务，本地仍然可以保证长时间的部署

### 二、数据可视化

我们通过DataEase进行数据的可视化分析，首先添加数据源：

<p align="center">
<img src="/imgs/image-20240121210716.png"/>
</p>

然后创建仪表盘和数据集

之后我们便可以根据自己需要的数据表形式，进行sql查询：

<p align="center">
<img src="/imgs/image-20240121211118.png"/>
</p>

然后在仪表盘中根据sql查询的数据集创建相应的视图

## 项目结果

项目最终制作的仪表盘如下：

<p align="center">
<img src="/imgs/image-20240121211228.png"/>
</p>

其中最上方的三个表属于全局分析（群人数排名、群发言数排名、发言数与人数关系），可以宏观展示自己所加的所有群的活跃度

而在关系表散点图中，左上方的点表示尤其活跃（群人数少，群发言数多），而右下方的点则表示不活跃的群

接着，在左边放置了一个文本下拉框组建，我们可以选择一个具体的群：

<p align="center">
<img src="/imgs/image-20240121211740.png"/>
</p>

之后可以看到根据所选择的群，呈现出的群成员发言排名、群成员性别分布以及群消息词云图，这样可以帮助我们查看群中活跃的成员、性别比例、讨论消息的主要方向

而在仪表盘最中间的部分，放置了群发言总数与平均情感值随时间的变化，最上方呈现的是最新的时间，通过此图我们可以看到群活跃度随时间的变化以及情绪值的变化：

<p align="center">
<img src="/imgs/image-20240121212505.png"/>
</p>

从图中我们可以看到本人所加的群中普遍在夜晚的消息数量较多

而在右下角，放置了一个可以选择所有群成员的文本下拉框：

<p align="center">
<img src="/imgs/image-20240121212632.png"/>
</p>

我们可以选择一个具体的群成员，然后查看他发言的消息以及情绪值随时间的变化，以了解他发言活跃度的变化，以及在一天中何时最活跃，另外还有他所发消息的词云图，以了解他的发言的关注点

比如对于图中这位用户，我们就发现他在凌晨时发消息较多，并且情绪值也较高，就可以推断他属于夜猫子类型，而从词云图中，我们也可以推断他与初中的关系较大

## 项目总结

### 一、功能完备性

我们通过部署websocket服务端与gocqhttp客户端，持续记录QQ群消息至数据库，然后通过DataEase制作了能够实时更新的仪表盘，从而快速、实时了解自己的群聊信息

整个仪表盘刚好可以平铺1080P分辨率的整个网页，通过这些图表可以迅速了解自己所有群、单个群、群成员的活跃状态、聊天的关注点

### 二、技术难点

难点之一便是如何获取QQ群消息，而项目选择使用gocqhttp客户端提供的API来持续记录消息

其次便是云数据库的搭建以及与DataEase的对接

最后便是较为复杂的SQL编写，以适应不同的图表创建

### 三、创新性

#### 1. 实时性

我们通过将项目部署到云主机和云数据库上，并且接入网站前端，可以持续不断的记录数据，而前端的数据可视化也是实时更新的

#### 2. 普适性

由于项目数据属于个人QQ群，分析的结果对个人更有价值，所以项目应当具备普适性，即可以方便的为所有人提供属于自己的QQ群数据可视化与分析

而实际上这也是易于实现的，而需要考虑的问题仅为隐私性问题

首先用户可以自行提供一个可以连接的云数据库，然后通过项目的代码与gocqhttp客户端将数据上传，之后可以套用本次项目创建的仪表盘模板，从而获得属于自己的QQ群数据可视化

更进一步，我们可以搭建一个云服务，只要用户提供QQ账号，以及可选的云服务器，我们就可以返还仪表盘链接，用户即可看到属于自己的QQ群数据分析
