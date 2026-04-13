---
title: HTTP/1.1、HTTP/2 与 gRPC 原理笔记
category: [笔记]
date: 2026-04-14 00:00
tags: [网络, HTTP, gRPC, 分布式系统]
---

## 1. HTTP/1.1 vs HTTP/2

### 1.1 HTTP/1.1 的核心瓶颈

```
加载一个网页需要 HTML + 10个CSS + 20个JS：

连接1: [请求HTML ]──[响应HTML ]
连接2: [请求CSS1 ]──[响应CSS1 ]
连接3: [请求CSS2 ]──[响应CSS2 ]
...（浏览器最多同时开6个TCP连接，其余排队等待）
```

**队头阻塞（Head-of-Line Blocking）**：一个请求卡住，后面的全等待。

### 1.2 HTTP/2 核心改进

#### 多路复用（最重要）

```
HTTP/1.1（串行）:
连接: [req1──────resp1][req2──────resp2]

HTTP/2（并发）:
连接: [req1][req2][req3]  同时发出
      [resp2][resp1][resp3]  乱序返回，按 stream ID 重组
```

底层引入 **Stream** 概念，每个请求/响应是一个独立逻辑流，共用一个 TCP 连接。

#### 二进制分帧

```
HTTP/1.1（文本协议）:
GET /index.html HTTP/1.1\r\n
Host: example.com\r\n

HTTP/2（二进制协议）:
┌──────────┬────────┬─────────────────┐
│ Length   │  Type  │   Stream ID     │
├──────────┴────────┴─────────────────┤
│            Payload                  │
└─────────────────────────────────────┘
```

解析更快，是多路复用的基础。

#### Header 压缩（HPACK）

```
HTTP/1.1: 每次请求完整发送 Header（Cookie 可能几KB，每次都发）

HTTP/2 HPACK:
首次：发送完整 header，双方建立"字典"
后续：只发变化的部分，不变的用索引代替，只需几个字节
```

#### 服务器推送

客户端请求 HTML 时，服务器主动推送 CSS/JS，不用等解析后再请求。
实际使用效果不稳定，HTTP/3 已基本废弃。

### 1.3 对比总结

| | HTTP/1.1 | HTTP/2 |
|--|----------|--------|
| 协议格式 | 文本 | 二进制 |
| 连接复用 | 每个请求排队 | 单连接多路复用 |
| Header | 每次完整发送 | HPACK 压缩 |
| 队头阻塞 | 有（应用层+传输层） | 解决了应用层，TCP层仍有 |
| 服务器推送 | 无 | 有 |
| 底层协议 | TCP | TCP |

### 1.4 HTTP/2 仍未解决 → HTTP/3

TCP 传输中一个包丢失，所有后续包等待重传，HTTP/2 多路复用白费。

**HTTP/3** 将底层换成 **QUIC（UDP-based）**，彻底解决传输层队头阻塞。

---

## 2. gRPC 原理

### 2.1 整体架构

```
客户端进程                              服务端进程
┌─────────────────────┐                ┌─────────────────────┐
│  业务代码            │                │  业务代码            │
│  stub.Predict(req)  │                │  def Predict(req)   │
├─────────────────────┤                ├─────────────────────┤
│   Stub（自动生成）   │                │  Servicer（自动生成）│
├─────────────────────┤                ├─────────────────────┤
│   Protobuf 序列化   │◄──── 网络 ────►│   Protobuf 反序列化 │
├─────────────────────┤    HTTP/2      ├─────────────────────┤
│   HTTP/2 传输层     │                │   HTTP/2 传输层     │
└─────────────────────┘                └─────────────────────┘
```

**gRPC = HTTP/2 + Protobuf + 代码生成**

### 2.2 Step 1：定义 .proto 文件

```protobuf
syntax = "proto3";

service InferenceService {
    rpc Predict(PredictRequest) returns (PredictResponse);               // 一元
    rpc PredictStream(PredictRequest) returns (stream PredictResponse);  // 服务端流
}

message PredictRequest {
    string text    = 1;   // 字段编号（不是值）
    int32  max_len = 2;
}

message PredictResponse {
    string result = 1;
    float  score  = 2;
}
```

### 2.3 Step 2：Protobuf 序列化原理

只传字段编号+值，不传字段名：

```
PredictRequest { text: "hello", max_len: 128 }

序列化结果（二进制，共9字节）:
tag(字段1,字符串) | 长度5 | "hello" | tag(字段2,int) | 128(varint)

对比 JSON: {"text":"hello","max_len":128} = 28字节
```

**Varint 编码**：小数字占1字节，大数字才扩展：
```
1   → 0x01       (1字节)
127 → 0x7f       (1字节)
128 → 0x80 0x01  (2字节，最高位为1表示"还有下一字节")
```

### 2.4 Step 3：代码生成

```bash
protoc --python_out=. --grpc_python_out=. service.proto
```

自动生成：
- `service_pb2.py`：数据类，含序列化/反序列化方法
- `service_pb2_grpc.py`：客户端 Stub + 服务端 Servicer 骨架

### 2.5 Step 4：HTTP/2 传输格式

```
HTTP/2 HEADERS 帧:
  :method = POST
  :path = /InferenceService/Predict    ← 服务名/方法名
  content-type = application/grpc

HTTP/2 DATA 帧:
  ┌───┬──────────┬────────────────┐
  │ 0 │  length  │  protobuf数据  │
  └───┴──────────┴────────────────┘
  压缩标志   4字节长度    实际payload
```

### 2.6 四种通信模式

```
1. 一元 RPC
   Client: ──[req]──►
   Server: ◄──[resp]─

2. 服务端流（适合大模型流式输出）
   Client: ──[req]──────────────────────►
   Server: ◄──[token1]─[token2]─[token3]─

3. 客户端流
   Client: ──[chunk1]─[chunk2]─[chunk3]─►
   Server: ◄──────────────────[resp]─────

4. 双向流（全双工）
   Client: ──[req1]────[req2]──────────►
   Server: ◄───[resp1]────[resp2]───────
```

### 2.7 关键机制

**Channel（连接池）：**
```python
channel = grpc.insecure_channel('localhost:50051')
# 多个 RPC 调用复用同一个 TCP 连接（HTTP/2 多路复用）
stub = InferenceServiceStub(channel)
```

**拦截器（中间件）：**
```python
class AuthInterceptor(grpc.UnaryUnaryClientInterceptor):
    def intercept_unary_unary(self, continuation, client_call_details, request):
        metadata = [('authorization', 'Bearer xxx')]
        return continuation(client_call_details._replace(metadata=metadata), request)
```

**Deadline（超时传播）：**
```python
stub.Predict(req, timeout=5.0)  # 整个调用链共享这5秒
```

### 2.8 与 REST 的本质区别

```
REST 思维：操作资源
  POST /predictions  {"text": "hello"}

gRPC 思维：调用函数（RPC = Remote Procedure Call）
  stub.Predict(PredictRequest(text="hello"))
  # 感觉就是本地函数调用，网络细节全部隐藏
```

### 2.9 为什么 gRPC 要求 HTTP/2

- **多路复用**：一个连接并发多个 RPC 调用
- **双向流**：客户端/服务端可同时持续发送数据
- **二进制帧**：与 Protobuf 配合，全程二进制，高效
