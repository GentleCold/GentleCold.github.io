---
title: GO学习笔记
category: [笔记]
date: 2024-04-02 13:13
tags: [笔记, GO]
---

浅学Golang，为分布式的学习做准备

参考书籍：

- The Go programming language

# Go基础

## 1. Neovim环境搭建

- pacman 安装 goland
- maven 添加 `gopls`
- formatter 添加 `gofmt`
- treesitter 添加 go
- dap 添加 `delve`

## 2. Golang基础

- gofmt 没有参数，强制格式统一，缩进为制表符
- 变量会隐式初始化
- `i++`是语句而不是表达式，`++i`非法
- `:=` 是声明，`=` 是赋值
- 每一次对变量的取地址/复制地址，都是为原变量创建了别名（用以垃圾回收
- `new` 函数只是语法糖，不用创建临时变量名
- 如果new两个类型都是空的，也就是说类型的大小是0，例如`struct{}`和`[0]int`，有可能有相同的地址
- 分配在堆上还是栈上由go决定(根据生命周期)
- 数值常量是高精度的值
- if的条件表达式前也可执行语句
- `switch`不需要`break`
- `defer`语句会将函数推迟到外层函数返回之后执行，被推迟的函数被压入栈中
- 切片实际是数组的引用，小心切片的引用导致整个数组无法被回收
- 切片有长度和容量的概念
- Go没有类，但是可以为struct或为非结构体类型定义方法，但是方法只是个带接受者参数的函数（也就是说将参数改为指针才能修改原类型值

```go
type Vertex struct {
	X, Y float64
}

func (v Vertex) Abs() float64 {
	return math.Sqrt(v.X*v.X + v.Y*v.Y)
}
```

- interface 实现类似动态绑定的功能
- 类型断言与类型选择：

```go
v := i.(T)
switch v := i.(type) {
case T:
    // v 的类型为 T
case S:
    // v 的类型为 S
default:
    // 没有匹配，v 与 i 的类型相同
}
```

- 接口示例：Stringer/Error
- go/<-/close/select/mux
