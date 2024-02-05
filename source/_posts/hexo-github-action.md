---
title: 利用Github Action部署Hexo博客
category: [博客搭建]
date: 2024-02-05 19:11
tags: [Hexo, Github]
---

> 涉及源码见 https://github.com/GentleCold/GentleCold.github.io

## 介绍

使用[ Hexo ](https://hexo.io)作为搭建博客的框架，流程如下：

```shell
npm install -g hexo-cli // 安装Hexo
hexo init blog          // 初始化博客
cd blog
npm install             // 安装依赖
hexo server
```

我们只需要关注以下文件：

```shell
.
├── public
├── source
│   └── _posts
└── _config.yml
```

- public: 生成的整个静态网站（index.html作为入口）
- source: 源文件 (此文件夹下内容会被复制到public目录下)
  - \_posts: 博客内容 (.md文件，被转换为对应的文章)
- \_config.yml: 配置文件

一些常用命令：

```shell
hexo g     // 生成静态网站(public/)
hexo clean // 清除缓存
hexo s     // 启动服务
```

## 部署方案选择

创建名为yourname.github.io的仓库，并设置可见性为public(也可以private，但是要加钱)

Github Pages 提供了两种部署方案：

<p align="center">
    <img src="/imgs/image-20240206004221.png"/>
</p>

对于第一种，需要编写workflows

对于第二种，只需将public目录下的内容放入即可

现考虑几种**一键部署**方案：

- 利用插件部署[hexo-deployer-git](https://github.com/hexojs/hexo-deployer-git)
  - ✅ 可以额外将Hexo配置文件上传并设为私密，以供多端部署
  - ❎ 需要提供Github token
- 再创建一个仓库，将Hexo文件上传（不包含public），并编写workflows，对于每次commit，将生成的public文件推送到pages仓库下
  - ✅ 可以将Hexo配置文件上传并设为私密
  - ❎ 需要提供Github token
- 将源改为Github Action，直接将Hexo文件上传pages仓库并触发action部署
  - ✅ 无需token
  - ❎ 无法设为私密

相对来说，更推荐第一种和第三种（第二种其实重复造轮了

最终笔者选择了第三种：

- **真**一键部署：一次commit，既保存了配置文件，又部署了网站
- 私密性问题：本身就是静态网站，用户可以看到所有内容，只不过这种方案还可以看到Hexo的配置内容，会产生配置中需要填写API token的问题(之后给出解决方法)

## 编写workflows

官方文档其实给出示例了，不过那个文档有些老（甚至触发条件还是main分支

修改如下：

```yml
name: Pages

on:
  push:
    branches:
      - master

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@main
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
          # If your repository depends on submodule, please see: https://github.com/actions/checkout
          submodules: recursive
      - name: Use Node.js 21.6.1
        uses: actions/setup-node@main
        with:
          node-version: '21'
      - name: Cache NPM dependencies
        uses: actions/cache@main
        with:
          path: node_modules
          key: ${{ runner.OS }}-npm-cache
          restore-keys: |
            ${{ runner.OS }}-npm-cache
      - name: Install Dependencies
        run: npm install
      - name: Build
        run: npm run build
      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@main
        with:
          path: ./public
  deploy:
    needs: build
    permissions:
      pages: write
      id-token: write
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@main
```

#### 解决token隐私性问题

问题：此方案无法将仓库设为私密，而配置文件中又需要填写API token

workflows的流程其实就是在一个linux环境中，先把仓库代码拉下来，然后生成public文件然后部署，我们只需在需要填写token的地方用占位符标记，在workflows中添加一步，利用sed命令进行替换即可：

```yml
- name: Replace variables in _config.fluid.yml
run: |
    sed -i 's/APP_ID/${{ secrets.APP_ID }}/g' _config.fluid.yml
    sed -i 's/APP_KEY/${{ secrets.APP_KEY }}/g' _config.fluid.yml
    sed -i 's/APP_SERVER/${{ secrets.APP_SERVER }}/g' _config.fluid.yml
    sed -i 's/BAIDU/${{ secrets.BAIDU }}/g' _config.fluid.yml
    sed -i 's/GOOGLE/${{ secrets.GOOGLE }}/g' _config.fluid.yml
```

## 结

配置好后只要commit到master分支就可以直接部署啦，而且支持多端部署，如果换了设备只要把这个仓库拉下来就能继续编写啦

<p align="center">
    <img src="/imgs/image-20240206012154.png"/>
</p>
