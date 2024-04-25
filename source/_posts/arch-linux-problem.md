---
title: ArchLinux使用问题汇总
category: [Linux]
date: 2024-04-25 09:54
tags: [Linux, Btrfs]
---

> 本文用以记录本人使用ArchLinux时遇到的问题，以备参考。

# btrfs文件系统metadata过大

在一次日常更新后，发现可用空间不足

以为仅仅是好久没清理快照的问题，遂删掉一大部分快照

使用`df -h`命令后发现，Size为256G，Used为77G，Avail确只有30G不到？

查阅资料发现df命令无法准确显示btrfs文件系统的占用

应使用命令：

```shell
sudo btrfs filesystem usage -T /
```

然后发现罪魁祸首：

```shell
                  Data     Metadata  System
Id Path           single   DUP       DUP      Unallocated Total     Slack
-- -------------- -------- --------- -------- ----------- --------- -----
 1 /dev/nvme0n1p7 91.01GiB 132.00GiB 16.00MiB    32.97GiB 256.00GiB     -
-- -------------- -------- --------- -------- ----------- --------- -----
   Total          91.01GiB  66.00GiB  8.00MiB    32.97GiB 256.00GiB 0.00B
   Used           72.74GiB   1.82GiB 48.00KiB
```

Metadata虚高啊，分配了132G的metadata但是只用了2G不到

猜测是快照问题，清理了快照但是没有释放分配空间

使用balance命令手动释放：

```shell
sudo btrfs balance start -v -musage=5 /
```

其中5表示压缩占用率小于5%的块，重新查看占用：

```shell
                  Data     Metadata System
Id Path           single   DUP      DUP      Unallocated Total     Slack
-- -------------- -------- -------- -------- ----------- --------- -----
 1 /dev/nvme0n1p7 91.01GiB 18.00GiB 64.00MiB   146.93GiB 256.00GiB     -
-- -------------- -------- -------- -------- ----------- --------- -----
   Total          91.01GiB  9.00GiB 32.00MiB   146.93GiB 256.00GiB 0.00B
   Used           72.74GiB  1.82GiB 16.00KiB
```

问题暂时解决

参考：

- https://wiki.tnonline.net/w/Btrfs/Balance
- https://www.reddit.com/r/archlinux/comments/1mavr4/why_is_btrfs_metadata_so_large/
- https://superuser.com/questions/654119/btrfs-huge-metadata-allocated

# 使用集成显卡时，无法通过Lutris启动游戏

偶然发现Lutris可以管理所有平台的游戏，并且可以通过wine在Linux下运行Windows游戏

但是使用集成显卡时，无法通过Lutris启动游戏。由于系统既有核显驱动又有独显驱动，Lutris应该是识别到了独显驱动，但是独显驱动没加载，所以就寄了

使用命令`ls /usr/share/vulkan/icd.d/`查看输出：

```shell
 intel_hasvk_icd.i686.json   intel_hasvk_icd.x86_64.json   intel_icd.i686.json   intel_icd.x86_64.json {} nvidia.json
```

找到此目录，将nvidia.json移走即可（可能会影响到独显，之后更换驱动时要注意移回来）

更换驱动命令为：

```shell
optimus-manager --switch integrated/nvidia
```

# 更新过程中突然退出到窗口管理器界面

有时候滚动更新，突然退出到窗口管理器界面，此时一定不能关闭或重启电脑，可能是内核构建一半失败导致的，关电脑就寄了

此时一定要先把pacman的进程锁删了，然后重新下载linux（或者别的内核版本）。此时会重新构建内核，一般会成功

只要保证内核不挂，一般更新不会有啥问题

除非供应链被污染或重大软件bug，此时直接回退版本
