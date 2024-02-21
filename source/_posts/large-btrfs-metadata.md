---
title: 记一次btrfs文件系统metadata过大的问题
category: [Linux]
date: 2024-02-21 23:12
tags: [Linux, Btrfs]
---

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
