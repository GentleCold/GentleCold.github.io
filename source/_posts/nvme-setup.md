---
title: 从NVMe磁盘安装到GDS支持
category: [Linux]
date: 2026-04-10 23:23
tags: [Linux, nvme]
---

### 1. 安装前准备

首先检查哪些PCIe插槽是空的：

`sudo dmidecode -t slot`

大概确定好要插的位置

服务器关机、断电、将NVMe插入PCIe插槽

### 2. 磁盘初始化

lspci检查安装是否被识别：

<p align="center">
    <img src="/imgs/image-20260410232502.png"/>
</p>

可见其型号为：Intel Corporation NVMe Datacenter SSD [Optane]，与GPU0最接近

安装对应工具Intel mas工具初始化：https://www.intel.cn/content/www/cn/zh/download/19520/intel-memory-and-storage-tool-cli-command-line-interface.html

### 3. GDS驱动安装

阅读官方文档：https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html#doca-requirements-and-installation

需要先安装mofed模块：https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/

```bash
./mlnxofedinstall --with-nvmf --with-nfsrdma --enable-gds --add-kernel-support --without-ucx-cuda
sudo update-initramfs -u -k `uname -r`
reboot
```

或者安装doca模块：https://developer.nvidia.com/doca-downloads

```bash
sudo apt install doca-ofed mlnx-fw-updater mlnx-nvme-dkms
```

之后尝试通过cuda toolkit安装：https://developer.nvidia.com/cuda-toolkit-archive
取消勾选驱动安装（已经安装过了），勾选nvidia-fs

检查：

```bash
modprobe nvidia-fs
modinfo nvidia_fs | grep version
/usr/local/cuda/gds/tools/gdscheck -p
/usr/local/cuda/gds/tools/gdsio -f /data/gds/test -d 4 -w 1 -s 4G -i 1M -x 0 -I 0
/usr/local/cuda/gds/tools/gdsio -f /data/gds/test -d 0 -w 1 -s 4G -i 1M -x 0 -I 1
cat cufile.log
```

其他注意事项：

- 禁用iommu：intel_iommu=off
- 禁用ACS
- hostnamectl检查是金属机还是云主机

拓扑检查：

```bash
Nvidia-smi topo
lspci -tv | egrep -i "nvidia | nvme"
```

状态检查：

```bash
cat /proc/driver/nvidia-fs/peer_distance
cat /proc/driver/nvidia-fs/stats
```

#### 4. 性能测试

普通NVMe测试（注意ioengine选择libaio或者io_uring来异步打满nvme带宽）：

```bash
fio --filename=/data/fio_test --direct=1 --rw=read \
    --bs=1M --size=10G --numjobs=4 --iodepth=32 \
    --ioengine=libaio --runtime=30 --time_based \
    --group_reporting --name=btrfs_read_real
```

结果：

```bashk
btrfs_read_real: (g=0): rw=read, bs=(R) 1024KiB-1024KiB, (W) 1024KiB-1024KiB, (T) 1024KiB-1024KiB, ioengine=libaio, iodepth=32
...
fio-3.28
Starting 4 processes
Jobs: 4 (f=4): [R(4)][100.0%][r=23.3GiB/s][r=23.9k IOPS][eta 00m:00s]
btrfs_read_real: (groupid=0, jobs=4): err= 0: pid=14479: Fri Apr 10 23:42:21 2026
  read: IOPS=23.6k, BW=23.0GiB/s (24.7GB/s)(691GiB/30005msec)
    slat (usec): min=20, max=13644, avg=35.33, stdev=109.14
    clat (usec): min=421, max=66318, avg=5391.00, stdev=2679.83
     lat (usec): min=450, max=67316, avg=5426.41, stdev=2697.45
    clat percentiles (usec):
     |  1.00th=[ 1532],  5.00th=[ 2671], 10.00th=[ 3228], 20.00th=[ 3785],
     | 30.00th=[ 4178], 40.00th=[ 4490], 50.00th=[ 4817], 60.00th=[ 5145],
     | 70.00th=[ 5604], 80.00th=[ 6325], 90.00th=[ 8225], 95.00th=[10421],
     | 99.00th=[15664], 99.50th=[17695], 99.90th=[23462], 99.95th=[28705],
     | 99.99th=[52167]
   bw (  MiB/s): min= 9614, max=26214, per=100.00%, avg=23578.92, stdev=916.56, samples=239
   iops        : min= 9614, max=26214, avg=23578.92, stdev=916.56, samples=239
  lat (usec)   : 500=0.01%, 750=0.12%, 1000=0.20%
  lat (msec)   : 2=1.72%, 4=22.95%, 10=69.32%, 20=5.44%, 50=0.22%
  lat (msec)   : 100=0.01%
  cpu          : usr=0.59%, sys=19.81%, ctx=542220, majf=0, minf=32916
  IO depths    : 1=0.1%, 2=0.1%, 4=0.1%, 8=0.1%, 16=0.1%, 32=100.0%, >=64=0.0%
     submit    : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.0%, 64=0.0%, >=64=0.0%
     complete  : 0=0.0%, 4=100.0%, 8=0.0%, 16=0.0%, 32=0.1%, 64=0.0%, >=64=0.0%
     issued rwts: total=707382,0,0,0 short=0,0,0,0 dropped=0,0,0,0
     latency   : target=0, window=0, percentile=100.00%, depth=32

Run status group 0 (all jobs):
   READ: bw=23.0GiB/s (24.7GB/s), 23.0GiB/s-23.0GiB/s (24.7GB/s-24.7GB/s), io=691GiB (742GB), run=30005-30005msec
```
