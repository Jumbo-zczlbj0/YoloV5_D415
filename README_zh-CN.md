# Yolo-Object-Detection-With-Intel-Realsense-Camera

## 安装
### 1.cuda && pytorch

在安装之前你需要明确一件非常重要的事：显卡型号、显卡驱动版本、cuda版本和python版本要一一对应。

#### 一.安装驱动（仅限NVIDIA显卡，AMD的显卡没办法跑深度学习，苹果..刚刚发布了自己的架构，你可以尝试但非常不建议，毕竟同学们只是为了完成作业）
在命令行输入ubuntu-drivers devices 可以电脑查看电脑能安装的显卡驱动版本，然后安装适合你的驱动：sudo apt install nvidia-driver-xxx。上述步骤有图形化界面，在应用程序找到“软件与更新”或者是其他的名字（由于作者使用的是繁体系统，不知道确切的名字）。选择“额外的驱动程序”，选择你想要安装的驱动版本。 无论你采用哪种安装方式，请务必重启电脑！！

验证安装：nvidia-smi

强烈建议直接通过ubuntu安装显卡驱动，当然你也可以通过cuda安装或者自己从官网下载安装包，只是有时候会报错。这种方式几乎不会有bug，为节省时间

#### 二.安装cuda

根据你的“显卡驱动版本”安装cuda。下面的网站可以找到cuda和驱动版本的对应关系。https://docs.nvidia.com/cuda/cuda-toolkit-release-notes。在 Table 3 CUDA Toolkit and Corresponding Driver Versions。逻辑上来说高版本的显卡驱动可以适配低版本的cuda，安装前请务必考虑清楚，因为报错你可能修不好，或者浪费你很长时间。

下面是下载cuda从的连接https://developer.nvidia.com/cuda-downloads。ubuntu是linux的一种，一般电脑都是X86。接下来选择安装方式，强烈建议选择最后一种“runfile”。因为别的容易报错，第二种安装方式对网络质量需求高，不建议使用。第一种安装方式的：wget（wget是一个从网络上自动下载文件的工具） 下载好的文件在/home/<你的用户名>。安装过程中，会让你选择安装的程序包括什么，请把“显卡驱动”取消。因为我们刚刚已经安装过。如果你需要其他的cuda版本请搜索：Archive of Previous CUDA Releases

#### 三.安装pytorch

pytorch安装之前通常会先安装conda，我比较喜欢miniconda：https://docs.conda.io/projects/miniconda/en/latest/index.html 可以找到你需要的版本。然后在“下载”文件夹开启Terminal 输入bash xxx.sh。安装过程在你可以一直选择同意，在每次激活conda环境前需要source ~/miniconda3/bin/activate。创建环境：conda create --name 你喜欢的名字（英文）之后激活环境conda activate 刚刚的名字。你也可以选择在ubuntu的python3环境安装各种包，只不过有报错的时候你要自己处理，包冲突的时候，，冲突能怎么办。。不用了吗emm。conda的环境如果装坏了，直接remove：conda remove -n 环境的名字 --all。逻辑上来说不同环境之间不会发生冲突，但实际上还是有几率发生，发生的时候重装环境一般就好了。

在你创建好的环境安装pytorch，之后：pip install ipython, 命令行输入 ipython，再输入：

import torch
torch.cuda.is_available()

为什么要安装ipython，，纯属个人习惯，你也可以在命令行输入python3，再粘贴。如果返回True就可以继续了，如果不是，那就寻求帮助。

### 2.安装 Intel RealSense SDK 2.0

这个是Intel相机的驱动，我使用的是D415，别的相机也支持。在他的官网安装就好：https://www.intelrealsense.com/sdk-2/。

Jetson Nano：https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md

ubuntu直接安装官网安装，Jetson Nano我就不清楚了这边有bug，可能是缺少供电导致的。pip install pyrealsense2是安装在conda环境中的。

验证安装：realsense-viewer

这里注意以下，插入摄像头之后，是不是USB3.0，如果是2.0请重新插入。按照USB的识别逻辑，如果你插入太慢，可能会被误识别成USB 2.0。


### 4.安装其他的包

你也可以直接运行程序，反正会有报错提示emm

顺便说一下，在ubuntu 20.04允许python2 和 python3 共存。在安装的时候，pip是安装在python2，pip3是安装在python3.但这个不是绝对的。

> pip install pyttsx3

> pip install threading

## 运行程序

> python pyrealsense2_camera.py


## 训练YOLOV5

看官方文档比较好，早晚都要学会看。

> python train.py --img 640 --batch 64 --epochs 300 --data coco128.yaml --weights yolov5s.pt --cache

# 移植到Jeston nano

下面的链接是一个大佬做的开源img，已经安装好了cuda和pytorch，当然你也可以想不开去安装官方版本

Tutorials URL: https://github.com/Qengineering/Jetson-Nano-Ubuntu-20-image

BalenaEtcher: https://etcher.balena.io/#download-etcher


1.Get a 32 GB (minimal) SD card with exFat to hold the image.

2.Download the image JetsonNanoUb20_3b.img.xz (8.7 GByte!) from our Sync.

3.Flash the image on the SD card with the Imager or balenaEtcher.

4.nsert the SD card in your Jetson Nano and enjoy.

Password: jetson

（The above content comes from Qengineering： https://github.com/Qengineering/Jetson-Nano-Ubuntu-20-image）

## 这里的安装只针对Jeston
### 1.安装 exfat，这是一个帮助系统读取U盘的工具，我觉得你们会用到
> sudo apt-get install exfat-fuse exfat-utils

### 2.安装 pip：安装pip之后安装各种python的包就很容易了，除非你想一个一个下载源码
> Download: https://bootstrap.pypa.io/get-pip.py 

> cd Downloads

> python3 get-pip.py

# ROS

这东西太难装了，，安装链接：https://wiki.ros.org/noetic/Installation/Ubuntu。主要原因是被墙了，要不然就科学上网，要不然就换源。比如说清华/阿里/中科大，但问题是我换源之后还是会有各种各样的问题，比如说找不到包emmm，这边没有建议，建议直接问老师。
