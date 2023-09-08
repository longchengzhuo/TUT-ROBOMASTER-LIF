# 天津理工大学ROBOMASTER_LIF战队视觉proj.

<img src="https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/blob/main/docs/0.png" width="230px">

**★★★RMer    NEVER    GIVE    UP★★★**



## Overview

**Keywords:** Robomaster, auto-aim, opencv, python



### License

The source code is released under a [MIT license](https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/blob/main/LICENSE).
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Author: Long chengzhuo**

**Maintainer: Long chengzhuo**
jacklongcn@vip.qq.com

The proj. has been tested with **jetson nx** on Ubuntu 18.04.


## Dependencies

- python 3.6.9

- onnxruntime 1.9.0

- CUDA 10.2.3

- numpy 1.19.5

- opencv 4.1.1

- TensorRT 8.0.1.6

- cuDNN 8.2.1.32

- matplotlib 3.3.0



## Packages

* [train_and_transONNXtoTRT](https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/blob/main/train_and_transONNXtoTRT.ipynb)
    > 使用colab进行训练

* [demo for verification](https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/tree/main/demo%20for%20verification)
    > 用于调参以及验证onnx, trt推理的小工具
    
    > 推理**支持动态batch_size!!**
    
* [V0.1 BETA 2022.8.2](https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/tree/main/V0.1%20BETA%202022.8.2) 
这是一个使用纯传统视觉，**面向过程编程，对新手友好**，极适合教学使用的python proj..
    > 纯传统视觉自瞄部分
    
    > 卡尔曼滤波器

* [V0.5 BETA 2023.9.2](https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/tree/main/V0.5%20BETA%202023.9.2) 离线视频测试版
    > 使用拓展卡尔曼滤波
    
    > 优化代码结构，发挥py优势，更加简洁
    
    > 缩小输入图像尺寸，提升帧率（nx上目前40fps）

    > 将程序解耦为图像处理层和坐标滤波串口发送层，两个独立进程提升cpu效率，提高帧率
    
    > 成立独立的调参空间，直接在yaml调参，更方便
    
    > tensorrt加速残差网络数字识别推理

### Outlook for recent updates
* [V0.51 BETA 2023.9.？（待定）](https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/tree/main/V0.51%20BETA%202023.9.%EF%BC%9F%EF%BC%88%E5%BE%85%E5%AE%9A%EF%BC%89) 实时读取正式版
    > 找到更合适的gamma和增益参数，提升神经网络推理质量
    
    > 优化卡尔曼与电控的衔接

* 年底前更新端到端神经网络版本

## Bugs & Feature Requests

Please report bugs and request features using the [Issue Tracker](https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/issues).

