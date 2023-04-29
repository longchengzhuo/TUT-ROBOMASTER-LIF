# 天津理工大学ROBOMASTER_LIF战队视觉proj.

<img src="https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/blob/main/docs/0.png" width="230px">

**★★★RMer    NEVER    GIVE    UP★★★**



## Overview

这是一个使用纯传统视觉，**面向过程编程，对新手友好**，极适合教学使用的python proj.

**Keywords:** Robomaster, auto-aim, opencv, python



### License

The source code is released under a [MIT license](rm_auto_aim/LICENSE).
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Author: Long chengzhuo**

**Maintainer: Long chengzhuo**
jacklongcn@vip.qq.com

The proj. has been tested with **jetson nx** on Ubuntu 18.04.


## Dependencies

- numpy 1.19.5

- opencv-contrib-python 3.3.0.10

- matplotlib 3.3.0



## Update record

* V0.1 BETA 2022.8.2 
    > 纯传统视觉自瞄部分
    
    > 卡尔曼滤波器

* V0.5 BETA 2023.4.19 
    > 优化代码结构，发挥py优势，更加简洁
    
    > 传统视觉灯条识别优化 （放弃hsv转向灰度图直接识别）
    
    > tensorrt加速残差网络数字识别推理

### Outlook for recent updates
* V0.51 BETA 2023.6.1
    > 优化识别速度
    
    > 优化卡尔曼与电控的衔接

* 年底前更新端到端神经网络版本

## Bugs & Feature Requests

Please report bugs and request features using the [Issue Tracker](https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/issues).

