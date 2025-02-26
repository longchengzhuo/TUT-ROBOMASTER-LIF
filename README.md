# ROBOMASTER_LIF_robotvision


**★★★RMer    NEVER    GIVE    UP★★★**

This project aims to construct a complete automatic aiming system for RoboMaster events using pure Python and addresses the fundamental issue of "reinventing the wheel" for newcomers in the team.（including serial communication with the electronic control system）

<img src="https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/blob/main/docs/real_test.gif" width="600px">

## Overview

**Keywords:** Robomaster, auto-aim, opencv, python



### License

The source code is released under a [MIT license](https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/blob/main/LICENSE).
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Maintainer: Chengzhuo Long**
chengzhuolong@outlook.com

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
    > Training with Colab

* [demo for verification](https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/tree/main/demo%20for%20verification)
    > A small tool for tuning and verifying ONNX, TRT inference
    
    > Inference **supports dynamic batch_size!!**
    
* [V0.1 BETA 2022.8.2](https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/tree/main/V0.1%20BETA%202022.8.2) 
The readme of this project contains **demonstration videos of the actual robot auto-aim system**. Open this project to view them.
    > Pure Traditional Vision Auto-Aim Part
    
    > Kalman Filter

* [V0.5 BETA 2023.9.2](https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/tree/main/V0.5%20BETA%202023.9.2) Offline Video Test Version
    > Using Extended Kalman Filter
    
    > Optimized code structure, leveraging Python's advantages for greater simplicity
    
    > Reducing input image size to increase frame rate (currently 40fps on NX)

    > Decoupling the program into an image processing layer and a coordinate filtering serial port transmission layer, two independent processes to improve CPU efficiency and frame rate
    
    > Establishing an independent tuning space, tuning directly in YAML for convenience
    
    > TensorRT accelerated residual network digital recognition inference

### Outlook for recent updates
* [V0.51 BETA 2023.9.？（To be determined）](https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/tree/main/V0.51%20BETA%202023.9.%EF%BC%9F%EF%BC%88%E5%BE%85%E5%AE%9A%EF%BC%89) Real-time reading official version
    > Finding more suitable gamma and other parameters, improving neural network inference quality
    
    > Optimizing the connection between Kalman and electronic control

* Updating to an end-to-end neural network version by the end of the year (If I have time 😉)

## Bugs & Feature Requests

Please report bugs and request features using the [Issue Tracker](https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/issues).

