## 文件关系

utils_for_autoaim  图像处理层需要使用到的工具函数

utils_for_EKF  滤波及角度发送层需要使用到的工具函数

EKF 卡尔曼滤波独立进程（暂未完善和图像层的多进程交互）

autoaim 图像处理进程 （可以直接run😘）

config.yaml 参数文件。（90%的调参可以在这里独立进行，避免与主函数直接接触，便于后期快速调试）



拓展卡尔曼滤波还未调试完善，有任何问题请直接联系

# EKF  公式参考
![image](https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/assets/89187533/e06e4974-b366-4276-bd9a-bd1d37e6e13c)
