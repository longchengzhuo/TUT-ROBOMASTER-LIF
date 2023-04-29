# V0.1 BETA 2022.8.2

* [参数调整](#参数调整)
* [追踪器](#追踪器)
* [卡尔曼预测](#卡尔曼预测)
<img src="https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/blob/main/docs/real_test.gif" width="600px">

在进入到自瞄线程时，摄像机拉流时分辨率为（1280，1024），未缩放

上述图片中所展示为每一帧都使用out.write(cv2.resize(img, (1920, 1200)))时的效果

在不使用存帧情况下，帧率110左右




## 参数调整



不同相机畸变情况不一样，hsv滤过之后灯光散光效果也不一样，需要根据自己的相机调整，装甲板长宽比和hsv红蓝系数

GainHSV提供了一个可视化调参界面，可以使用一个mask，或者两个...

<img src="https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/blob/main/docs/hsv.gif" width="600px">

使用如下函数能够可视化一个测试视频中长宽比比值散点图(在main_Autoaim中已经开启)

```
Para_visualization(CALC, X)
```

<img src="https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/blob/main/docs/Figure_1.png" width="600px">

通过我们mindvision相机测试，可以得到装甲板矩形长宽比为1.23 < calcCache < 2.5（小装甲板），2.5 < calcCache < 4.1（大装甲板）




## 追踪器





##### 由于当前版本代码我们使用传统模板匹配的“蠢办法”来进行数字识别，经常会发现误识别和错识别的情况，所以我使用了一个基于简单逻辑判断的结构来手动帮助识别。

### 追踪器逻辑结构：

如果识别到数字存在且和上一帧数字相同，且当前装甲板在上一帧装甲板附近（“附近”是指两帧之间装甲板中心距离的平方小于上一帧装甲板半个周长的平方的0.125倍），则继承当前装甲板信息，代码如下：

```python
if Recognized_number == last_Recognized_number and (
        (last_locx - rect_Center_X) ** 2 + (last_locy - rect_Center_Y) ** 2) < (((last_width + last_height) ** 2) / 8):
```



如果识别到数字存在且和上一帧数字相同，且当前装甲板不在上一帧装甲板附近，则连续观察两帧，若持续两帧都是这种情况，则可以继承当前装甲板信息。这样做虽然会多浪费掉一帧的时间，但是可以防止误识别带来的装甲板位置跳变（你懂的，谁也不希望哨兵偶尔抽风锁到观众席或者微弱反光的地板上去）代码如下：

```python
if Recognized_number == last_Recognized_number and (
        (last_locx - rect_Center_X) ** 2 + (last_locy - rect_Center_Y) ** 2) >= (
        ((last_width + last_height) ** 2) / 8):

    Switch_co1 = Switch_co1 + 1

    if Switch_co1 > 2:
```



如果识别到数字存在且和上一帧数字不同，则连续观察三帧，若持续三帧都是这种情况，则可以继承当前装甲板信息。显然，这种情况比Switch_co1 > 2要更加谨慎，代码如下：

```python
if Recognized_number != last_Recognized_number:
    Switch_co2 = Switch_co2 + 1
    if Switch_co2 > 3:
```



如果根本没识别到数字，但是曾经识别到过数字，且当前装甲板在上一帧装甲板附近，则可以继承，代码如下：

```python
elif realnum_count != 0 and (
        (last_locx - rect_Center_X) ** 2 + (last_locy - rect_Center_Y) ** 2) < (((last_width + last_height) ** 2) / 8):
```



495行附近有一个else，是针对非小装甲板的情况使用的，这种情况下没有使用追踪器，因为在模板匹配的思路下，**很难**进行大小装甲板的划分（受限于当时的知识储备）。




## 卡尔曼预测





<img src="https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/blob/main/docs/kal2_2.gif" width="600px">

<img src="https://github.com/longchengzhuo/TUT-ROBOMASTER-LIF/blob/main/docs/kal2_1.gif" width="600px">

##### RM的自瞄中我们观测值选取为解算后的XYZ，实例化如下

```
kf = cv2.KalmanFilter(6, 3)
```

**过程噪声协方差我们取0.01**

```
kf.processNoiseCov = np.array(
    [[1, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1]], np.float32) * 0.01
```

### 预测逻辑结构：

在检测到装甲板之后，在程序启动的开始，将会重置卡尔曼

```
if real_rectbox:
    if aim_count == 0:
        kf.statePost = np.array([0, 0, 0, 0, 0, 0], np.float32)

```

在前后两帧装甲板跳变较大时，也会重置卡尔曼

    if ((last_locx - LOCX[real_targetNum]) ** 2 + (last_locy - LOCY[real_targetNum]) ** 2) >= (
            ((last_width + last_height) ** 2) / 2):
        kf.statePost = np.array([0, 0, 0, 0, 0, 0], np.float32)

若追踪器丢失，当然也会重置卡尔曼

```
else:
    kf.statePost = np.array([0, 0, 0, 0, 0, 0], np.float32)
```

在预测的时候，我并没有直接将“套公式”之后得到的修正值拿来用，而是在那个基础上加上了时间补偿。coordZ是子弹距离枪口的**大概**距离，330是考虑到子弹射速下坠以及系统机械电控延迟整定出来的参数。

```python
T = 1 + coordZ / 330
x, y, z = int(kf.statePost[0] + T * kf.statePost[3]), \
          int(kf.statePost[1] + T * kf.statePost[4]), \
          int(kf.statePost[2] + T * kf.statePost[5])
```

**然后就来到比较精彩的地方了：**

```python
lost_count < 5
```

表示在丢失5帧以内，依然继承丢失前卡尔曼滤过之后的速度进行预测，以达到在短暂丢失目标时，**云台依然能够“猜测”对方的移动方向继续进行跟踪**，避免短暂丢失时依然需要重置卡尔曼重新收敛的情况。***使用极其简单高效，这就是卡尔曼***











