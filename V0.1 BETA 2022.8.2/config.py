import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
from CRC import *
import serial

# 串口参数
serialPort = "COM3"  # 串口
baudRate = 115200  # 波特率
# ser = serial.Serial(serialPort, baudRate, timeout=0.5)

# 模板匹配所用模板
number_img = cv2.imread("template_matching_pic.jpg")

# 填写敌方阵营的颜色，可以是 RED 和 BLUE
mode = "RED"

# 1为开启调试参数反馈以及参数可视化，0反之
debug = 1

# 腐蚀膨胀核
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 要打哨兵或者英雄吗？
attack_guard_or_hero = 0

# 调试用视频地址

cap = cv2.VideoCapture('测试视频地址')
# 大小装甲板参数 （单位mm）---------------------------------------------------------------------------------
small_obj_points = [[-67., -27.5, 0.],
                    [67., -27.5, 0.],
                    [-67., 27.5, 0.],
                    [67., 27.5, 0.]]

big_obj_points = [[-112.5, -27.5, 0.],
                  [112.5, -27.5, 0.],
                  [-112.5, 27.5, 0.],
                  [112.5, 27.5, 0.]]

# 内参数矩阵  --------------------------------------------------------------------------------------------
DAHENG_OldCamera_intrinsic = {

    "mtx": np.array([[1.34386883e+03, 0., 1.02702867e+03],
                     [0., 1.38920787e+03, 5.62395968e+02],
                     [0., 0., 1.]],
                    dtype=np.double),
    "dist": np.array([[-0.13939675, 0.42409417, -0.00454986, 0.01027033, -0.61637364]], dtype=np.double)

}

MDV_BlackCamera_intrinsic = {

    "mtx": np.array([[1.60954972e+03, 0., 6.36716902e+02],
                     [0., 1.61293941e+03, 5.27210575e+02],
                     [0., 0., 1.]],
                    dtype=np.double),
    "dist": np.array([[-0.08045919, 0.28519144, 0.00188052, -0.00196669, -0.07837469]], dtype=np.double)

}

MDV_WhiteCamera_intrinsic = {

    "mtx": np.array([[1.38012881e+03, 0.00000000e+00, 6.32563131e+02],
                     [0.00000000e+00, 1.41745621e+03, 3.95972569e+02],
                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.double),
    "dist": np.array([[ 0.00417728, -0.30223262, -0.01761736, -0.00159902, 1.24798931]], dtype=np.double)

}

# 追踪器参数初始化----------------------------------------------------------------------------------------
realnum_count = 0


last_Recognized_number = 0

Switch_co1 = 0
Switch_co2 = 0
# 模板匹配相关系数
correlation = 0.2

# 以下为用于参数可视化的参数初始化（勿动）---------------------------------------------------------------------
method = 4  # 1为大矩形长宽比，2为灯条面积比可视化，3为两灯条中心连线斜率，4为相关系数结果可视化
X = []
XX = 0
CALC = []
Match_counter = 0

# 以下为用于 上一帧 和 当前帧 对比检测的 循环 参数初始化（勿动）--------------------------------------------------
last_locx = 0
last_locy = 0
# 上一帧中装甲板中心坐标
last_width = 0
last_height = 0
# 上一帧中装甲板长宽

# Pipei 中的参数初始化------------------------------------------------------------------------------------
R = 0  # 相关系数的累计，方便可视化均值
N = 0  # 达到几次所要求相似度
x_sum = 0
y_sum = 0
# 卡尔曼-------------------------------------------------------------------------------------------------


# 卡尔曼重置标志（计数用）
lost_count = 0
aim_count = 0

kf = cv2.KalmanFilter(6, 3)
kf.measurementMatrix = np.array(
    [[1, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0]], np.float32)

kf.transitionMatrix = np.array(
    [[1, 0, 0, 1, 0, 0],
     [0, 1, 0, 0, 1, 0],
     [0, 0, 1, 0, 0, 1],
     [0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1]], np.float32)

kf.processNoiseCov = np.array(
    [[1, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0],
     [0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 0, 1]], np.float32) * 0.01
# ↑↑↑↑↑此系数越小表明,越相信"过程",越不相信"观测",就会离pnp解算值越远


# 函数------------------------------------------------------------------------------------------------

def Match(match_img, number_img, rect_Center_X, rect_Center_Y, R, N, x_sum, y_sum, img):
    global Match_counter
    Recognized_number = 0
    # 图像去噪、大津法二值化、缩小为原来的1/16

    match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2GRAY)

    match_img = cv2.erode(match_img, kernel)
    match_img = cv2.dilate(match_img, kernel)
    match_img = match_img[0:25, 14:40]

    ret2, temple = cv2.threshold(match_img, 0, 255, cv2.THRESH_OTSU)
    temple11 = cv2.cvtColor(temple, cv2.COLOR_GRAY2BGR)
    temple = cv2.resize(temple11, (int(temple11.shape[1] / 4), int(temple11.shape[0] / 4)))
    number_img = cv2.resize(number_img, (int(number_img.shape[1] / 4), int(number_img.shape[0] / 4)))
    h, w, c = temple.shape

    results = cv2.matchTemplate(number_img, temple, cv2.TM_SQDIFF_NORMED)  # 按照标准相关系数匹配
    
    for y in range(len(results)):  # 遍历结果数组的行
        for x in range(len(results[y])):  # 遍历结果数组的列
            if results[y][x] < correlation:  # 如果相关系数小于0.2则认为匹配成功

                R = R + results[y][x]
                N = N + 1
                y_sum = y_sum + y
                x_sum = x_sum + x
    if N != 0:
        Match_counter = Match_counter + 1

        if debug and method == 4:
            CALC.append(R / N)
            X.append(Match_counter)

        if 0 < int(y_sum / N + h / 2) < 50 / 4 and 0 < int(x_sum / N + w / 2) < 60 / 4:
            cv2.putText(img, "4", (rect_Center_X - 20, rect_Center_Y + 20), cv2.FONT_HERSHEY_COMPLEX, 2.0,
                        (255, 255, 255), 3)

            Recognized_number = 4
        elif 0 < int(y_sum / N + h / 2) < 50 / 4 and 60 / 4 < int(x_sum / N + w / 2) < 140 / 4:
            cv2.putText(img, "1", (rect_Center_X - 20, rect_Center_Y + 20), cv2.FONT_HERSHEY_COMPLEX, 2.0,
                        (255, 255, 255), 3)

            Recognized_number = 1

        elif 0 < int(y_sum / N + h / 2) < 50 / 4 and 140 / 4 < int(x_sum / N + w / 2) < 200 / 4:
            cv2.putText(img, "5", (rect_Center_X - 20, rect_Center_Y + 20), cv2.FONT_HERSHEY_COMPLEX, 2.0,
                        (255, 255, 255), 3)

            Recognized_number = 5
        elif 0 < int(y_sum / N + h / 2) < 50 / 4 and 200 / 4 < int(x_sum / N + w / 2) < 255 / 4:
            cv2.putText(img, "3", (rect_Center_X - 20, rect_Center_Y + 20), cv2.FONT_HERSHEY_COMPLEX, 2.0,
                        (255, 255, 255), 3)

            Recognized_number = 3

        return Recognized_number


# 大小装甲板透视变换，为模板匹配做准备----------------------------------------------------------------------------

def small_WarpPerspect(img, ts_box):
    wid, hei = 50, 25
    pts1 = np.float32(ts_box)
    pts2 = np.float32([[0, 0], [wid, 0], [wid, hei], [0, hei]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M, (wid, hei))

    return warped


def big_WarpPerspect(img, ts_box):
    new_zs = [int(ts_box[0][0] + (ts_box[1][0] - ts_box[0][0]) * (5.5 / 24)),
              int(ts_box[0][1] + (ts_box[1][1] - ts_box[0][1]) * (5.5 / 24))]
    new_ys = [int(ts_box[1][0] - (ts_box[1][0] - ts_box[0][0]) * (5.5 / 24)),
              int(ts_box[1][1] - (ts_box[1][1] - ts_box[0][1]) * (5.5 / 24))]
    new_yx = [int(ts_box[2][0] + (ts_box[3][0] - ts_box[2][0]) * (5.5 / 24)),
              int(ts_box[2][1] + (ts_box[3][1] - ts_box[2][1]) * (5.5 / 24))]
    new_zx = [int(ts_box[3][0] - (ts_box[3][0] - ts_box[2][0]) * (5.5 / 24)),
              int(ts_box[3][1] - (ts_box[3][1] - ts_box[2][1]) * (5.5 / 24))]

    ts_box = [new_zs, new_ys, new_yx, new_zx]
    wid, hei = 50, 25
    pts1 = np.float32(ts_box)
    pts2 = np.float32([[0, 0], [wid, 0], [wid, hei], [0, hei]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M, (wid, hei))

    return warped


# 卡尔曼结果可视化展示（受chenjunB站视频启发）-----------------------------------------------------------------------

def Kal_Visualization(predicted_X, predicted_Y, predicted_Z, img):
    coordinate = [[predicted_X], [predicted_Y], [predicted_Z]]
    Zc = coordinate[2][0]
    if Zc != 0:
        focal_length = np.mat(MDV_BlackCamera_intrinsic["mtx"])
        image_coordinate = (focal_length * coordinate) / Zc
        data = np.array(image_coordinate)
        if np.isnan(data).sum() == 3:
            pass
        else:
            image_coordinat = np.array(image_coordinate).reshape(-1)
            cv2.circle(img, (int(image_coordinat[0]), int(image_coordinat[1])), 20, (0, 0, 255), 5)


def Kal_predict(coordX, coordY, coordZ):
    measured = np.array([[np.float32(coordX)], [np.float32(coordY)], [np.float32(coordZ)]])
    predicted = kf.predict()
    kf.correct(measured)
    T = 1 + coordZ / 330
    x, y, z = int(kf.statePost[0] + T * kf.statePost[3]), \
              int(kf.statePost[1] + T * kf.statePost[4]), \
              int(kf.statePost[2] + T * kf.statePost[5])
    return x, y, z


def WriteVideo():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("width:", width, "height:", height)

    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1920, 1200))
    return out


def Para_visualization(CALC, X):
    Y = CALC
    plt.plot(X, Y, 'o')
    plt.show()


def SHOW(auto_aim, img):
    t1 = cv2.getTickCount()
    img = auto_aim(img)
    t2 = cv2.getTickCount()
    spendTime = (t2 - t1) * 1 / (cv2.getTickFrequency())
    FPS = int(1 / spendTime)
    FPS = 'The fps is %d' % (FPS)
    img = cv2.resize(img, (960, 600))
    cv2.putText(img, FPS, (200, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
    cv2.imshow('frame', img)
    return img


def PNP(real_rectbox, Kal_predict):
    img_points = np.array(real_rectbox, dtype=np.double)

    obj_points = np.reshape(small_obj_points, (4, 3))

    success, rvecs, tvecs = cv2.solvePnP(obj_points, img_points,
                                         MDV_BlackCamera_intrinsic["mtx"],
                                         MDV_BlackCamera_intrinsic["dist"])

    tvecs = np.array(tvecs)
    positions = [(tvecs[0][0], tvecs[1][0], tvecs[2][0])]
    predicted_X, predicted_Y, predicted_Z = Kal_predict(positions[0][0], positions[0][1], positions[0][2])
    return predicted_X, predicted_Y, predicted_Z, positions


def rotateXYZtoYAWPITCH(predicted_X, predicted_Y, predicted_Z):
    theta_x = np.arctan2(predicted_X, predicted_Z) / np.pi
    theta_y = np.arctan2(predicted_Y, np.sqrt(predicted_X * predicted_X + predicted_Z * predicted_Z)) / np.pi
    print(f"Euler angles:\ntheta_x: {theta_x}\ntheta_y: {theta_y}")
    print(predicted_X, predicted_Y, predicted_Z)
    return theta_x, theta_y


def Serial_communication(theta_x, theta_y, fps, is_autoaim):
    if is_autoaim == 1:
        f1 = bytes("$", encoding='utf8')
        f2 = 10
        f3 = float(theta_x)
        f4 = float(theta_y)
        f5 = fps
    else:
        f1 = bytes("$", encoding='utf8')
        f2 = 10
        f3 = 0
        f4 = 0
        f5 = 1
    # print('fps_______-----------------------------------------------------__', f2, f3, fps)
    pch_Message1 = get_Bytes(f1, is_datalen_or_fps=0)
    pch_Message2 = get_Bytes(f2, is_datalen_or_fps=1)
    pch_Message3 = get_Bytes(f3, is_datalen_or_fps=0)
    pch_Message4 = get_Bytes(f4, is_datalen_or_fps=0)
    pch_Message5 = get_Bytes(f5, is_datalen_or_fps=2)
    pch_Message = pch_Message1 + pch_Message2 + pch_Message3 + pch_Message4 + pch_Message5

    wCRC = get_CRC16_check_sum(pch_Message, CRC16_INIT)
    # ser.write(struct.pack("=cBffHi", f1, f2, f3, f4, f5, wCRC))  #分别是帧头，长度，数据，数据，fps，校验

