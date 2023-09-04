import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
# from CRC import *
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
from torchvision import transforms
import torch
import yaml
# import serial
# import mvsdk
import argparse

cap = cv2.VideoCapture('/home/rcclub/test/133_.avi')  #测试视频地址

parser = argparse.ArgumentParser(description='击打对象颜色和参数设置')
parser.add_argument('--enemy_armor_colour', type=str, default="RED", help='set colour.')
parser.add_argument('--my_camera', type=str, default="134", help='set camera.')  #133好像要用134？
args = parser.parse_args()
print("这局敌人是什么颜色装甲板{}，我是什么摄像头{}".format(args.enemy_armor_colour, args.my_camera))

with open("config.yaml", 'r', encoding='utf-8') as f:
    yaml = yaml.safe_load(f)

# ser = serial.Serial(yaml["serialPort"], yaml["baudRate"], timeout=0.5)
video_size = tuple(yaml['video_size'])
BATCH_SIZE = yaml["BATCH_SIZE"]
two_class_trt = yaml["two_class_trt"]
two_class_txt = yaml["two_class_txt"]
eight_class_trt = yaml["eight_class_trt"]
eight_class_txt = yaml["eight_class_txt"]
small_obj_points = yaml["small_obj_points"]
small_armor_list = yaml["small_armor_list"]
big_armor_list = yaml["big_armor_list"]
conf_is_it_armor = yaml["conf_is_it_armor"]
conf_small_armor = yaml["conf_small_armor"]
conf_big_armor = yaml["conf_big_armor"]
img_shrink_scale = yaml["img_shrink_scale"]
if args.my_camera == "134":
    mtx = yaml["MDV_BlackCamera_intrinsic"]["mtx"]
    dist = yaml["MDV_BlackCamera_intrinsic"]["dist"]
elif args.my_camera == "133":
    mtx = yaml["MDV_WhiteCamera_intrinsic"]["mtx"]
    dist = yaml["MDV_WhiteCamera_intrinsic"]["dist"]
elif args.my_camera == "DH":
    mtx = yaml["DAHENG_OldCamera_intrinsic"]["mtx"]
    dist = yaml["DAHENG_OldCamera_intrinsic"]["dist"]
# 腐蚀膨胀核
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


# 图像颜色提取 ---------------------------------------------------------------------------------------------------
def color_extra(img):
    if args.enemy_armor_colour == "BLUE":
        # 蓝色
        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # HSV BLUE
        # lower = np.array([84, 7, 196])
        lower = np.array([83, 70, 194])  #大恒
        upper = np.array([114, 255, 255])
        mask = cv2.inRange(hsv_image, lower, upper)
        img_after_extract = mask

    else:
        # 红色
        if args.my_camera == "134":
            hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # 区间1
            lower_red = np.array([0, 79, 49])  # lower = np.array([0, 103, 0])  # [70, 0, 250]
            upper_red = np.array([12, 255, 255])  # upper = np.array([194, 255, 255])
            mask0 = cv2.inRange(hsv_image, lower_red, upper_red)
            # 区间2
            lower_red = np.array([152, 37, 105])  # lower_red = np.array([0, 0, 255])
            upper_red = np.array([182, 255, 255])  # upper_red = np.array([255, 255, 255])
            mask1 = cv2.inRange(hsv_image, lower_red, upper_red)
            # 拼接两个区间
            mask = mask0 + mask1
            img_after_extract = mask
        elif args.my_camera == "133":
            hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 101, 224], dtype="uint8")  # [0, 79, 105]
            upper = np.array([29, 255, 255], dtype="uint8")
            mask = cv2.inRange(hsv_image, lower, upper)
            img_after_extract = mask
    img_after_extract = cv2.dilate(img_after_extract, kernel)
    contours, hierarchy = cv2.findContours(
        img_after_extract, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 获取轮廓
    return contours


# 神经网络相关函数-----------------------------------------------------------------------------------------------------
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine, max_batch_size=16):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        dims = engine.get_binding_shape(binding)
        # print(dims)
        if dims[0] == -1:
            assert (max_batch_size is not None)
            dims[0] = max_batch_size  # 动态batch_size适应

        # size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        size = trt.volume(dims) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # print(dtype,size)
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)  # 开辟出一片显存
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference_v2(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


def creat_buffer(BATCH_SIZE, two_class_trt, two_class_txt, eight_class_trt, eight_class_txt):
    with open(two_class_trt, "rb") as f:
        two_serialized_engine = f.read()

    with open(eight_class_trt, "rb") as f:
        eight_serialized_engine = f.read()

    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    two_engine = runtime.deserialize_cuda_engine(two_serialized_engine)
    eight_engine = runtime.deserialize_cuda_engine(eight_serialized_engine)

    with open(two_class_txt) as f:
        two_classes = [line.strip() for line in f.readlines()]

    with open(eight_class_txt) as f:
        eight_classes = [line.strip() for line in f.readlines()]

    two_context = two_engine.create_execution_context()
    two_context.set_binding_shape(0, (BATCH_SIZE, 1, 32, 32))  # 这句非常重要！！！定义batch为动态维度
    inputs_2, outputs_2, bindings_2, stream_2 = allocate_buffers(two_engine, max_batch_size=BATCH_SIZE)  # 构建输入，输出，流指针

    eight_context = eight_engine.create_execution_context()
    eight_context.set_binding_shape(0, (BATCH_SIZE, 1, 32, 32))  # 这句非常重要！！！定义batch为动态维度
    inputs_11, outputs_11, bindings_11, stream_11 = allocate_buffers(eight_engine, max_batch_size=BATCH_SIZE)  # 构建输入，输出，流指针

    # ↑↑↑↑↑ 创建上下文管理context并获取相关buffer, 每当batch要变化时, 要重新set_binding_shape, 并且需要重新申请buffer ↑↑↑↑↑
    return inputs_2, outputs_2, bindings_2, stream_2, two_classes, two_context, inputs_11, outputs_11, bindings_11, stream_11, eight_classes, eight_context


def is_it_armor(is_it_armor_img_joint, BATCH_SIZE, inputs, outputs, bindings, stream, classes, context, how_many_probably_armor_exist): # 这是装甲板吗
    classnum = 2
    is_it_armor_and_confidence = []
    np.copyto(inputs[0].host, is_it_armor_img_joint.ravel())
    result = do_inference_v2(context, bindings, inputs, outputs, stream)[0]
    result = np.reshape(result, [BATCH_SIZE, -1, classnum])
    for i in range(how_many_probably_armor_exist): #这里换成batchsize？？
        for j in range(result.shape[1]):
            max_index = np.argmax(softmax(np.array(result[i][j])))
            confidence = softmax(np.array(result[i][j]))[max_index]
            is_it_armor_and_confidence.append((int(classes[max_index]), confidence))
    return is_it_armor_and_confidence


def what_is_num(what_is_num_img_joint, BATCH_SIZE, inputs, outputs, bindings, stream, classes, context, how_many_armor_exist): # 这是几号装甲板
    classnum = 11
    what_is_number_and_confidence = []
    np.copyto(inputs[0].host, what_is_num_img_joint.ravel())
    result = do_inference_v2(context, bindings, inputs, outputs, stream)[0]
    result = np.reshape(result, [BATCH_SIZE, -1, classnum])
    for i in range(how_many_armor_exist): #这里换成batchsize？？
        for j in range(result.shape[1]):
            max_index = np.argmax(softmax(np.array(result[i][j])))
            confidence = softmax(np.array(result[i][j]))[max_index]
            what_is_number_and_confidence.append((str(classes[max_index]), confidence))
    return what_is_number_and_confidence


# 一般函数-----------------------------------------------------------------------------------------------------
def feedback_coordinates(right_points_y, right_points_x, left_points_y, left_points_x, rectbox11, rectbox21, rectbox12, rectbox22):
    # 这里主要得到左边灯条矩形上边的一组x和y， 1为左上，2为右上，3为左下，4为右下
    y1 = np.array(right_points_y)
    x1 = np.array(right_points_x)
    y2 = np.array(left_points_y)
    x2 = np.array(left_points_x)
    # ↓↓↓ y1是右边四个点， 现在进行y轴排序 ↓↓↓ --------------------------------------------------------------------------
    top11 = y1.argsort()[-1]
    top12 = y1.argsort()[-2]
    top13 = y1.argsort()[-3]
    top14 = y1.argsort()[-4]
    top21 = y2.argsort()[-1]
    top22 = y2.argsort()[-2]
    top23 = y2.argsort()[-3]
    top24 = y2.argsort()[-4]

    rectbox11.append([float((x1[top11] + x1[top12]) / 2),
                      float((y1[top11] + y1[top12]) / 2)])
    rectbox12.append([float((x1[top13] + x1[top14]) / 2),
                      float((y1[top13] + y1[top14]) / 2)])
    rectbox21.append([float((x2[top21] + x2[top22]) / 2),
                      float((y2[top21] + y2[top22]) / 2)])
    rectbox22.append([float((x2[top23] + x2[top24]) / 2),
                      float((y2[top23] + y2[top24]) / 2)])

    if float((rectbox21[0][0] + rectbox22[0][0]) / 2) >= float(
            (rectbox11[0][0] + rectbox12[0][0]) / 2):  # 始终保持2在左
        a = rectbox11
        b = rectbox12
        rectbox11 = rectbox21
        rectbox12 = rectbox22
        rectbox21 = a
        rectbox22 = b

    box = rectbox22 + rectbox12 + rectbox21 + rectbox11

    yy4 = np.array(
        [y2[top21], y2[top22]])  # 右边高度较高的两个点
    yy2 = np.array(
        [y2[top23], y2[top24]])  # 右边高度较低的两个点
    yy1 = np.array(
        [y1[top13], y1[top14]])  # 左边高度较低的两个点
    yy3 = np.array(
        [y1[top11], y1[top12]])  # 左边高度较高的两个点

    xx4 = np.array(
        [x2[top21], x2[top22]])
    xx2 = np.array(
        [x2[top23], x2[top24]])
    xx1 = np.array(
        [x1[top13], x1[top14]])
    xx3 = np.array(
        [x1[top11], x1[top12]])
    # ↓↓↓ 现在进行x轴排序 ↓↓↓ -----------------------------------------------------------------------------------------
    r111 = xx1.argsort()[-1]  # -1代表返回最大值索引
    r112 = xx1.argsort()[-2]
    r311 = xx3.argsort()[-1]
    r312 = xx3.argsort()[-2]
    r221 = xx2.argsort()[-1]
    r222 = xx2.argsort()[-2]
    r421 = xx4.argsort()[-1]
    r422 = xx4.argsort()[-2]

    if xx1[r111] > xx2[r222]:
        r111 = r112
        r222 = r221
        r311 = r312
        r422 = r421

    # 小矩形

    ts_zs = [xx1[r111], yy1[r111]]
    ts_ys = [xx2[r222], yy2[r222]]
    ts_zx = [xx3[r311], yy3[r311]]
    ts_yx = [xx4[r422], yy4[r422]]
    if ts_zs[0] > ts_ys[0]:  # 始终保持z是左，y是右
        h_zs = ts_zs
        h_zx = ts_zx
        ts_zs = ts_ys
        ts_zx = ts_yx
        ts_ys = h_zs
        ts_yx = h_zx
    ts_box = [ts_zs, ts_ys, ts_yx, ts_zx]  # 透视变换的坐标box
    rect_Center_X = float((xx3[r311] + xx2[r222]) / 2)
    rect_Center_Y = float((yy1[r111] + yy4[r422]) / 2)  # rect_Center_X,rect_Center_Y为矩形中间点坐标
    return ts_box, rect_Center_X, rect_Center_Y, box


def big_coor2small_coor(what_is_num_rectbox, little_count): # 把大装甲板的角点转换成对应的小装甲板的角点 便于后续进行解算
    last_zs = [float(what_is_num_rectbox[little_count][0][0] + (
                what_is_num_rectbox[little_count][1][0] - what_is_num_rectbox[little_count][0][0]) * (145 / 693)),
               float(what_is_num_rectbox[little_count][0][1] + (
                           what_is_num_rectbox[little_count][1][1] - what_is_num_rectbox[little_count][0][1]) * (
                                 145 / 693))]
    last_ys = [float(what_is_num_rectbox[little_count][1][0] - (
                what_is_num_rectbox[little_count][1][0] - what_is_num_rectbox[little_count][0][0]) * (145 / 693)),
               float(what_is_num_rectbox[little_count][1][1] - (
                           what_is_num_rectbox[little_count][1][1] - what_is_num_rectbox[little_count][0][1]) * (
                                 145 / 693))]
    last_yx = [float(what_is_num_rectbox[little_count][2][0] + (
                what_is_num_rectbox[little_count][3][0] - what_is_num_rectbox[little_count][2][0]) * (145 / 693)),
               float(what_is_num_rectbox[little_count][2][1] + (
                           what_is_num_rectbox[little_count][3][1] - what_is_num_rectbox[little_count][2][1]) * (
                                 145 / 693))]
    last_zx = [float(what_is_num_rectbox[little_count][3][0] - (
                what_is_num_rectbox[little_count][3][0] - what_is_num_rectbox[little_count][2][0]) * (145 / 693)),
               float(what_is_num_rectbox[little_count][3][1] - (
                           what_is_num_rectbox[little_count][3][1] - what_is_num_rectbox[little_count][2][1]) * (
                                 145 / 693))]
    return [last_zs, last_ys, last_zx, last_yx]


def add_light_bar_coor(rectbox1y, rectbox2y, rectbox3y, rectbox4y, rectbox1x, rectbox2x, rectbox3x, rectbox4x, rectpoint): # 添加灯条角点坐标
    rectbox1x.append(float(rectpoint[0][0]))
    rectbox2x.append(float(rectpoint[1][0]))
    rectbox3x.append(float(rectpoint[2][0]))
    rectbox4x.append(float(rectpoint[3][0]))
    rectbox1y.append(float(rectpoint[0][1]))
    rectbox2y.append(float(rectpoint[1][1]))
    rectbox3y.append(float(rectpoint[2][1]))
    rectbox4y.append(float(rectpoint[3][1]))
    return rectbox1y, rectbox2y, rectbox3y, rectbox4y, rectbox1x, rectbox2x, rectbox3x, rectbox4x


def paired_light_bar_coor(rectbox1y, rectbox2y, rectbox3y, rectbox4y, rectbox1x, rectbox2x, rectbox3x, rectbox4x, target, targetNum): # 提取出配对的左右八个角点坐标
    right_points_y = [rectbox1y[target[targetNum][0]],
              rectbox2y[target[targetNum][0]],
              rectbox3y[target[targetNum][0]],
              rectbox4y[target[targetNum][0]]]

    right_points_x = [rectbox1x[target[targetNum][0]],
              rectbox2x[target[targetNum][0]],
              rectbox3x[target[targetNum][0]],
              rectbox4x[target[targetNum][0]]]

    left_points_y = [rectbox1y[target[targetNum][1]],
             rectbox2y[target[targetNum][1]],
             rectbox3y[target[targetNum][1]],
             rectbox4y[target[targetNum][1]]]

    left_points_x = [rectbox1x[target[targetNum][1]],
             rectbox2x[target[targetNum][1]],
             rectbox3x[target[targetNum][1]],
             rectbox4x[target[targetNum][1]]]
    return right_points_y, right_points_x, left_points_y, left_points_x


def WarpPerspect(img, ts_box): # 通过灯条坐标变换透视变化图像得到32*32的数字图像，便于后续送入网络
    new_zs = [float(ts_box[0][0] - (ts_box[3][0] - ts_box[0][0]) * 0.5),
              float(ts_box[0][1] - (ts_box[3][1] - ts_box[0][1]) * 0.5)]
    new_ys = [float(ts_box[1][0] - (ts_box[2][0] - ts_box[1][0]) * 0.5),
              float(ts_box[1][1] - (ts_box[2][1] - ts_box[1][1]) * 0.5)]
    new_yx = [float(ts_box[2][0] + (ts_box[2][0] - ts_box[1][0]) * 0.5),
              float(ts_box[2][1] + (ts_box[2][1] - ts_box[1][1]) * 0.5)]
    new_zx = [float(ts_box[3][0] + (ts_box[3][0] - ts_box[0][0]) * 0.5),
              float(ts_box[3][1] + (ts_box[3][1] - ts_box[0][1]) * 0.5)]

    ts_box = [new_zs, new_ys, new_yx, new_zx]
    wid, hei = 32, 32
    pts1 = np.float32(ts_box)
    pts2 = np.float32([[0, 0], [wid, 0], [wid, hei], [0, hei]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, M, (wid, hei))
    ret2, temple = cv2.threshold(warped, 0, 255, cv2.THRESH_OTSU)
    return temple


def PNP(real_rectbox):
    img_points = np.array(real_rectbox, dtype=np.double)
    img_points = img_points * img_shrink_scale
    print("real_rectbox", img_points)
    obj_points = np.reshape(np.array(small_obj_points), (4, 3))

    success, rvecs, tvecs = cv2.solvePnP(obj_points, img_points,
                                        np.array(mtx, dtype=np.double),
                                        np.array(dist, dtype=np.double))

    tvecs = np.array(tvecs)
    positions = [(tvecs[0][0], tvecs[1][0], tvecs[2][0])]
    predicted_X, predicted_Y, predicted_Z = tvecs[0][0], tvecs[1][0], tvecs[2][0]
    yaw = np.arctan2(predicted_X, predicted_Z) *180/np.pi
    pitch = np.arctan2(predicted_Y, np.sqrt(predicted_X * predicted_X + predicted_Z * predicted_Z))*180/np.pi
    print("yawpitch", yaw, pitch)
    return predicted_X, predicted_Y, predicted_Z, positions


def WriteVideo(video_size):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("width:", width, "height:", height)

    out = cv2.VideoWriter('output1.avi', fourcc, 20.0, video_size)
    return out


def SHOW(img2coor_autoaim, img):
    t1 = cv2.getTickCount()
    img = img2coor_autoaim(img, is_it_armor, what_is_num)
    t2 = cv2.getTickCount()
    spendTime = (t2 - t1) * 1 / (cv2.getTickFrequency())
    FPS = int(1 / spendTime)
    FPS = 'The fps is %d' % (FPS)
    print(FPS)
    cv2.putText(img, FPS, (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 200, 200), 3)
    return img


def Kal_Visualization(predicted_X, predicted_Y, predicted_Z, img): # 三维卡尔曼结果二维可视化
    coordinate = [[predicted_X], [predicted_Y], [predicted_Z]]
    Zc = coordinate[2][0]
    if Zc != 0:
        focal_length = np.mat(yaml["MDV_BlackCamera_intrinsic"]["mtx"])
        image_coordinate = (focal_length * coordinate) / Zc
        data = np.array(image_coordinate)
        if np.isnan(data).sum() == 3:
            pass
        else:
            image_coordinat = np.array(image_coordinate).reshape(-1)
            cv2.circle(img, (int(image_coordinat[0]), int(image_coordinat[1])), 20, (0, 0, 255), 5)


