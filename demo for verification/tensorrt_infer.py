import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time
from torchvision import transforms
import torch

BATCH_SIZE = 10
CLASS_NUMBER = 8

t0 = cv2.getTickCount()
img1 = cv2.imread("yourpic.png", -1)
img2 = ...
img3 = ...
img4 = ...
img5 = ...
img6 = ...
t1 = cv2.getTickCount()
print("读图", float((t1 - t0) * 1 / (cv2.getTickFrequency())))

with open("yourengine.trt", "rb") as f:
    serialized_engine = f.read()
logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(serialized_engine)
with open('imagenet_classes.txt') as f:
    classes = [line.strip() for line in f.readlines()]

t2 = cv2.getTickCount()
print("读trt", float((t2 - t1) * 1 / (cv2.getTickFrequency())))

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


def preproccess(*arguments):
    image_size = (32, 32)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(image_size)])
    i = 0
    for arg in arguments:
        arg = transform(arg)
        if i == 0:
            img_joint = arg
        else:
            img_joint = torch.cat((img_joint, arg), dim=0)
        i = i + 1
    return img_joint


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

t3 = cv2.getTickCount()

context = engine.create_execution_context()
context.set_binding_shape(0, (BATCH_SIZE, 1, 32, 32))  # 这句非常重要！！！定义batch为动态维度
inputs, outputs, bindings, stream = allocate_buffers(engine, max_batch_size=BATCH_SIZE)  # 构建输入，输出，流指针
# ↑↑↑↑↑ 创建上下文管理context并获取相关buffer, 每当batch要变化时, 要重新set_binding_shape, 并且需要重新申请buffer ↑↑↑↑↑
t4 = cv2.getTickCount()
print("engine和buferr", float((t4 - t3) * 1 / (cv2.getTickFrequency())))
batch_data = preproccess(img1, img2, img3, img4, img5, img6)
t5 = cv2.getTickCount()
print("preproccess", float((t5 - t4) * 1 / (cv2.getTickFrequency())))
np.copyto(inputs[0].host, batch_data.ravel())
t6 = cv2.getTickCount()
print("np.copyto", float((t6 - t5) * 1 / (cv2.getTickFrequency())))
result = do_inference_v2(context, bindings, inputs, outputs, stream)[0]
t7 = cv2.getTickCount()
print("do_inference_v2", float((t7 - t6) * 1 / (cv2.getTickFrequency())))
result = np.reshape(result, [BATCH_SIZE, -1, CLASS_NUMBER])
for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        max_index = np.argmax(result[i][j])
        print(f"第{i+1}个图片识别为{classes[max_index]}")
t8 = cv2.getTickCount()
print("解析结果", float((t8 - t7) * 1 / (cv2.getTickFrequency())))