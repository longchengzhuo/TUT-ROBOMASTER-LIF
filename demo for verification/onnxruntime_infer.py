import cv2
import numpy as np
import onnxruntime as ort
from torchvision import transforms

providers = ['CPUExecutionProvider']
w = "youronnx.onnx"
img1 = cv2.imread('yourpic.png', -1)
img2 = ...
img3 = ...

def preproccess(*arguments):
    image_size = (32, 32)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(image_size)])
    i = 0
    for arg in arguments:
        arg = transform(arg)
        arg = np.expand_dims(arg, 0)
        if i == 0:
            img_joint = arg
        else:
            img_joint = np.concatenate((img_joint, arg), axis=0)
        i = i + 1
    return img_joint


def Inference(image):
    session = ort.InferenceSession(w, providers=providers)
    inp = {"inputs": image}
    outputs = session.run(None, inp)[0]
    print(outputs.argmax(1))

t1 = cv2.getTickCount()
img = preproccess(img1, img2, img3)
t2 = cv2.getTickCount()
spendTime1 = (t2 - t1) * 1 / (cv2.getTickFrequency())

Inference(img)
t3 = cv2.getTickCount()
spendTime2 = (t3 - t2) * 1 / (cv2.getTickFrequency())

Inference(img)
t4 = cv2.getTickCount()
spendTime3 = (t4 - t3) * 1 / (cv2.getTickFrequency())

Inference(img)
t5 = cv2.getTickCount()
spendTime4 = (t5 - t4) * 1 / (cv2.getTickFrequency())

Inference(img)
t6 = cv2.getTickCount()
spendTime5 = (t6 - t5) * 1 / (cv2.getTickFrequency())

Inference(img)
t7 = cv2.getTickCount()
spendTime6 = (t7 - t6) * 1 / (cv2.getTickFrequency())

print("spendTime1, spendTime2, spendTime3, spendTime4, spendTime5, spendTime6", spendTime1, spendTime2, spendTime3, spendTime4, spendTime5, spendTime6)
#spendTime1, fps2 0.000954 41