from utils_for_autoaim import *
import platform


inputs_2, outputs_2, bindings_2, stream_2, classes_2, context_2, \
inputs_11, outputs_11, bindings_11, stream_11, classes_11, context_11 \
    = creat_buffer(BATCH_SIZE, two_class_trt, two_class_txt, eight_class_trt, eight_class_txt)
transform = transforms.Compose([transforms.ToTensor()])


def img2coor_autoaim(img, is_it_armor, what_is_num):
    t11 = cv2.getTickCount()
    global aim_count
    global confid


    real_rectbox = []
    n = 0
    x = []
    y = []
    locX = []
    locY = []  # 存储计算得到的中心点坐标
    target = []  # 存储配对的两个灯条的编号 (L1, L2)
    pairNum = 0  # 初始化计数变量
    longSide = []
    rectbox1x = []
    rectbox2x = []
    rectbox3x = []
    rectbox4x = []
    rectbox1y = []
    rectbox2y = []
    rectbox3y = []
    rectbox4y = []
    shortSide = []

    # 定义准星位置
    sightX = int((img.shape[1]) / 2)
    sightY = int((img.shape[0]) / 2)

    img_forgray = img.copy()
    contours = color_extra(img)
    t22 = cv2.getTickCount()
    spendTime = (t22 - t11) * 1 / (cv2.getTickFrequency())
    print("取颜色找轮廓", spendTime)
    for contour in contours:
        rect = cv2.minAreaRect(contour)  # 获取最小包围矩形
        xCache, yCache = rect[0]  # 获取矩形的中心坐标
        widthC, heigthC = rect[1]  # 获取矩形的长宽
        rectpoint = cv2.boxPoints(rect)  # 获取矩形四个角点

        if (widthC == 0) | (heigthC == 0):
            break
        angleCache = rect[2]  # 角度为x轴逆时针旋转，第一次接触到矩形边界时的值，范围：0~-90°，第一次接触的边界是宽！！！！！
        rectpoint = np.int0(rectpoint)

        if angleCache < -45:
            v = heigthC
            heigthC = widthC
            widthC = v

        if 15 >= (heigthC / widthC) >= 1:  # 灯条是竖直放置，长宽比满足条件
            x.append(float(xCache))
            y.append(float(yCache))
            rectbox1y, rectbox2y, rectbox3y, rectbox4y,\
            rectbox1x, rectbox2x, rectbox3x, rectbox4x = add_light_bar_coor(rectbox1y, rectbox2y,
                                  rectbox3y, rectbox4y,
                                  rectbox1x, rectbox2x,
                                  rectbox3x, rectbox4x, rectpoint)
            longSide.append(float(heigthC))
            shortSide.append(float(widthC))
            n = n + 1  # 有效矩形计数

    if n >= 2:  # 图像中找到两个以上的灯条
        for count in range(0, n):
            findCache = count + 1  # 初始化计数变量
            while findCache < n:  # 未超界，进行匹配运算
                calcCache = math.sqrt((x[findCache] - x[count]) ** 2 + (y[findCache] - y[count]) ** 2)  # 求中心点连线长
                calcCache = (2 * calcCache) / (longSide[count] + longSide[findCache])  # 求快捷计算单位
                area1 = longSide[count] * shortSide[count]
                area2 = longSide[findCache] * shortSide[findCache]

                if area1 != 0 and area2 != 0:
                    calc_area = area1 / area2
                    if calc_area < 1:
                        calc_area = 1 / calc_area
                    if (1 <= calc_area < 5) and (1.23 < calcCache < 4.1) and (x[findCache] - x[count]) ** 2 > (
                            y[findCache] - y[count]) ** 2:
                        target.append((count, findCache))
                        locX.append(float((x[count] + x[findCache]) / 2))
                        locY.append(float((y[count] + y[findCache]) / 2))

                        pairNum = pairNum + 1
                    findCache = findCache + 1
                elif area1 == 0:
                    break
                else:
                    findCache = findCache + 1
        t33 = cv2.getTickCount()
        spendTime = (t33 - t22) * 1 / (cv2.getTickFrequency())
        print("找配对", spendTime)
        if pairNum != 0:
            it_is_armor = 0
            is_it_armor_box = []
            is_it_armor_locX = []
            is_it_armor_locY = []
            is_it_armor_dis = []
            probably_armor_exist = 0
            how_many_probably_armor_exist = 0
            warped = cv2.cvtColor(img_forgray, cv2.COLOR_BGR2GRAY)
            for count in range(0, pairNum):
                rectbox11 = []
                rectbox21 = []
                rectbox12 = []
                rectbox22 = []
                targetNum = count
                K = abs(y[target[targetNum][0]] - y[target[targetNum][1]]) / abs(
                    x[target[targetNum][0]] - x[target[targetNum][1]])  # 两灯条中心连线斜率

                if K < 0.3: #0.4

                    probably_armor_exist = 1
                    right_points_y, right_points_x, left_points_y, left_points_x = paired_light_bar_coor(rectbox1y, rectbox2y,
                                                                                                         rectbox3y, rectbox4y,
                                                                                                         rectbox1x, rectbox2x,
                                                                                                         rectbox3x, rectbox4x,
                                                                                                         target, targetNum)
                    ts_box, rect_Center_X, rect_Center_Y, box = feedback_coordinates(right_points_y, right_points_x,
                                                                                     left_points_y, left_points_x,
                                                                                     rectbox11, rectbox21, rectbox12,
                                                                                     rectbox22)
                    is_it_armor_box.append(box)
                    is_it_armor_locX.append(locX[count])
                    is_it_armor_locY.append(locY[count])
                    is_it_armor_dis.append(math.sqrt((locX[count] - sightX) ** 2 + (locY[count] - sightY) ** 2))
                    is_it_armor_img = WarpPerspect(warped, ts_box)
                    is_it_armor_img = transform(is_it_armor_img)

                    if how_many_probably_armor_exist == 0:
                        is_it_armor_img_joint = is_it_armor_img
                    else:
                        is_it_armor_img_joint = torch.cat(
                            (is_it_armor_img_joint, is_it_armor_img), dim=0)

                    how_many_probably_armor_exist = how_many_probably_armor_exist + 1
            t44 = cv2.getTickCount()
            spendTime = (t44 - t33) * 1 / (cv2.getTickFrequency())
            print("找出每对各自坐标", spendTime)
            if probably_armor_exist:

                is_it_armor_count = 0
                how_many_armor_exist = 0
                what_is_num_rectbox = []
                what_is_num_LOCX = []
                what_is_num_LOCY = []
                what_is_num_dis = []

                y = torch.zeros((BATCH_SIZE - how_many_probably_armor_exist), 32, 32)  # 创建一个[6, 32, 32]形状的全0tensor
                is_it_armor_img_joint = torch.cat((is_it_armor_img_joint, y), dim=0)
                is_it_armor_and_confidence = is_it_armor(is_it_armor_img_joint, BATCH_SIZE, inputs_2, outputs_2, bindings_2, stream_2, classes_2, context_2, how_many_probably_armor_exist)

                for is_it_armor, confidence in is_it_armor_and_confidence:
                    if is_it_armor and confidence >= conf_is_it_armor:
                        it_is_armor = 1
                        what_is_num_rectbox.append(is_it_armor_box[is_it_armor_count])
                        what_is_num_LOCX.append(is_it_armor_locX[is_it_armor_count])
                        what_is_num_LOCY.append(is_it_armor_locY[is_it_armor_count])
                        what_is_num_dis.append(is_it_armor_dis[is_it_armor_count])
                        if how_many_armor_exist == 0:
                            what_is_num_img_joint = torch.unsqueeze(is_it_armor_img_joint[is_it_armor_count], 0)
                        else:
                            what_is_num_img_joint = torch.cat(
                                (what_is_num_img_joint, torch.unsqueeze(is_it_armor_img_joint[is_it_armor_count], 0)), dim=0)
                        how_many_armor_exist = how_many_armor_exist + 1
                    is_it_armor_count = is_it_armor_count + 1
            if it_is_armor :
                little_count = 0
                last_rectbox = []
                last_LOCX = []
                last_LOCY = []
                last_dis = []
                realrealnum = 0

                y = torch.zeros((BATCH_SIZE - how_many_armor_exist), 32, 32)  # 创建一个[6, 32, 32]形状的全0tensor
                what_is_num_img_joint = torch.cat((what_is_num_img_joint, y), dim=0)

                what_is_number_and_confidence = what_is_num(what_is_num_img_joint, BATCH_SIZE, inputs_11, outputs_11, bindings_11, stream_11, classes_11, context_11, how_many_armor_exist)

                for arm_number, confidence in what_is_number_and_confidence:
                    if arm_number in small_armor_list and confidence >= conf_small_armor:
                        print("small  arm_number, confidence", arm_number, confidence)
                        realrealnum = realrealnum + 1
                        last_rectbox.append(what_is_num_rectbox[little_count])
                        last_LOCX.append(what_is_num_LOCX[little_count])
                        last_LOCY.append(what_is_num_LOCY[little_count])
                        last_dis.append(what_is_num_dis[little_count])
                        cv2.putText(img, arm_number,(int(what_is_num_LOCX[little_count]), int(what_is_num_LOCY[little_count])), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
                    elif arm_number in big_armor_list and confidence >= conf_big_armor:

                        print("big   arm_number, confidence", arm_number, confidence)
                        realrealnum = realrealnum + 1
                        what_is_num_rectbox[little_count] = big_coor2small_coor(what_is_num_rectbox, little_count)
                        last_rectbox.append(what_is_num_rectbox[little_count])
                        last_LOCX.append(what_is_num_LOCX[little_count])
                        last_LOCY.append(what_is_num_LOCY[little_count])
                        last_dis.append(what_is_num_dis[little_count])
                        cv2.putText(img, arm_number, (int(what_is_num_LOCX[little_count]), int(what_is_num_LOCY[little_count])), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
                    else:
                        little_count = little_count + 1
                        continue
                    little_count = little_count + 1
                t55 = cv2.getTickCount()
                spendTime = (t55 - t44) * 1 / (cv2.getTickFrequency())
                print("推理", spendTime)
                if realrealnum != 0:
                    disCalcCache = last_dis[0]
                    real_targetNum = 0  # 存储距离准星最进的装甲板编号
                    for count in range(0, realrealnum):
                        if last_dis[count] < disCalcCache:
                            real_targetNum = count
                            disCalcCache = last_dis[count]

                    real_rectbox = last_rectbox[real_targetNum]

                    if 1:
                        cv2.line(img, (int(last_LOCX[real_targetNum]), int(last_LOCY[real_targetNum])), (sightX, sightY), (255, 0, 255),
                                 2)  # 画指向线

    if real_rectbox:
        predicted_X, predicted_Y, predicted_Z, positions = PNP(real_rectbox)
        print(predicted_X, predicted_Y, predicted_Z)
    return img


out = WriteVideo(video_size)

while cap.isOpened():
    ret, img = cap.read()
    img = cv2.resize(img, (int(img.shape[1] / img_shrink_scale), int(img.shape[0] / img_shrink_scale)), interpolation=cv2.INTER_AREA)
    img = SHOW(img2coor_autoaim, img)
    if out != 0:
        out.write(cv2.resize(img, video_size))
        # out.write(img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()  # 释放视频
out.release()
cv2.destroyAllWindows()  # 释放所有的显示窗口






