# coding=utf-8
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt



# 实现了两个颜色之间的插值
# 将两个颜色的RGB通道按权重进行插值，生成一个新的颜色值
def interplateColor(color1, color2, weight=0.5):
    c1_1 = color1[0]
    c1_2 = color1[1]
    c1_3 = color1[2]

    c2_1 = color2[0]
    c2_2 = color2[1]
    c2_3 = color2[2]

    c3_1 = int((1 - weight) * c1_1 + weight * c2_1)
    c3_2 = int((1 - weight) * c1_2 + weight * c2_2)
    c3_3 = int((1 - weight) * c1_3 + weight * c2_3)
    return [c3_1, c3_2, c3_3]

# 生成显示不同pH值颜色的彩色条图像
# phColor是一个包含不同pH值对应颜色的列表
def genPhColorPlate(phColor):
    # 颜色条的宽度
    color_bar_width = 50
    # 颜色条的高度
    color_bar_height = 250
    # 颜色条之间的间距
    color_bar_margin = 20
    # 图像的高度
    height = 300
    # 图像的宽度
    width = len(phColor) * color_bar_width + (len(phColor) + 1) * color_bar_margin
    # 创建一个空白图像，全白
    blank_img = np.zeros((height, width, 3), np.uint8) + 255
    # 循环遍历phColor列表中的每个元素
    for i in range(len(phColor)):
        # 计算当前颜色条的中心位置center
        center = color_bar_margin + color_bar_width // 2 + i * (color_bar_width + color_bar_margin)
        # 将phColor[i]的颜色值填充到彩色条的相应位置上，从而在图像中显示了对应pH值的颜色条
        blank_img[color_bar_margin:color_bar_height,
                  center - color_bar_width // 2:center + color_bar_width // 2] = phColor[i]
        blank_img[:color_bar_height, center] = [0, 0, 255]
        # 在颜色条下方的位置添加文本，显示当前pH值。如果i大于等于9，则使用较小的字体和偏移量绘制文本，否则使用较大的字体和偏移量绘制文本
        if i >= 9:
            cv2.putText(blank_img, str(i + 1),
                        (center - 22, color_bar_height + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(blank_img, str(i + 1),
                        (center - 12, color_bar_height + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 0), 1, cv2.LINE_AA)
    return blank_img


# 根据给定的颜色值和pH颜色列表，计算出最接近的pH值
# 使用欧氏距离度量颜色值与pH颜色之间的距离，并返回最接近的pH值的整数
def getPhValueInt(color, phColor):
    dists = []
    for i in range(len(phColor)):
        dist = math.sqrt(
            pow(color[0] - phColor[i][0], 2) + pow(color[1] - phColor[i][1], 2) + pow(color[2] - phColor[i][2], 2))
        # print(dist)
        # dist = ColourDistance(color, phColor[i])
        dists.append(dist)
    return dists.index(min(dists)) + 1


# 加权欧式距离
def ColourDistance(rgb_1, rgb_2):
    R_1, G_1, B_1 = rgb_1
    R_2, G_2, B_2 = rgb_2
    rmean = (R_1 + R_2) / 2
    R = R_1 - R_2
    G = G_1 - G_2
    B = B_1 - B_2
    return math.sqrt((2 + rmean / 256) * (R ** 2) + 4 * (G ** 2) + (2 + (255 - rmean) / 256) * (B ** 2))


# 返回一个浮点值，表示在两个最接近的pH值之间的插值。这样可以更精确地计算pH值
# 通过颜色插值方法获取在两个最接近的pH值之间的插值
# color是一个包含RGB颜色值的列表，表示要进行插值的颜色，phColor是一个包含不同pH值对应颜色的列
def getPhValueFloat(color, phColor):
    # 存储color与每个phColor之间的距离
    dists = []
    # 循环遍历phColor列表
    for i in range(len(phColor)):
        # 计算color与每个phColor之间的欧氏距离
        dist = math.sqrt(
            pow(color[0] - phColor[i][0], 2) + pow(color[1] - phColor[i][1], 2) + pow(color[2] - phColor[i][2], 2))
        # 将距离添加到dists列表中
        dists.append(dist)

    # 找到dists列表中距离最小的元素的索引，即与color最接近的phColor的索引
    min_index1 = dists.index(min(dists))
    # 索引赋值
    dist1 = dists[min_index1]

    # 设为一个较大的值,以便下一步查找次小距离
    dists[min_index1] = 999999

    # 找到列表中距离次小的元素的索引
    min_index2 = dists.index(min(dists))
    dist2 = dists[min_index2]

    # 通过插值计算出在这两个最接近的pH值之间的插值
    if min_index1 <= min_index2:
        final_ph = min_index1 + abs(min_index1 - min_index2) * (dist1 / (dist1 + dist2))
    else:
        final_ph = min_index1 - abs(min_index1 - min_index2) * (dist1 / (dist1 + dist2))
    return final_ph + 1, min_index1 + 1, min_index2 + 1


# 用于在彩色条图像上绘制检测到的pH值
# 函数会在彩色条图像上标注检测到的pH值，并将对应的颜色标记出来
def drawPh(ph_color, ph_val, phColor):
    ph_img = genPhColorPlate(phColor)
    color_bar_width = 50
    color_bar_margin = 20
    color_bar_height = 150
    # 将待标记的pH值减去1
    ph = ph_val[0] - 1
    # 将最接近的两个已知pH值减去1，赋值给变量ph1和ph2
    ph1 = ph_val[1] - 1
    ph2 = ph_val[2] - 1
    # ph1表示较小的pH值，ph2表示较大的pH值
    if ph1 > ph2:
        tmp = ph1
        ph1 = ph2
        ph2 = tmp

    # 计算开始位置和结束位置的像素值,使用了线性插值的思想
    start_pix = color_bar_margin + color_bar_width / 2 + ph1 * (color_bar_width + color_bar_margin)
    end_pix = color_bar_margin + color_bar_width / 2 + ph2 * (color_bar_width + color_bar_margin)

    loc = int((ph - ph1) * (end_pix - start_pix) + start_pix)
    # ph_img[:, loc - 1:loc + 1] = ph_color
    ph_img[:, loc - 1:loc + 1] = (0,0,0)

    cv2.putText(ph_img, "PH=" + round(ph_val[0], 2).__str__(),
                (loc + 1, color_bar_height + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 0), 2, cv2.LINE_AA)
    return ph_img


if __name__ == '__main__':
    phColor = []  # B  G   R
    phColor.append([0, 4, 206])
    phColor.append([4, 41, 253])
    phColor.append([7, 64, 250])
    phColor.append([79, 83, 255])
    phColor.append([4, 112, 254])
    phColor.append([47, 171, 255])
    phColor.append([93, 216, 196])
    phColor.append([5, 201, 118])
    phColor.append([241, 59, 17])
    phColor.append([163, 0, 1])
    phColor.append([110, 3, 0])
    phColor.append([78, 25, 28])

    # 顺序为BGR
    ph_color = [79, 83, 200]    #   待检测的颜色ph_color

    # 调用getPhValueFloat函数计算得到最接近的pH值ph_val
    ph_val = getPhValueFloat(ph_color, phColor)
    print ('ph', ph_val[0])
    print ('index1', ph_val[1])
    print ('index2', ph_val[2])

    # 在彩色条图像上绘制出检测到的pH值的标记，并将结果保存为图像文件
    ph_img = drawPh(ph_color, ph_val, phColor)
    cv2.imshow('ph_img', ph_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    cv2.imwrite("PH" + round(ph_val[0], 2).__str__() + ".png", ph_img)
