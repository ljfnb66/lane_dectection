# 调库
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 引用其他模块
import main

""" 这一部分是图像预处理的过程，包括灰度化，高斯滤波，高斯模糊，canny边缘提取 """


# 灰度图转换 处理单张照片
def img_gray(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_gray

# 二值化
def img_binary(img):
    ret,img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return img_binary


# canny边缘检测
def img_canny(img):
    # 低阈值
    low_threshold = 15
    # 高阈值
    high_threshold = 145
    out_img = cv2.Canny(img, low_threshold, high_threshold)
    return out_img


# 获取 HSI 模型中的色调 H
def get_hsi_h(color):
    B = color[0]
    G = color[1]
    R = color[2]
    CMax = max(R, G, B)
    CMin = min(R, G, B)

    delta = CMax - CMin
    rt_val = 0

    if delta == 0:
        return 0
    elif CMax == R:
        GB = (int(G) - int(B)) / delta
        rt_val = GB * 60
        if G < B:
            rt_val += 360
    elif CMax == G:
        BR = (int(B) - int(R)) / delta
        rt_val = BR * 60 + 120
    elif CMax == B:
        RG = (int(R) - int(G)) / delta
        rt_val = RG * 60 + 240

    return rt_val

# 获取 HSI 模型中的色调 s
def get_hsi_s(color):
    B = color[0]
    G = color[1]
    R = color[2]
    mx = max(R, G, B)
    mi = min(R, G, B)
    s = (mx-mi)/mx

    return s


# 中值滤波
def median_filter(image, ksize):
    height, width, channels = image.shape
    filtered_image = np.zeros_like(image)

    padding = ksize // 2

    for y in range(padding, height - padding):
        for x in range(padding, width - padding):
            window = image[y - padding:y + padding + 1, x - padding:x + padding + 1]

            # 分离窗口中的红、绿、蓝通道
            b, g, r = cv2.split(window)

            # 将每个通道的像素值按从小到大排序
            b_sorted = np.sort(b.flatten())
            g_sorted = np.sort(g.flatten())
            r_sorted = np.sort(r.flatten())

            # 计算中值
            b_median = b_sorted[len(b_sorted) // 2] if len(b_sorted) % 2 == 1 else (b_sorted[len(b_sorted) // 2 - 1] + b_sorted[len(b_sorted) // 2]) // 2
            g_median = g_sorted[len(g_sorted) // 2] if len(g_sorted) % 2 == 1 else (g_sorted[len(g_sorted) // 2 - 1] + g_sorted[len(g_sorted) // 2]) // 2
            r_median = r_sorted[len(r_sorted) // 2] if len(r_sorted) % 2 == 1 else (r_sorted[len(r_sorted) // 2 - 1] + r_sorted[len(r_sorted) // 2]) // 2

            # 将中值赋给对应位置的像素点
            filtered_image[y, x] = (b_median, g_median, r_median)

    return filtered_image

