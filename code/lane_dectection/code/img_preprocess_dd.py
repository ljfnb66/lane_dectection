# 调库
import cv2
import numpy as np
from matplotlib import pyplot as plt

# 引用其他模块
import main

""" 这一部分是图像预处理的过程，包括灰度化，高斯滤波，高斯模糊，canny边缘提取 """


# 直方图均衡化
def histogram_equalization(img):
    (b, g, r) = cv2.split(img)  # 通道分解
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge((bH, gH, rH), )  # 通道合成
    # res = np.hstack((img, result))
    return result




# 灰度图转换 处理多张照片
def grayscale(num_img):
    for i in range(num_img):
        filename = main.origin_img_save_path + 'img' + str(i) + '.jpg'
        img = cv2.imread(filename)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        filename = main.gray_img_save_path + 'img_gray' + str(i) + '.jpg'
        cv2.imwrite(filename, img_gray)

# 灰度图转换 处理单张照片
def img_gray(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_gray

# 自定义的高斯滤波
def GaussianFilter(img):
    h, w, c = img.shape
    # 高斯滤波
    K_size = 3
    sigma = 1.3

    # 零填充
    pad = K_size // 2
    out = np.zeros((h + 2 * pad, w + 2 * pad, c), dtype=float) # np.float 改为float
    out[pad:pad + h, pad:pad + w] = img.copy().astype(float)

    # 定义滤波核
    K = np.zeros((K_size, K_size), dtype=float)

    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
    K /= (sigma * np.sqrt(2 * np.pi))
    K /= K.sum()

    # 卷积的过程
    tmp = out.copy()
    for y in range(h):
        for x in range(w):
            for ci in range(c):
                out[pad + y, pad + x, ci] = np.sum(K * tmp[y:y + K_size, x:x + K_size, ci])

    out_image = out[pad:pad + h, pad:pad + w].astype(np.uint8)

    return out_image


# 将all灰度图片进行高斯滤波之后保存
def Gaussian_img_save(num_img):
    for i in range(num_img):
        read_filename = main.gray_img_save_path + 'img_gray' + str(i) + '.jpg'
        img = cv2.imread(read_filename)
        img_gaussian_after = GaussianFilter(img)
        write_filename = main.gaussian_img_save_path + 'img_gaussian' + str(i) + '.jpg'
        cv2.imwrite(write_filename, img_gaussian_after)


# 将all经过高斯滤波之后的图片进行Canny边缘检测之后保存
def Canny_img_save(num_img):
    for i in range(num_img):
        read_filename = main.gaussian_img_save_path + 'img_gaussian' + str(i) + '.jpg'
        img = cv2.imread(read_filename)
        # src——输入图像，low_threshold ——低阈值，high_threshold——高阈值
        img_canny_after = cv2.Canny(img, 75, 225)
        write_filename = main.canny_img_save_path + 'img_canny' + str(i) + '.jpg'
        cv2.imwrite(write_filename, img_canny_after)

# canny边缘检测
def img_canny(img):
    # 低阈值
    low_threshold = 75
    # 高阈值
    high_threshold = 225
    out_img = cv2.Canny(img, low_threshold, high_threshold)
    return out_img


