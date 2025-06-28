import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


# 现在有一个值result，我希望找到它与H_values 中的元素距离最小的一个，返回该元素的索引
def find_nearest_index(result, H_values):
    min_distance = math.inf  # 初始化最小距离为正无穷大
    nearest_index = -1  # 初始化最近元素的索引为-1

    # 遍历H_values列表
    for i, value in enumerate(H_values):
        distance = abs(value - result)  # 计算当前元素与result的距离

        if distance < min_distance:
            min_distance = distance
            nearest_index = i

    return nearest_index+1