# 调库
import cv2
import numpy as np

# 引用其他模块
import main

"""" 获取感兴趣的区域，这一段是针对车道检测的部分 """

# 车道检测
# 根据车的方向，构建一个梯形和三角形区域，消除四周的背景干扰
def roi_trapezoid(image):
    # 生成感兴趣区域即Mask掩模
    mask = np.zeros_like(image)  # 生成图像大小一致的mask矩阵

    row = image.shape[0]  # 行 y  height
    line = image.shape[1]  # 列 x  width
    print('row = ', row)
    print('line = ', line)

    # 填充顶点vertices中间区域
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # 定义多边形的顶点坐标 1280*720
    # 缩放为 640*360
    # 1   4
    # 2   3                         1          2          3           4
    # vertices = np.array([[250, 200], [60,355], [600, 355], [400, 200]], np.int32)

    # 定义多边形的顶点坐标 1280*720
    # 缩放为 640*360
    # 1   4
    # 2   3                           1          2                3           4
    vertices = np.array([[0, row*0.65], [0, row], [line, row], [line, row*0.65]], np.int32)

    # 将多边形的顶点数组组合成列表
    polygons = [vertices]

    # 填充梯形区域
    cv2.fillPoly(mask, polygons, ignore_mask_color)
    # 目标区域提取：逻辑与
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

# 针对路沿检测
# 生成ROI感兴趣区域即Mask掩模
def region_of_interest(image):
    mask = np.zeros_like(image)  # 生成图像大小一致的空白掩模 mask,np.zeros_like 函数生成一个与输入图像大小一致的全零矩阵

    # 填充顶点vertices中间区域
    # 判断输入图像的维度数，如果大于2，则表示输入图像是一个彩色图像，需要考虑通道数
    if len(image.shape) > 2:
        # 对于彩色图像，使用通道数来生成一个颜色元组
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    # 否则，表示输入图像是一个灰度图像
    else:
        # 对于灰度图像，直接使用灰度值 255
        ignore_mask_color = 255

    row = image.shape[0]  # 行 y  height
    line = image.shape[1]  # 列 x  width
    # 定义多边形的顶点坐标 1280*720
    # 缩放为 640*360
    # 1   4
    # 2   3                           1                 2                3                  4
    vertices = np.array([[line*0.4, row*0.1], [line*0.5, row], [line*0.9, row], [line*0.5, row*0.1]], np.int32)

    # 将多边形的顶点数组组合成列表
    polygons = [vertices]

    # 填充函数
    # 根据给定的顶点坐标 vertices，将掩模 mask 中对应的区域填充为忽略掩模颜色
    cv2.fillPoly(mask, polygons, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    # 将输入图像 image 和掩模 mask 进行按位与运算，得到在感兴趣区域内的图像
    return masked_image