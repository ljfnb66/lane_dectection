# 调库
import cv2
import numpy as np

# 引用其他模块
import main

""" 获得白色和黄色的车道线 """

# 在图像的中下部获得白色的车道线的最大值
def Get_White_Max_Value(img):
    max_val = 0

    imshape = img.shape  # 获取图像大小

    # 使用两个嵌套的循环遍历车道附近的行和列
    for i in range(int(img.shape[0]/2), img.shape[0]):
        for j in range(int(img.shape[1] * 0.2), int(img.shape[1] * 0.8)):
            t_min = 255
            # 每个像素位置，遍历颜色通道
            for c in range(3):
                val = img[i, j, c]
                if val < t_min:
                    t_min = val
            if t_min > max_val:
                max_val = t_min
    # 返回最大的白色值
    return max_val




# RGB_Value 是一个长度为3的整型数组，表示颜色阈值；radius 是颜色半径
# 做颜色切割
def do_color_slicing(image, rgb_value, radius):
    row_size, col_size, _ = image.shape
    # 嵌套的循环遍历图像的每个像素点
    for i in range(row_size):
        for j in range(col_size):
            # 计算当前像素点与给定颜色阈值之间的差值
            l_r = int(image[i, j, 2]) - rgb_value[0]
            l_g = int(image[i, j, 1]) - rgb_value[1]
            l_b = int(image[i, j, 0]) - rgb_value[2]
            # 如果平方和大于颜色半径的平方，则将该像素点的RGB分量设置为0，即将其切割掉
            if (l_r * l_r + l_g * l_g + l_b * l_b > radius * radius):
                image[i, j] = [0, 0, 0]
    return image


# 获得白色掩膜
def Get_White_Line(input_image):
    # 2.获取 img_after_gaussian 的白色线的最大值
    white_max = Get_White_Max_Value(input_image)
    white_max *= 0.9

    rgb_white = [white_max, white_max, white_max]
    color_radius = 70
    # 3.对 img_after_gaussian 进行颜色切割，阈值为 rgb_white ，颜色半径为 color_radius
    white_mask = do_color_slicing(input_image, rgb_white, color_radius)

    return white_mask



# 通过HSI模型得到满足要求的黄色线
def Get_Yellow_Line(input_image):
    output_image = np.copy(input_image)

    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            # 获取当前像素的 RGB 值
            R = output_image[i, j, 2]
            G = output_image[i, j, 1]
            B = output_image[i, j, 0]

            if(R == 0 and G == 0 and B == 0):
                # 纯黑色直接pass
                pass
            else:
                # 将 RGB 值转换为 HSI 模型中的 H（色调）和 S（饱和度）
                H = get_hsi_h(R, G, B)
                S = get_hsi_s(R, G, B)

                # 设置色调范围和饱和度阈值
                range_value = 3
                value = 45
                threshold = 0.40

                # 判断当前像素 的 色滴Hue 是否满足黄色线的条件
                if (S >= threshold and
                         # 42 48
                         (H >= value - range_value and H <= value + range_value)):
                    # print(H)
                    # 符合条件的保留，不做处理
                    pass
                else:
                    # 不符合条件的像素设为黑色
                    output_image[i, j] = [0, 0, 0]

    return output_image

# 获取 HSI 模型中的色调 H
def get_hsi_h(R, G, B):
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


# 获取 HSI 模型中的饱和度 S
def get_hsi_s(R, G, B):
    # 将其范围映射到 [0, 1]
    CMax = R / 255.0
    CMin = R / 255.0
    g = G / 255.0
    b = B / 255.0

    if g > CMax:
        CMax = g
    if b > CMax:
        CMax = b
    if g < CMin:
        CMin = g
    if b < CMin:
        CMin = b

    L = (CMax + CMin) / 2

    if CMax == CMin:
        return 0
    elif L >= 0.5:
        return (CMax - CMin) / (2.0 - CMax - CMin)
    else:
        return (CMax - CMin) / (CMax + CMin)


#  悬小小博客，判断黄色车道和白色车道
def color_judge(frame, frame1, lines):
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    L = []
    if len(lines) > 0:
        for i in range(len(lines)):
            L.append(lines[i])
        for line in L:
            # print(line)
            slope, x1, y1, x2, y2= line
            p = 0
            for i in range(1, 11):
                y = int(y1 + (y2 - y1) * i / 11)
                x = int(x1 + (x2 - x1) * i / 11)
                # print(x, y)
                h, s, v = frame2[y][x]
                if s > 40:
                    p += 1
            if p >= 3:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame1, 'Yellow', (int((x1 + x2) / 2), int((y1 + y2) / 2)), font, 0.5, (0, 255, 255), 2)
            else:
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame1, 'White', (int((x1 + x2) / 2), int((y1 + y2) / 2)), font, 0.5, (255, 255, 255), 2)





# 把两幅图像相加
def tow_img_overlay(image_white, image_yellow):
    out_img = np.zeros_like(image_white)  # 创建与 image1 相同大小的零矩阵作为结果

    for i in range(out_img.shape[0]):
        for j in range(out_img.shape[1]):
            # 白色掩膜的 rgb
            w_R = image_white[i, j, 2]
            w_G = image_white[i, j, 1]
            w_B = image_white[i, j, 0]
            # 黄色掩膜的 rgb
            y_R = image_yellow[i, j, 2]
            y_G = image_yellow[i, j, 1]
            y_B = image_yellow[i, j, 0]
            rs_R = (w_R + y_R)
            rs_G = (w_G + y_G)
            rs_B = (w_B + y_B)
            # 判断输出图像的rgb是否大于255
            rs_R = 255 if rs_R > 255 else rs_R
            rs_G = 255 if rs_G > 255 else rs_G
            rs_B = 255 if rs_B > 255 else rs_B
            # 赋值
            out_img[i, j, 2] = rs_R
            out_img[i, j, 1] = rs_G
            out_img[i, j, 0] = rs_B

    return out_img


# 闭运算
def img_close_operation(img):
    kernel_size = 6
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 膨胀操作
    dilation1 = cv2.dilate(img, kernel, iterations=1)

    # 腐蚀操作
    erosion = cv2.erode(dilation1, kernel, iterations=1)

    # 腐蚀操作
    # kernel_size = 1
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # erosion2 = cv2.erode(erosion, kernel, iterations=1)

    # 膨胀操作
    # kernel_size = 1
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # dilation2 = cv2.dilate(erosion, kernel, iterations=1)

    return erosion



# 得到经过改模块处理之后的img
def gain_lane_bb(img_after_gaussian):
    # 1.得到一个黄色线的掩膜 yellow_mask
    yellow_mask = Get_Yellow_Line(img_after_gaussian)

    # 2.得到一个白色线的掩膜 white_mask
    white_mask = Get_White_Line(img_after_gaussian)

    # 4. 将两幅图像相加
    out_image = tow_img_overlay(yellow_mask, white_mask)

    # 输出出去
    return out_image