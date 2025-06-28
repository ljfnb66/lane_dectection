# 调库
import cv2
import numpy as np

# 引用其他模块
import main

""" 对图像进行霍夫变换 """

#   全局变量
white = "White"
yellow = "Yellow"

# Canny边缘检测
def canny(image):
    # canny算子提取边缘
    # 低阈值
    low_threshold = 75
    # 高阈值
    high_threshold = 225
    return cv2.Canny(image, low_threshold, high_threshold)


# 霍夫变换
def hough_lines(img, rho, theta, hof_threshold, min_line_len, max_line_gap):
    """"
     rho  # 霍夫像素单位
     theta   # 霍夫角度移动步长
     hof_threshold   # 霍夫平面累加阈值threshold
     min_line_len   # 线段最小长度
     max_line_gap   # 最大允许断裂长度
    """
    lines = cv2.HoughLinesP(img, rho, theta, hof_threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines


# 筛选斜率在0.5到1.2之间的直线
def select_line(lines):
    slope_min = 0.5  # 斜率低阈值
    slope_max = 1.2  # 斜率高阈值
    filtered_lines = []
    for line in lines:
        # 获取线段的起点和终点坐标
        x1, y1, x2, y2 = line[0]
        # 计算斜率
        slope = (y2 - y1) / (x2 - x1)
        # 判断斜率是否在0.5到1.2之间
        if slope_min <= abs(slope) <= slope_max:
            print(slope)
            filtered_lines.append(line)
    return filtered_lines


# 绘制车道线，将经过hough变换之后检测到的直线绘制出来
def draw_chedao_lines(image, lines, thickness):
    # 分别用于存储右侧车道线的y坐标、x坐标和斜率
    right_y_set = []
    right_x_set = []
    right_slope_set = []
    # 存储左侧车道线的相关信息
    left_y_set = []
    left_x_set = []
    left_slope_set = []

    # slope_min = .25  # 斜率低阈值
    # slope_max = .90  # 斜率高阈值
    slope_min = 0.5  # 斜率低阈值
    slope_max = 1.2  # 斜率高阈值
    middle_x = image.shape[1] / 2  # 图像中线x坐标   image.shape[1]是图像的列数
    max_y = image.shape[0]  # 最大y坐标  image.shape[0]是图像的行数

    for line in lines:       # 迭代检测到的直线集合lines
        for x1, y1, x2, y2 in line:     # 遍历每条直线的端点坐标 (x1, y1, x2, y2)
            fit = np.polyfit((x1, x2), (y1, y2), 1)  # 拟合成直线
            slope = fit[0]  # 斜率

            # print('x1 = ', x1, ' x2 = ', x2, ' y1 = ', y1, ' y2 = ', y2 ,'\n')
            print('slope = ', slope)
                # 比较斜率是否在阈值范围内，判断直线是属于左侧车道线还是右侧车道线
            if slope_min < np.absolute(slope) <= slope_max:
                # 将斜率大于0且线段X坐标在图像中线右边的点存为右边车道线
                if slope > 0 and x1 > middle_x and x2 > middle_x:
                    right_y_set.append(y1)
                    right_y_set.append(y2)
                    right_x_set.append(x1)
                    right_x_set.append(x2)
                    right_slope_set.append(slope)
                # 将斜率小于0且线段X坐标在图像中线左边的点存为左边车道线
                # elif slope < 0 and x1 < middle_x and x2 < middle_x:
                elif slope < 0:
                    left_y_set.append(y1)
                    left_y_set.append(y2)
                    left_x_set.append(x1)
                    left_x_set.append(x2)
                    left_slope_set.append(slope)

    # BGR
    left_line_color = (1, 255, 0)   # 左车道颜色 绿色
    right_line_color = (255, 81, 22)  # 右车道颜色  蓝色

    # 绘制左车道线
    if left_y_set:
        lindex = left_y_set.index(min(left_y_set))  # 在左侧车道线的y坐标集合中找到最高点的索引 lindex
        # 获取对应的x坐标 left_x_top 和y坐标 left_y_top
        left_x_top = left_x_set[lindex]
        left_y_top = left_y_set[lindex]
        lslope = np.median(left_slope_set)  # 计算左侧车道线的平均斜率 lslope
        # 根据斜率计算车道线与图片下方交点作为起点
        left_x_bottom = int(left_x_top + (max_y - left_y_top) / lslope)
        # 绘制线段
        cv2.line(image, (left_x_bottom, max_y), (left_x_top, left_y_top), left_line_color, thickness)

    # 绘制右车道线
    if right_y_set:
        rindex = right_y_set.index(min(right_y_set))  # 最高点
        right_x_top = right_x_set[rindex]
        right_y_top = right_y_set[rindex]
        rslope = np.median(right_slope_set)
        # 根据斜率计算车道线与图片下方交点作为起点
        right_x_bottom = int(right_x_top + (max_y - right_y_top) / rslope)
        # 绘制线段
        cv2.line(image, (right_x_top, right_y_top), (right_x_bottom, max_y), right_line_color, thickness)

# 实线和虚线判断 原博客内容 用于word的 6.2章节代码
# def solid_dotted_judge(frame, lines):
#     if len(lines) > 0:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             fit = np.polyfit((x1, x2), (y1, y2), 1)
#             slope = fit[0]
#             k = abs(y2 - y1)
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             # 右车道线
#             if slope > 0:
#                 if k >= 60:
#                     cv2.putText(frame, 'Right Solid', (int((x1 + x2) / 2), int((y1 + y2) / 2)), font, 0.5, (0, 255, 0), 1)
#                 else:
#                     cv2.putText(frame, 'Right Dotted', (int((x1 + x2) / 2), int((y1 + y2) / 2)), font, 0.5, (0, 255, 0), 1)
#             # 左车道线
#             else:
#                 if k >= 60:
#                     cv2.putText(frame, 'Left Solid', (int((x1 + x2) / 2), int((y1 + y2) / 2)), font, 0.5, (0, 255, 0), 1)
#                 else:
#                     cv2.putText(frame, 'Left Dotted', (int((x1 + x2) / 2), int((y1 + y2) / 2)), font, 0.5, (0, 255, 0), 1)


# 实线和虚线判断，扩展了颜色识别，用于 6.3子章节代码
def solid_dotted_judge(img, lines):
    threshold = 60  # 判断线段长度
    font_size = 1   # 字体粗细
    if len(lines) > 0:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            k = abs(y2 - y1)
            font = cv2.FONT_HERSHEY_SIMPLEX

            # 判断出来的车道颜色
            line_color = detect_lane_color(img, line[0])
            if line_color == 'Unknown':
                continue

            # 字体颜色
            font_yellow_color = (255, 239, 59)
            font_white_color = (252, 253, 237)

            # 文字位置
            text_position_x = int((x1 + x2) / 2 - 55)
            text_position_y = int((y1 + y2) / 2)
            text_position = (text_position_x, text_position_y)

            print('slope = ', slope, 'line_color= ', line_color, ' text_position=', text_position)


            # 右车道线
            if slope > 0:
                if k >= threshold:
                    text = line_color+' Right solid'
                    cv2.putText(img, text,
                                text_position, font, 0.5,
                                font_white_color if line_color == 'White' else font_yellow_color,
                                font_size)
                else:
                    text = line_color+' Right Dotted'
                    cv2.putText(img, text,
                                text_position, font, 0.5,
                                font_white_color if line_color == white else font_yellow_color,
                                font_size)
            # 左车道线
            else:
                if k >= threshold:
                    text = line_color+' Left Solid'
                    if line_color == white:continue
                    cv2.putText(img, text,
                                text_position, font, 0.5,
                                font_white_color if line_color == white else font_yellow_color,
                                font_size)
                else:
                    if line_color == white: continue
                    text = line_color+' Left Dotted'
                    cv2.putText(img, text,
                                text_position, font, 0.5,
                                font_white_color if line_color == white else font_yellow_color,
                                font_size)


# 检测车道的颜色
def detect_lane_color(image, line):
    # 将图像转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 提取直线上的像素点坐标
    x1, y1, x2, y2 = line
    points = []
    for i in range(11):
        x = int(x1 + (x2 - x1) * i / 11)
        y = int(y1 + (y2 - y1) * i / 11)
        points.append((x, y))

    # 统计直线上黄色和白色像素点的数量
    yellow_count = 0
    white_count = 0
    for point in points:
        x, y = point
        h, s, v = hsv_image[y, x]
        if s > 45:
            yellow_count += 1
        elif s <= 30:
            white_count += 1

    # 判断直线的颜色
    if yellow_count > 6:
        return yellow
    elif white_count > 6:
        return white
    else:
        return "Unknown"




# def color_judge(frame, frame1, lines):
#     frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     L = []
#     if len(lines) > 0:
#         for i in range(len(lines)):
#             L.append(lines[i])
#         for line in L:
#             # print(line)
#             slope, x1, y1, x2, y2= line
#             p = 0
#             for i in range(1, 11):
#                 y = int(y1 + (y2 - y1) * i / 11)
#                 x = int(x1 + (x2 - x1) * i / 11)
#                 # print(x, y)
#                 h, s, v = frame2[y][x]
#                 if s > 40:
#                     p += 1
#             if p >= 3:
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 cv2.putText(frame1, 'Yellow', (int((x1 + x2) / 2), int((y1 + y2) / 2)), font, 0.5, (0, 255, 255), 2)
#             else:
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 cv2.putText(frame1, 'White', (int((x1 + x2) / 2), int((y1 + y2) / 2)), font, 0.5, (255, 255, 255), 2)





# 绘制路沿线，将经过hough变换之后检测到的直线绘制出来
def draw_luyan_lines(image, lines, color=[0, 0, 255], thickness=2):
    line_image = np.zeros_like(image)  # 创建与原始图像大小一致的全零图像

    # 保存中间路沿直线的相关信息
    target_line = None
    target_line_x_diff = 0

    # 遍历所有直线
    for line in lines:
        x1, y1, x2, y2 = line[0]

        # 计算直线斜率
        slope = (y2 - y1) / (x2 - x1)

        # 计算直线的中点坐标
        mid_x = (x1 + x2) // 2

        # 判断直线是否在中间位置，即判断 mid_x 是否位于图像宽度的 1/3 到 2/3 范围
        if mid_x > image.shape[1] // 3 and mid_x < (2 * image.shape[1] // 3):

            # 如果直线满足中间位置的条件，则更新路沿直线为具有最大 x 轴差值的直线
            x_diff = abs(x2 - x1)
            if target_line is None or x_diff > target_line_x_diff:
                target_line = line
                target_line_x_diff = x_diff

    # 绘制路沿直线
    if target_line is not None:
        x1, y1, x2, y2 = target_line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), color, thickness=thickness)

    return line_image


""""
# 绘制路沿线，将经过hough变换之后检测到的直线绘制出来
def draw_luyan_lines(image, lines, color=[0, 0, 255], thickness=2):
    line_image = np.zeros_like(image)  # 生成与原始图像大小一致的zeros矩,具有相同的行数、列数和通道数
    # 定义点的颜色
    # color = (0, 0, 255)  # 红色，以 (B, G, R) 的顺序表示

    slope_threshold_low = 1.5     # 低阈值
    slope_threshold_high = 4  # 高阈值

    # 在图片上绘制直线
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 计算直线斜率
        slope = (y2 - y1) / (x2 - x1)
        # 判断直线斜率是否在阈值范围内
        if slope_threshold_low <= abs(slope) <= slope_threshold_high:
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness=thickness)

        # 绘制点
        # cv2.circle(line_image, (x1, y1), radius=2, color=color, thickness=-1)
        # cv2.circle(line_image, (x2, y2), radius=2, color=color, thickness=-1)

    return line_image

"""

# 显示经过霍夫变换检测出来的直线
def img_detect_line(img):
    rho = 1  # 霍夫像素单位
    theta = np.pi / 180  # 霍夫角度移动步长
    hof_threshold = 20  # 霍夫平面累加阈值threshold
    min_line_len = 30  # 线段最小长度
    max_line_gap = 60  # 最大允许断裂长度

    # 基于霍夫变换的直线检测
    lines = hough_lines(img, rho, theta, hof_threshold, min_line_len, max_line_gap)
    line_image = np.zeros_like(img)

    # 绘制车道线线段
    draw_chedao_lines(line_image, lines, thickness=2)

    return line_image

# 原图像与车道线图像按照a:b比例融合
def weighted_img(img_after_gaussian, line_image):
    alpha = 0.8  # 原图像权重
    beta = 1  # 车道线图像权重
    lambda_ = 0
    blended_image = cv2.addWeighted(img_after_gaussian, alpha, line_image, beta, lambda_)
    return blended_image







