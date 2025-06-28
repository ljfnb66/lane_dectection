# 调库
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image, ImageDraw, ImageFont



# 引用其他模块
import ph_color_detect_aa as aa
import img_process_bb as bb

# 全局变量
img_read_path = 'D:\\AllCode\\python\\ph\\img\\'
logo = '-ljf-632109160602'  # 个人logo
# 设置窗口大小
window_width = 300
window_height = 400

phColor = []  # B  G   R        ph
phColor.append([41, 46, 191])     # 1
phColor.append([32, 79, 232])    # 2
phColor.append([35, 79, 226])    # 3
phColor.append([23, 109, 239])   # 4
phColor.append([4, 139, 253])   # 5
phColor.append([0, 178, 255])  # 6
phColor.append([21, 159, 202])  # 7
phColor.append([44, 151, 158])   # 8
phColor.append([45, 112, 115])   # 9
phColor.append([34, 42, 41])     # 10
phColor.append([69, 52, 55])     # 11
phColor.append([69, 52, 61])    # 12
phColor.append([57, 36, 68])    # 13
phColor.append([40, 28, 48])    # 14

# 给H 赋值
H_values = []
def init_h_values():
    # 初始化H_values为包含14个空列表的列表
    for _ in range(14):
        H_values.append([])

    # 不同ph对应的H分量
    for i in range(14):
        # 获取H分量的值
        H_values[i].append(bb.get_hsi_h(phColor[i]))
        print("ph值为", i + 1, "的试纸对应的H分量为：", H_values[i])

# 现在有一个值result，找到它与H_values 中的元素距离最小的一个，返回该元素的索引
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


# 纯视觉ph检测
def ph_color_detect():
    # 绘制标准比色卡
    standard_ph_img = aa.genPhColorPlate(phColor)
    cv2.imshow('standard_ph_img' + logo, standard_ph_img)

    # # 顺序为   B    G    R
    # ph_color = [83, 79, 200]  # 待检测的颜色ph_color1
    # ph_color = [34, 115, 255]  # 待检测的颜色ph_color1
    # ph_color = [109, 255, 221]  # 待检测的颜色ph_color2
    # ph_color = [47, 197, 255]  # 待检测的颜色ph_color3
    ph_color = [32, 79, 232]  # 待检测的颜色ph_color3

    # 欧式距离计算
    ph_val1 = aa.getPhValueInt(ph_color, phColor)
    print('采用欧式距离计算，与标准比色卡最接近的ph值为：', ph_val1)

    # 采用插值法计算得到最接近的pH值ph_val
    ph_val = aa.getPhValueFloat(ph_color, phColor)
    print('采用插值法计算，与标准比色卡最接近的ph值为：', ph_val[0])
    print(ph_val[1])
    print(ph_val[2])

    # 在彩色条图像上绘制出检测到的pH值的标记，并将结果保存为图像文件
    ph_img = aa.drawPh(ph_color, ph_val, phColor)
    cv2.imshow('ph_img' + logo, ph_img)

    # cv2.imwrite("PH" + round(ph_val[0], 2).__str__() + ".png", ph_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 计算一个列表的平均值
def calculate_average(lst):
    if len(lst) == 0:
        return 0  # 处理空列表的情况，返回0或其他默认值

    total = sum(lst)  # 计算列表中所有元素的总和
    average = total / len(lst)  # 计算平均值
    return average

# 在图像上作出文字
def draw_image(image, text):
    image_ph = image.copy()
    # 设置文本和字体
    text = 'pH:'+text  # 替换为你要添加的文本
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 指定文本的左上角位置
    position = (10, 30)  # 这里(10, 30)是左上角的坐标，你可以根据需要调整

    # 设置文本的尺寸和颜色
    font_scale = 1
    color = (0, 0, 255)  # 红色
    thickness = 2

    # 添加文本到图像
    cv2.putText(image_ph, text, position, font, font_scale, color, thickness)

    return image_ph


# 第三章
def detect():
    # 1.读取彩色图像
    # image = cv2.imread(img_read_path+"ph8.jpg")
    # image = cv2.imread(img_read_path+"ph7(3).jpg")
    image = cv2.imread(img_read_path+"ph7(2).jpg")
    # image = cv2.imread(img_read_path+"ph10.jpg")
    image = cv2.resize(image, (120, 220))
    cv2.imshow("Image", image)

    height = image.shape[0]  # 行 y  height
    width = image.shape[1]  # 列 x  width
    print(height, width)

    # 2.中值滤波
    image_after_median = bb.median_filter(image, 3)
    cv2.imshow("image_after_median", image_after_median)

    # 3.将黄色区域删除
    low_threshold = 30
    high_threshold = 55
    image_del_yellow = image_after_median.copy()
    # 遍历图像中的每个像素[
    for x in range(height):
        for y in range(width):
            # 获取像素的RGB值
            b, g, r = image[x, y]
            color = (b, g, r)
            # 获取h
            h = bb.get_hsi_h(color)
            if( h > low_threshold and h < high_threshold):
                image_del_yellow[x, y] = (0, 0, 0)
    cv2.imshow("image_del_yellow", image_del_yellow)

    # 4.将周围区域删除
    image_del_round = image_del_yellow.copy()
    # 遍历图像中的每个像素
    for x in range(height):
        for y in range(width):
            # 获取像素的RGB值
            b, g, r = image[x, y]
            if b > 150 and g > 150 and r > 150:
                image_del_round[x, y] = (0, 0, 0)
    cv2.imshow("image_del_round", image_del_round)

    # 5.优化,去除目标区域边缘的部分
    image_youhua = image_del_round.copy()
    # 存储H分量的列表
    h_values = []
    # 遍历图像中的每个像素
    for x in range(height):
        for y in range(width):
            # 获取像素的RGB值
            b, g, r = image_youhua[x, y]

            # 全黑pass
            if b == 0 and g == 0 and r == 0:
                continue

            color = (b, g, r)
            # 获取h
            h = bb.get_hsi_h(color)

            # 将h分量添加到列表中
            h_values.append(h)

    # 将h分量列表转换为NumPy数组
    h_values = np.array(h_values)
    # 对列表进行排序
    h_values = sorted(h_values)
    print("原始长度：", int(len(h_values)))

    # 6.计算要去掉的元素个数
    remove_count = int(len(h_values) * 0.1)
    h_values_1 = []
    # 去掉前10%和后10%的元素
    for i in range(remove_count, len(h_values) - remove_count):
        h_values_1.append(h_values[i])
    print("去掉前10%和后10%的元素之后的长度", int(len(h_values_1)))
    # 计算列表的平均值
    h_average = calculate_average(h_values_1)
    print("优化之后的目标区域的平均H分量为：", h_average)

    # 7.画出频率分布直方图
    # 计算每个数据出现的次数
    frequency_distribution = {}
    for item in h_values_1:
        frequency_distribution[item] = frequency_distribution.get(item, 0) + 1

    # 将频率分布转换为可以绘图的格式
    items = list(frequency_distribution.keys())
    frequencies = list(frequency_distribution.values())

    # 8.找出出现次数最多的数据
    max_frequency = max(frequency_distribution.values())
    most_frequent_items = [item for item, frequency in frequency_distribution.items() if frequency == max_frequency]

    # 如果有多个数据出现次数相同，计算它们的平均值
    if len(most_frequent_items) > 1:
        result = sum(most_frequent_items) / len(most_frequent_items)
    else:
        result = most_frequent_items[0]

    print("优化之后的目标区域出现频率最高的H值是:", result)

    # 9.计算ph值
    ph_measure = find_nearest_index(result, H_values)
    print("该试纸的pH值为", ph_measure)
    ph_measure = str(ph_measure)

    # 作图
    image_ph = draw_image(image, ph_measure)  # 替换为你的图像路径
    cv2.imshow("image_ph", image_ph)




    # 绘制直方图
    plt.bar(items, frequencies)
    # 设置标题和轴标签
    plt.title('Frequency Distribution of H'+logo)
    plt.xlabel('the Data Values of H')
    plt.ylabel('H Frequency')

    # 显示直方图
    plt.show()


    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    print(logo)
    # img_process()

    # 纯视觉
    # ph_color_detect()

    init_h_values()

    detect()



    print("over\n")

if __name__=="__main__":
    main()

