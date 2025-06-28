# 调库
import cv2
import numpy as np
from PIL import ImageEnhance

# 引用其他模块
import lane_white_yellow_bb as bb
import video_img_convert_aa as aa
import img_preprocess_dd as dd
import img_roi_gain_cc as cc
import img_hough_transform_ee as ee



# 全局变量
# 视频路径
# video_path = "D:\\AllCode\\python\\lane_dectection\\vedio\\project_video.mp4"
# # 图片保存地址
# origin_img_save_path = "D:\\AllCode\\python\\lane_dectection\\all_image\\image\\"


# 以下是测试使用的 5s video，用于验证实验
# 全局变量
# 视频路径
video_path = "D:\\AllCode\\python\\lane_dectection\\test_image\\vedio\\video_5s.mp4"
# 原始图片保存地址
origin_img_save_path = "D:\\AllCode\\python\\lane_dectection\\test_image\\image\\origin_image\\"
# 灰度图片保存地址
gray_img_save_path = "D:\\AllCode\\python\\lane_dectection\\test_image\\image\\gray_image\\"
# 将灰度图片进行高斯滤波之后的保存地址
gaussian_img_save_path = "D:\\AllCode\\python\\lane_dectection\\test_image\\image\\gaussian_image\\"
# 将经过高斯滤波之后的照片进行Canny边缘检测之后的保存地址
canny_img_save_path = "D:\\AllCode\\python\\lane_dectection\\test_image\\image\\Canny_image\\"


# 原始图片保存地址
other_img_save_path = "D:\\AllCode\\python\\lane_dectection\\test_image\\image\\other\\"

# 以下是测试使用的 8s 有路沿的video，用于验证实验
video_luyan_path = "D:\\AllCode\\python\\lane_dectection\\test_image\\vedio\\video_luyan.mp4"
img_luyan_save_path = "D:\\AllCode\\python\\lane_dectection\\test_image\\img_luyan\\py_img\\"

# 个人logo
logo = '-ljf-632109160602'

# 设置窗口大小
window_width = 640
window_height = 480


# 直方图均衡化，为了判断效果
def histogram_equalization_test():
    # 原始图片
    img = cv2.imread(other_img_save_path + '22.jpg')
    img = cv2.resize(img, (window_width, window_height))
    cv2.imshow('img'+logo, img)
    # 直方图均衡化的图片
    img_after_histogram_equalization = dd.histogram_equalization(img)
    cv2.imshow('img_after_histogram_equalization'+logo, img_after_histogram_equalization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 悬小小博客，判断黄色车道和白色车道
def test_y_w():
    # 1.读取图像
    # img_origin = cv2.imread(origin_img_save_path + 'img36.jpg')
    img_origin = cv2.imread(other_img_save_path + '22.jpg')
    img_origin = cv2.resize(img_origin, (window_width, window_height))
    cv2.imshow('img_origin'+logo, img_origin)

    # 4.获得白色掩膜
    img_white_roi = bb.Get_White_Line(img_origin)
    cv2.imshow('img_white_roi'+logo, img_white_roi)


    cv2.waitKey(0)
    cv2.destroyAllWindows()





# 检测路沿
def show_luyan_img():
    window_width = 640
    window_height = 480

    # 1.读取图像
    img_origin = cv2.imread(img_luyan_save_path + 'luyan1.jpg')
    img_origin = cv2.resize(img_origin, (window_width, window_height))
    # cv2.imshow('img_origin'+logo, img_origin)

    # 2.进行高斯模糊
    img_after_gaussian = dd.GaussianFilter(img_origin)
    # cv2.imshow('img_after_gaussian'+logo, img_after_gaussian)

    # 3.图像转化（彩->灰）
    img_after_gray = dd.img_gray(img_after_gaussian)
    # cv2.imshow('img_after_gray'+logo, img_after_gray)


    # 4.Canny边缘检测
    img_after_canny = dd.img_canny(img_after_gray)
    # cv2.imshow('img_after_canny' + logo, img_after_canny)

    # 5.提取ROI
    img_after_roi = cc.region_of_interest(img_after_canny)
    cv2.imshow('img_after_roi' + logo, img_after_roi)

    # 6. 将霍夫变换的直线检测出来
    rho = 0.25  # 霍夫像素单位
    theta = np.pi / 180  # 霍夫角度移动步长
    hof_threshold = 20  # 霍夫平面累加阈值threshold
    min_line_len = 100   # 线段最小长度
    max_line_gap = 8   # 最大允许断裂长度
    # lines = cv2.HoughLinesP(img_after_canny, 1, np.pi / 180, 100, np.array([]), minLineLength=200, maxLineGap=3)  # 过度判断
    # lines = cv2.HoughLinesP(img_after_canny, 1, np.pi / 180, 100, np.array([]), minLineLength=100, maxLineGap=3)  # 过度检测(未筛选)
    # lines = cv2.HoughLinesP(img_after_roi, 0.25, np.pi / 180, 20, np.array([]), minLineLength=120, maxLineGap=8)  # 效果最好
    # lines = cv2.HoughLinesP(img_after_roi, 0.25, np.pi / 180, 20, np.array([]), minLineLength=40, maxLineGap=5)  # 测试
    lines = ee.hough_lines(img_after_roi, rho, theta, hof_threshold, min_line_len, max_line_gap)  # 过度检测(未筛选)
    print(logo)
    print('李骏飞')
    print(lines)


    # 7.绘制到经过高斯滤波后的图片上,并筛选直线
    img_after_hough = ee.draw_luyan_lines(img_after_gaussian, lines, color=[0, 0, 255], thickness=3)
    cv2.imshow('img_after_hough'+logo, img_after_hough)

    # 图像融合
    img_mix = ee.weighted_img(img_after_gaussian, img_after_hough)
    img_mix = cv2.resize(img_mix, (window_width, window_height))
    cv2.imshow('img_mix'+logo, img_mix)


    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_chedao_img():

    # 1.读取图像
    # img_origin = cv2.imread(origin_img_save_path + 'img0.jpg')
    img_origin = cv2.imread(origin_img_save_path + 'img37.jpg')
    # img_origin = cv2.imread(origin_img_save_path + 'img82.jpg')
    img_origin = cv2.resize(img_origin, (window_width, window_height))
    # cv2.imshow('img_origin'+logo, img_origin)

    # 2.进行高斯模糊
    img_after_gaussian = dd.GaussianFilter(img_origin)
    # cv2.imshow('img_after_gaussian'+logo, img_after_gaussian)

    # 3.获得黄色掩膜
    img_yellow_roi = bb.Get_Yellow_Line(img_after_gaussian)
    cv2.imshow('img_yellow_roi' + logo, img_yellow_roi)

    # 4.获得白色掩膜
    img_white_roi = bb.Get_White_Line(img_after_gaussian)
    # cv2.imshow('img_white_roi'+logo, img_white_roi)

    # 5.黄色和白色两幅图像相加
    img_overlay = bb.tow_img_overlay(img_white_roi, img_yellow_roi)
    # cv2.imshow('img_overlay'+logo, img_overlay)

    # 6.提取彩色 roi
    img_color_roi = cc.roi_trapezoid(img_overlay)
    # cv2.imshow('img_roi'+logo, img_color_roi)

    # 7.将图像进行闭运算
    img_close_operation = bb.img_close_operation(img_color_roi)
    # cv2.imshow('img_close_operation'+logo, img_close_operation)

    # 8.canny 边缘提取
    img_after_canny = ee.canny(img_close_operation)
    cv2.imshow('img_after_canny'+logo, img_after_canny)

    # 9.基于霍夫变换的直线检测
    lines = cv2.HoughLinesP(img_after_canny, 1, np.pi / 180, 20, np.array([]), minLineLength=30, maxLineGap=60)  # ta参
    print(lines)


    # 6.4章节代码需要，霍夫变换检测出多条直线，为了 做对比
    # line_image = np.zeros_like(img_after_gaussian)
    # color = (0, 0, 255)  # 红色，以 (B, G, R) 的顺序表示
    # for line in lines:      # 在图片上绘制直线
    #     x1, y1, x2, y2 = line[0]
    #     fit = np.polyfit((x1, x2), (y1, y2), 1)  # 拟合成直线
    #     slope = fit[0]  # 斜率
    #     print(slope)
    #     cv2.line(line_image, (x1, y1), (x2, y2), color, thickness=1)
    #     # cv2.circle(img_after_canny, (x1, y1), radius=2, color=color, thickness=1)
    #     # cv2.circle(img_after_canny, (x2, y2), radius=2, color=color, thickness=1)
    # cv2.imshow('line_image', line_image)


    #  10. 筛选直线 斜率在指定范围之内
    filtered_lines = ee.select_line(lines)
    print('筛选之后', filtered_lines)

    # 判断实线还是虚线  6.1 6.2章节代码需要
    # img_solid = np.copy(img_close_operation)    # 复制一个副本
    # ee.solid_dotted_judge(img_solid, filtered_lines)    # 执行判断
    # cv2.imshow('img_solid'+logo, img_solid)

    # 11.霍夫变换提取的直线绘制到空白图像上，即绘制车道线段
    line_image = np.zeros_like(img_color_roi)
    ee.draw_chedao_lines(line_image, filtered_lines, 4)
    cv2.imshow('line_image'+logo, line_image)

    # 12.图像融合
    img_mix = ee.weighted_img(line_image, img_origin)
    cv2.imshow('img_mix'+logo, img_mix)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_chedao_vedio():
    video_chedao_path = "D:\\AllCode\\python\\lane_dectection\\chedao\\vedio\\video_5s.mp4"
    img_chedao_save_path = "D:\AllCode\\python\\lane_dectection\\chedao\\origin\\"

    img_chedao_final_handle_path = "D:\\AllCode\\python\\lane_dectection\\chedao\\final_handle\\"
    vedio_chedao_final_handle_path = "D:\\AllCode\\python\\lane_dectection\\chedao\\vedio\\"
    video_time_len = aa.get_duration_from_cv2(video_chedao_path)
    print("视频时长:", video_time_len, "s\n")

    # 1.将视频保存成原始图片
    aa.Video_splitting(video_chedao_path, img_chedao_save_path)

    # 处理图片并保存
    for i in range(144):
        filename = img_chedao_save_path + 'img' + str(i) + '.jpg'
        img_origin = cv2.imread(filename)
        final_img = handle_chedao_img(img_origin)
        final_filename = img_chedao_final_handle_path + 'img_final' + str(i) + '.jpg'
        cv2.imwrite(final_filename, final_img)
        print(i)

    # 图片合成视频
    videoname = vedio_chedao_final_handle_path + "video_final.mp4"  # 要创建的视频文件名称
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器修改
    fps = 24  # 帧率(多少张图片为输出视频的一秒)
    size = (window_width, window_height)
    videoWrite = cv2.VideoWriter(videoname, fourcc, fps, size)    # 1.要创建的视频文件名称 2.编码器 3.帧率 4.size
    for i in range(144):    # 图片合成视频
        filename = img_chedao_final_handle_path + 'img_final' + str(i) + '.jpg'
        img = cv2.imread(filename)
        videoWrite.write(img)  # 写入
        print(i)


def handle_chedao_img(img_origin):
    img_origin = cv2.resize(img_origin, (window_width, window_height))

    # 2.进行高斯模糊
    img_after_gaussian = dd.GaussianFilter(img_origin)

    # 3.获得黄色掩膜
    img_yellow_roi = bb.Get_Yellow_Line(img_after_gaussian)

    # 4.获得白色掩膜
    img_white_roi = bb.Get_White_Line(img_after_gaussian)

    # 5.黄色和白色两幅图像相加
    img_overlay = bb.tow_img_overlay(img_white_roi, img_yellow_roi)

    # 6.提取彩色 roi
    img_color_roi = cc.roi_trapezoid(img_overlay)

    # 7.将图像进行闭运算
    img_close_operation = bb.img_close_operation(img_color_roi)

    # 8.canny 边缘提取
    img_after_canny = ee.canny(img_close_operation)

    # 9.基于霍夫变换的直线检测
    lines = cv2.HoughLinesP(img_after_canny, 1, np.pi / 180, 20, np.array([]), minLineLength=30, maxLineGap=60)  # ta参
    # print(lines)

    #  10. 筛选直线 斜率在指定范围之内
    filtered_lines = ee.select_line(lines)
    # print('筛选之后', filtered_lines)

    # 11.霍夫变换提取的直线绘制到空白图像上，即绘制车道线段
    line_image = np.zeros_like(img_color_roi)
    ee.draw_chedao_lines(line_image, filtered_lines, 4)

    # 12.图像融合
    img_mix = ee.weighted_img(line_image, img_origin)

    return img_mix


def main():
    # video_time_len = aa.get_duration_from_cv2(video_path)


    # video_time_len = aa.get_duration_from_cv2(video_luyan_path)
    # print("视频时长:", video_time_len, "s\n")

    # 1.将视频保存成原始图片
    # aa.Video_splitting(video_luyan_path, img_luyan_save_path)

    # 2.将原始图像灰度化
    # grayscale(144)

    # 3.将灰度化之后的图片进行高斯滤波
    # Gaussian_img_save(144)

    # 4.将高斯滤波之后的图片进行 canny算子边缘提取

    # test_y_w()
    # show_chedao_img()

    # show_luyan_img()

    show_chedao_vedio()

    # # 原始图片
    # img = cv2.imread(other_img_save_path + '22.jpg')
    # img = cv2.resize(img, (window_width, window_height))
    # cv2.imshow('img' + logo, img)
    #
    # # 获得白色掩膜
    # img_white_mask = bb.Get_White_Line(img)
    # cv2.imshow('img_white_mask' + logo, img_white_mask)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    print("over\n")

if __name__=="__main__":
    main()