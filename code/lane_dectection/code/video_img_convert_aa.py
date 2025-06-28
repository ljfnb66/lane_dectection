import cv2
import math
import main

""" 这一部分是视频转换成图像和图像合成视频的操作"""

# 获取视频时长
def get_duration_from_cv2(filename):
    cap = cv2.VideoCapture(filename)
    if cap.isOpened():
        rate = cap.get(5)
        frame_num = cap.get(7)
        duration = frame_num / rate
        return duration
    return -1


# 视频拆分
def Video_splitting(video_name, img_save_path):
    cap = cv2.VideoCapture(video_name)
    isOpened = cap.isOpened  # 判断视频是否可读
    print(isOpened)
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取图像的帧，即该视频每秒有多少张图片
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取图像的宽度和高度
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = 640
    height = 480
    print(fps, width, height)
    length = math.floor(get_duration_from_cv2(video_name))  # 向下取整

    i = 0
    while isOpened:
        if i == 24 * length:  # 分解为多少帧(i)
            break
        # 读取每一帧，flag表示是否读取成功，frame为图片的内容
        (flag, frame) = cap.read()
        filename = img_save_path + 'img' + str(i) + '.jpg'  # 文件的名字
        if flag:
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])  # 保存图片
        i += 1
    return length


# 视频合成
def Video_compositing(length, video_save_path, img_path):
    img = cv2.imread('img0.jpg')
    width = img.shape[0]
    height = img.shape[1]
    size = (height, width)
    print(size)

    videoname = "2.mp4"  # 要创建的视频文件名称
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') # 编码器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码器修改
    fps = 24  # 帧率(多少张图片为输出视频的一秒)

    # 1.要创建的视频文件名称 2.编码器 3.帧率 4.size
    videoWrite = cv2.VideoWriter(videoname, fourcc, fps, size)
    for i in range(fps * length):
        filename = 'img_line' + str(i) + '.jpg'
        img = cv2.imread(filename)
        videoWrite.write(img)  # 写入
