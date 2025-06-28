from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


def delta_e_cie2000_distance(color1, color2):
    """
    使用CIEDE2000色差公式计算两个颜色之间的距离。

    参数:
    color1, color2 -- 两个颜色的RGB值，每个值的范围是0到255。
    """
    # 将RGB颜色转换为CIELAB颜色
    lab1 = convert_color(sRGBColor(*color1), LabColor)
    lab2 = convert_color(sRGBColor(*color2), LabColor)

    # 计算CIEDE2000色差
    return delta_e_cie2000(lab1, lab2)


# 示例：计算两个颜色之间的CIEDE2000色差
color1 = (255, 0, 0)  # 红色
color2 = (0, 255, 0)  # 绿色
print("CIEDE2000色差:", delta_e_cie2000_distance(color1, color2))