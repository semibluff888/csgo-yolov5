"""
csgo自动截图程序
"""
from screen_inf import grab_screen_mss, grab_screen_win32, get_parameters, grab_screen_pyqt5, get_window_rect
import cv2
import win32gui
import win32con
import time
import os
from PyQt5.QtWidgets import QApplication
import sys
import mss
import numpy as np
from PIL import Image


output_dir = r'J:\Project\yolov5\mydatasets\Bo自己截图的csgo_未标注\images'  # 图片保存目录
time_interval = 5  # 每隔几秒截图一次

screenshot_method = 'pyqt5'  # 'pyqt5','mss','wind32'
# 使用pyqt5的好处是移动了窗口，检测区域会跟着动。而mss则固定检测屏幕中心区域(虽然也可以动态改变位置，但不推荐)
# 如果用mss截图，下面2个参数会起作用，csgo窗口模式刚好为1/2；如果用pyqt5截图是直接指定窗口，下面2个参数不起作用
region = (0.5, 0.5)  # '检测范围；分别为横向和竖向，(1.0, 1.0)表示全屏检测，越低检测范围越小(始终保持屏幕中心为中心)')
region_stay_center = True  # added by Bo: 如果为False则region参数不起作用，会跟pyqt5一样，直接去寻找窗口的位置
# region_stay_center= False的话，效果并不好，因为需要在while循环里一直GetWindowRect，不建议使用！


# 放在while循环之前增加FPS
hwnd = win32gui.FindWindow(None, 'Counter-Strike: Global Offensive - Direct3D 9')  # CSGO窗口模式
app = QApplication(sys.argv)
screen = QApplication.primaryScreen()

top_x, top_y, x, y = get_parameters()
if region_stay_center:  # 始终保持屏幕中心为中心
    len_x, len_y = int(x * region[0]), int(y * region[1])
    top_x, top_y = int(top_x + x // 2 * (1. - region[0])), int(top_y + y // 2 * (1. - region[1]))
else:  # 不要求始终保持屏幕中心为中心，则跟pyqt5一样去寻找位置。但这样效果并不好，不推荐！
    x1, y1, x2, y2 = win32gui.GetWindowRect(hwnd)  # 返回的是窗口左上角与右下角坐标：x1,y1,x2,y2
    top_x, top_y = x1, y1
    len_x, len_y = x2 - x1, y2 - y1
monitor = {'left': top_x, 'top': top_y, 'width': len_x, 'height': len_y}


while True:
    # Bo:截屏
    if screenshot_method == 'pyqt5':
        # pyqt5检测截图直接传窗口对象进去，不用传位置参数！
        hwnd = win32gui.FindWindow(None, 'Counter-Strike: Global Offensive - Direct3D 9')  # CSGO窗口模式
        img0 = screen.grabWindow(hwnd).toImage()
        img0.save(os.path.join(output_dir, f'{time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())}.jpg'))
    elif screenshot_method == 'mss':
        with mss.mss() as m:
            img0 = m.grab(monitor)
            pim = Image.new("RGB", img0.size)
            pim.frombytes(img0.rgb)
            pim.save(os.path.join(output_dir, f'{time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())}.jpg'),
                     quality=95)
            # 95的满级quality(品质)可以保持原来的高清晰度
            del img0, pim

    else:
        pass

    # 每几秒截一张
    time.sleep(time_interval)




