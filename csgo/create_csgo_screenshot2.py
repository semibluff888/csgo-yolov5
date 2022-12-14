# ————————————————
# 用mss截图，
# 监听鼠标侧键点击事件截图：
# pynput官网说明文档如下：
# https://pypi.org/project/pynput/

import time
from pynput import mouse
import mss
import mss.tools
import os
from screen_inf import get_parameters

# lock_mode = False
output_dir = r'J:\Project\yolov5\mydatasets\Bo自己截图的csgo_未标注\images'  # 图片保存目录
region = (0.5, 0.5)  # '检测范围；分别为横向和竖向，(1.0, 1.0)表示全屏检测，越低检测范围越小(始终保持屏幕中心为中心)')

top_x, top_y, x, y = get_parameters()
len_x, len_y = int(x * region[0]), int(y * region[1])
top_x, top_y = int(top_x + x // 2 * (1. - region[0])), int(top_y + y // 2 * (1. - region[1]))
monitor = {'left': top_x, 'top': top_y, 'width': len_x, 'height': len_y}


def on_move(x, y):
    pass


def on_click(x, y, button, pressed):
    # print('{0} at {1}'.format(
    #     'Pressed' if pressed else 'Released',
    #     (x, y)))

    # global lock_mode

    if pressed and button == mouse.Button.x2:  # x2是鼠标左侧前面那个按键
        file_name = os.path.join(output_dir,
                                 f'{time.strftime("%Y-%m-%d %H_%M_%S", time.localtime())}.png')
        # mss截图，并保存为png格式！
        with mss.mss() as m:
            img = m.grab(monitor)
            mss.tools.to_png(img.rgb, img.size, 6, file_name)
        print(file_name)

    # 下面是释放按键的时候返回false，程序会退出，监听结束。。根据需要！
    # if not pressed:
    #     # Stop listener
    #     return False


def on_scroll(x, y, dx, dy):
    pass


# 阻塞版：即程序会一直卡在这里运行。
# Collect events until released
with mouse.Listener(
        on_move=on_move,
        on_click=on_click,
        on_scroll=on_scroll) as listener:
    listener.join()  # Bo: join即是多线程中阻塞的概念，即会等待直到完成。

# 多线程的相关概念可以参考这个链接
# https://blog.csdn.net/piglite/article/details/79427639

# 非阻塞版：即程序会接着往后面运行，继续运行后面的代码。貌似是多线程的。可以在后面加个while True查看效果
# # ...or, in a non-blocking fashion:
# listener = mouse.Listener(
#     on_move=on_move,
#     on_click=on_click,
#     on_scroll=on_scroll)
# listener.start()


# # added by bo，可以添加下面的语句来查看non-blocking fashion的效果
# while True:
#     # Set pointer position
#     if lock_mode:
#         mouse.Controller.position = (2560/2, 1440/2)
#     pass

