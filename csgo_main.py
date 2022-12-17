# THIS FILE IS PART OF Bo PROJECT
# csgo_main.py - The core part of the AI assistant
#
# THIS PROGRAM IS A FREE PROGRAM, WHICH IS LICENSED UNDER Bo
# DO NOT FORWARD THIS PROGRAM TO ANYONE

from csgo.screen_inf import grab_screen_mss, grab_screen_win32, get_parameters, grab_screen_pyqt5, get_window_rect
from csgo.cs_model import load_model
import cv2
import win32gui
import win32con
import torch
import numpy as np
from utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from utils.augmentations import letterbox

import time
import os
import pynput
from csgo.aim_lock import Locker
from threading import Thread
import argparse
import winsound

parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str,
                    # default=r'J:\Project\yolov5\mydatasets\csgo_for_training4\csgo_for_training4_120epoch_6905pic_best.pt',
                    # default=r'J:\Project\yolov5\mydatasets\csgo_for_training4\csgo_for_training4_120epoch_6905pic_best.engine',
                    # default='csgo/csgo.pt',
                    default='csgo/csgo.engine',
                    help='模型地址')
parser.add_argument('--use-cuda', type=bool, default=True, help='是否使用cuda')
parser.add_argument('--imgsz', type=int, default=640, help='和你训练模型时imgsz一样')
parser.add_argument('--conf-thres', type=float, default=0.50, help='置信阈值')  # yolov5 default 0.25; up主default 0.75
parser.add_argument('--iou-thres', type=float, default=0.45, help='交并比阈值')

# Bo: show_window貌似会降低30帧的速度。。
parser.add_argument('--show-window', type=bool, default=False, help='是否显示实时检测窗口(新版里改进了效率。若为True，不要去点右上角的X！)')
parser.add_argument('--top-most', type=bool, default=True, help='是否保持实时检测窗口置顶')
parser.add_argument('--resize-window', type=float, default=1/2, help='缩放实时检测窗口大小')
parser.add_argument('--thickness', type=int, default=3, help='画框粗细，必须大于1/resize-window')
parser.add_argument('--show-fps', type=bool, default=True, help='是否显示帧率')
parser.add_argument('--show-label', type=bool, default=True, help='是否显示标签')
parser.add_argument('--show-conf', type=bool, default=True, help='是否显示置信度')  # added by Bo
parser.add_argument('--show-debug-info', type=bool, default=False, help='是否显示Debug信息，比如检测时间、FPS、PID信息等')  # added by Bo

# TBD:pyqt5的方式，因为窗体的标题栏和边框问题，目前最终计算的像素位置，跟mss可能还是有些许偏差(详见下文中计算窗体及屏幕相关信息的部分)，mss跟模型实际的锁定位置完全一致。
# 截屏方式：'pyqt5','mss','win32'
parser.add_argument('--screenshot-method', type=str, default='mss', help='pyqt5, mss, win32')
# 使用pyqt5的好处是移动了窗口，检测区域会跟着动。而mss则固定检测屏幕中心区域(虽然也可以动态改变位置，但不推荐)
# 如果用mss截图，下面2个参数会起作用，1280*720 csgo窗口模式刚好为1/2(窗体有标题栏，可能略有出入)；如果用pyqt5截图是直接指定窗口，下面2个参数不起作用
# parser.add_argument('--region', type=tuple, default=(0.3, 0.3), help='检测范围；分别为横向和竖向，(1.0, 1.0)表示全屏检测，越低检测范围越小(始终保持屏幕中心为中心)')
# Bo: 因为改用TensorRT加速推理时，输入图像最好要是正方形！不然要修改letterbox参数，进而可能会影响推理速度，所以修改了上面up主原来的方式，直接用像素代替。
parser.add_argument('--region', type=tuple, default=(640, 640), help='检测范围；分别为横向和竖向，始终保持屏幕中心为中心，截图的宽和高')
parser.add_argument('--region-stay-center', type=bool, default=True, help='为False则region参数不起作用，会跟pyqt5一样，直接去寻找窗口的位置')  # added by Bo
# region_stay_center= False的话，效果并不好，因为需要在while循环里一直GetWindowRect，不建议使用！


###########################################################################################
# 原up主下面这两个参数作用实际一样的，作用重复。
parser.add_argument('--lock-sen', type=float, default=1.4, help='lock幅度系数；若在桌面试用请调成1，在游戏中(csgo)则为灵敏度')
parser.add_argument('--lock-smooth', type=float, default=3, help='lock平滑系数；越大越平滑，最低1.0')  # up之前默认是3(yolov5-6.1重构版本)
parser.add_argument('--lock-smooth-bo', type=float, default=1, help='lock平滑系数；越大越平滑，最低1.0')  # Bo 计算FOV后的调整参数
###########################################################################################


parser.add_argument('--lock-button', type=str, default='x2', help='lock按键；只支持鼠标按键')  # x2是鼠标侧键
parser.add_argument('--hold-lock', type=bool, default=False, help='lock模式；True为按住，False为切换')
parser.add_argument('--lock-sound', type=bool, default=True, help='切换到lock模式时是否发出提示音')
parser.add_argument('--head-first', type=bool, default=True, help='是否优先瞄头')
parser.add_argument('--lock-tag', type=list, default=[0, 1, 2, 3], help='为了防止用别人的pt，类别顺序可能不一样，0:t_head, 1:t_body, 2:ct_head, 3:ct_body')
parser.add_argument('--lock-choice', type=list, default=[0, 1, 2, 3], help='目标选择；可自行决定锁定的目标，从自己的标签中选')
###########################################################################################
# PID yyds！在CSGO bot地图，bot移动速度调max即5档，开启PID后相比于不开pid，命中率大幅提高！！！哈哈哈哈
parser.add_argument('--lock-strategy', type=str, default='pid', help='lock模式移动改善策略，为空时无策略，为pid时使用PID控制算法，暂未实现其他算法捏')
# 默认(1.1, 0.1, 0.1)效果很好！！
# (1.1, 0.2, 0.1)效果更好？！但是很抖。。 似乎检测帧数越高，I越大越好。使用TensorRT的.engine文件时，并且关掉show_window用这个效果更好！D用0.2也不错
# (0.5, 0.2, 0.1)貌似P值小些就没那么晃了，而且移动靶特别准！但是貌似P低于0.3后，也更晃。。。
# (1.1, 0.2, 0.2)效果也不错！但是D是0.2了也很晃。。
# 单变量P：2晃飞！1.3也开始晃，0.5就跟不上了有静态误差。1.1差不多刚刚追上
# 单变量I(给定P=1.1)：貌似0.1就能追踪上目标了
# 单变量D(给定P=1.1)：貌似0.3以上就开始震荡了！
# 0.8； 0.1； 0.3打远处的目标貌似相当不错
# 0.8； 0.1； 0.1貌似比上面更稳也更吊?
# 0.9 0.1 0也不错
# PID yyds!!貌似考虑采样时间delta_t后，系统更加稳定了！！！哈哈哈
# 如果采用PID算法2，即考虑采样时间，经测试，平均loop时间为0.016810093150538507秒，平均FPS(即倒数):59.5。则原PID参数的I应该放大60倍，D要缩小60倍。
parser.add_argument('--p-i-d', type=tuple, default=(0.8, 0.1*60, 0.1/60), help='PID控制算法p,i,d参数调整')
# parser.add_argument('--p-i-d', type=tuple, default=(0.8, 0.1, 0.1), help='PID控制算法p,i,d参数调整')
parser.add_argument('--anti-flag', type=bool, default=False, help='PID抗积分饱和, True则error_sum_x会清0')  # True震荡小了，但P和I貌似要增加
parser.add_argument('--auto-shoot', type=bool, default=False, help='是否开启自动射击模式')  # Bo: 自动开枪模式
parser.add_argument('--auto-switch', type=bool, default=False, help='是否开启自动切枪')  # Bo: 自动开枪模式时,awp射击后自动切枪
###########################################################################################
parser.add_argument('--recoil-sen', type=float, default=1, help='压枪幅度；自己调，调到合适')
parser.add_argument('--recoil-button', type=str, default='x1', help='ak47压枪按键；只支持鼠标按键,用不到置为0')


args = parser.parse_args()

cur_dir = os.path.dirname(os.path.abspath(__file__)) + '\\'

args.lock_tag = [str(i) for i in args.lock_tag]  # 转str方便后面lock函数进行str判断
args.lock_choice = [str(i) for i in args.lock_choice]  # 转str方便后面lock函数进行str判断

device = 'cuda' if args.use_cuda else 'cpu'
imgsz = args.imgsz
conf_thres = args.conf_thres
iou_thres = args.iou_thres


# 初始化Locker类
locker = Locker(args)

# 窗体及屏幕相关信息(放在while循环之前增加FPS)
if args.screenshot_method == 'pyqt5':
    hwnd = win32gui.FindWindow(None, 'Counter-Strike: Global Offensive - Direct3D 9')  # CSGO窗口模式
    x1, y1, x2, y2 = win32gui.GetWindowRect(hwnd)  # 返回的是窗口左上角与右下角坐标：x1,y1,x2,y2: 637 342 1923 1097  (csgo 1280*720分辨率下)
    # x1, y1, x2, y2 = get_window_rect(hwnd)  # 貌似这个网上的自定义方法跟上面官方的效果差不多了，官方的估计已经修复
    # len_x, len_y = 1280, 720  # hard code
    locker.top_x, locker.top_y = x1, y1
    locker.len_x, locker.len_y = x2 - x1, y2 - y1
    # print(locker.top_x, locker.top_y, locker.len_x, locker.len_y)  # bo debug
    # 注意：csgo 设置1280*720分辨率下，返回值包含了边框部分特别是标题栏的像素！！即print(top_x, top_y, len_x, len_y)： 637 342 1286 755
    # 但后面pyqt5实际截图时，图片大小确实只有1280*720，对比mss的值可以发现，窗体的左右边框厚度，应该是各占640-637=3个像素，水平方向刚好(1923-3)-(637+3)=1280像素
    # 同理猜测窗体的下边框也是3个像素(但似乎是5个像素才对的上，不太理解)。窗体的上边框即标题栏，经实际测量应该是30像素，所以截图是从y=372开始，垂直方向(1097-5)-372=720像素
    # 似乎也可能上边框是32个像素，下边框为3个像素，哎，无所谓了！
    # 所以，重新调整截图区域的真实位置！！
    locker.top_x = locker.top_x + 3
    locker.top_y = locker.top_y + 30
    locker.len_x = x2 - 3 - locker.top_x  # 1280
    locker.len_y = y2 - 5 - locker.top_y  # 720
    # TBD： pyqt5似乎仍然还是有些偏差！！
else:  # screenshot_method == 'mss'
    top_x, top_y, x, y = get_parameters()
    if args.region_stay_center:  # 始终保持屏幕中心为中心
        # len_x, len_y = int(x * args.region[0]), int(y * args.region[1])
        # top_x, top_y = int(top_x + x // 2 * (1. - args.region[0])), int(top_y + y // 2 * (1. - args.region[1]))
        # 注意：返回值区别于pyqt5！1280*720分辨率下，print(top_x, top_y, len_x, len_y)： 640 360 1280 720
        # Bo: 注意使用tensorRT生成的engine文件加速推理时，要将输入图片改成正方形，不然后面的letterbox函数会报错，得修改auto参数为False，但这样又会影响推理速度，故截图取正方形
        locker.len_x, locker.len_y = args.region[0], args.region[1]
        locker.top_x, locker.top_y = int(x // 2 - args.region[0] / 2), int(y // 2 - args.region[1] / 2)
    else:  # 不要求始终保持屏幕中心为中心，则跟pyqt5一样去寻找位置。但这样效果并不好，不推荐！
        hwnd = win32gui.FindWindow(None, 'Counter-Strike: Global Offensive - Direct3D 9')  # CSGO窗口模式
        x1, y1, x2, y2 = win32gui.GetWindowRect(hwnd)  # 返回的是窗口左上角与右下角坐标：x1,y1,x2,y2
        locker.top_x, locker.top_y = x1, y1
        locker.len_x, locker.len_y = x2 - x1, y2 - y1
    monitor = {'left': locker.top_x, 'top': locker.top_y, 'width': locker.len_x, 'height': locker.len_y}

model, stride, names = load_model(args)  # names返回的是pt文件中包含的classes的字典信息,如果是engine文件貌似就没有类别名称，只有序号了！！

if args.show_window:
    cv2.namedWindow('csgo-detect', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('csgo-detect', int(locker.len_x * args.resize_window), int(locker.len_y * args.resize_window))


# B站up主的视频yolov5 csgo压枪代码(yolov5-6.1重构)
# *******************************************************************************
# 启动一个线程，并将封装有事件监听的方法名作为target参数传入
# t = Thread(target=recoil_control, kwargs={'args': args})
# t.start()  # 非阻塞，程序会继续往下运行
# *******************************************************************************

# mouse = pynput.mouse.Controller()  # 同时创建Controller和下面的Listener的非阻塞版本的话，貌似B站视频up主说有时候会出问题！！
# TBD!!!!!

# 鼠标侧键x2的事件监听
def on_click(x, y, button, pressed):
    if button == eval('pynput.mouse.Button.' + args.lock_button):  # x2侧键，事件触发模式切换
        if args.hold_lock:
            if pressed:
                locker.lock_mode = True
                print('locking...')
                if args.lock_sound:
                    winsound.Beep(1000, 300)
            else:
                locker.lock_mode = False
                print('lock mode off')
                if args.lock_sound:
                    winsound.Beep(500, 300)
        else:
            if pressed:
                locker.lock_mode = not locker.lock_mode
                print('lock mode', 'on' if locker.lock_mode else 'off')
                if args.lock_sound:
                    winsound.Beep(1000 if locker.lock_mode else 500, 300)
                if not locker.lock_mode:  # 关闭锁定时，PID清0，关闭auto_shoot和auto_switch. Added by Bo.
                    locker.reset_pid_error()
                    locker.auto_shoot = False
                    locker.auto_switch = False
                    print('auto shoot mode off', 'auto switch mode off')

    # 切枪测试方法一(不推荐)：在开启自动开枪，且开枪awp自动切枪时，鼠标左键(松开时)的事件监听：
    # 注意：切枪操作放在on_click这里的话，切枪里面的time.sleep()操作，貌似会阻塞主线程即main loop，造成卡顿的情况，不推荐这么写。
    # *******************************************************************************
    # if locker.auto_shoot and locker.auto_switch:
    #     if button == pynput.mouse.Button.left:  # awp左键开枪
    #         # 松开左键时，Released的时候，否则会卡。。建议更改后面的time.sleep，变成根据时间判断。。。
    #         if not pressed:
    #             print('awp auto switching')
    #             locker.awp_auto_switch_after_shoot()
    # *******************************************************************************

    # else:  # 如果是其他按键，比如按了右键
    #     # Stop listener
    #     return False  # 调用这个语句，会停止事件监听，这个方法就结束失效了，即使是在这个非阻塞版中(阻塞版也会失效)


listener = pynput.mouse.Listener(on_click=on_click)
listener.start()  # 非阻塞版本，启动一个线程来监听


# 切枪测试方法二(推荐)：在开枪自动开枪时，且开枪awp自动切枪时，鼠标左键(松开时)的事件监听
# 注意：切枪操作的方法中因为存在time.sleep操作，为了避免阻塞主程序，通过python多线程Thread的方式启动监听。
# *******************************************************************************
# 启动一个线程，并将封装有事件监听的方法名作为target参数传入，其中args参数须为元组形式。
# 注意，线程的target指定的方法：awp_auto_switch_after_shoot必须定义为静态方法！！否则，用小写的locker(实例化对象)的实例方法，下面的语句不起作用！！
t = Thread(target=Locker.awp_auto_switch_after_shoot, args=(locker,))
t.start()  # 非阻塞，程序会继续往下运行
# *******************************************************************************


# 键盘监听事件:
def on_press(key):
    if key == pynput.keyboard.Key.shift:  # Shift键：警匪锁定目标切换，调整lock_choice锁人
        if locker.lock_t_or_ct == 0:  # if lock all, then to lock t
            locker.lock_t_or_ct = 1
            print('lock t only.')
            locker.lock_choice = ['0', '1']  # 注意是str类型
            if args.lock_sound:
                winsound.Beep(300, 300)
        elif locker.lock_t_or_ct == 1:  # if lock t, then to lock ct
            locker.lock_t_or_ct = 2
            print('lock ct only.')
            locker.lock_choice = ['2', '3']  # 注意是str类型
            if args.lock_sound:
                winsound.Beep(600, 300)
        else:  # if lock ct, then to lock all
            locker.lock_t_or_ct = 0
            print('lock all.')
            locker.lock_choice = ['0', '1', '2', '3']  # 注意是str类型
            if args.lock_sound:
                winsound.Beep(1000, 300)

    if hasattr(key, 'vk') and key.vk == 101:  # 小键盘5：切换自动开枪模式。 注意！并不是所有的key都有vk属性，其他按键key.vk会报错
        locker.auto_shoot = not locker.auto_shoot
        print('auto shoot mode', 'on' if locker.auto_shoot else 'off')
        if args.lock_sound:
            winsound.Beep(1000 if locker.auto_shoot else 500, 300)

    # # 小键盘数字5的另外一种写法。注意！并不是所有的key都有vk属性，其他按键key.vk会报错
    # try:
    #     if key.vk == 101:  # 小键盘的5
    #         locker.auto_shoot = not locker.auto_shoot
    #         print('auto shoot mode', 'on' if locker.auto_shoot else 'off')
    #         if args.lock_sound:
    #             winsound.Beep(1000 if locker.auto_shoot else 500, 300)
    # except AttributeError:
    #     pass

    if hasattr(key, 'vk') and key.vk == 98:  # 小键盘2：在自动开枪模式时，awp射击后是否自动切枪
        if locker.auto_shoot:  # 仅在自动开枪模式时
            locker.auto_switch = not locker.auto_switch
            print('auto switch mode', 'on' if locker.auto_switch else 'off')
            if args.lock_sound:
                winsound.Beep(1000 if locker.auto_switch else 500, 300)


def on_release(key):
    pass
    # if key == pynput.keyboard.Key.esc:
    #     # Stop listener
    #     return False
# ...or, in a non-blocking fashion:


listener2 = pynput.keyboard.Listener(
    on_press=on_press,
    on_release=on_release)
listener2.start()


print('enjoy yourself!')
t0 = time.time()
cnt = 0
sum_time1 = 0  # added by Bo for debug，仅截图的累计时间
sum_time2 = 0  # added by Bo for debug，仅推理的累计时间
sum_time3 = 0  # added by Bo for debug，loop(PID算法采样时间)的累计时间
while True:
    # Bo:没看懂原作者这里是要干嘛？？可以是为了防止屏幕分辨率等参数发生变化？
    # if cnt % 20 == 0:
    #     top_x, top_y, x, y = get_parameters()
    #     len_x, len_y = int(x * region[0]), int(y * region[1])
    #     top_x, top_y = int(top_x + x // 2 * (1. - region[0])), int(top_y + y // 2 * (1. - region[1]))
    #     monitor = {'left': top_x, 'top': top_y, 'width': len_x, 'height': len_y}
    #     cnt = 0

    # Bo:截屏
    if args.screenshot_method == 'pyqt5':
        # pyqt5检测截图直接传窗口对象进去，不用传位置参数！
        hwnd = win32gui.FindWindow(None, 'Counter-Strike: Global Offensive - Direct3D 9')  # CSGO窗口模式
        img0 = grab_screen_pyqt5(hwnd)
    elif args.screenshot_method == 'mss':
        # 注意，下面的grab_screen_mss返回值是(height, width, channel)
        # cv2里面的cv2.imread和cv2.resize的返回值，也一样，都是(H,W,C)
        img0 = grab_screen_mss(monitor)
        img0 = cv2.resize(img0, (locker.len_x, locker.len_y))  # 注意第2个参数是目标size，是元组形式(w,h)，顺序与输入(HWC)相反。
        # 上面这句话似乎没用
    else:
        img0 = grab_screen_win32(region=(locker.top_x, locker.top_y, locker.top_x + locker.len_x, locker.top_y + locker.len_y))
        img0 = cv2.resize(img0, (locker.len_x, locker.len_y))

    # added by Bo for debug
    if args.show_debug_info:
        sum_time1 += (time.time() - t0)
        print('截图 took {} seconds'.format(time.time() - t0), ', 平均截图时间为', sum_time1 / (cnt + 1))
        t1 = time.time()

    img = letterbox(img0, imgsz, stride=stride, auto=True)[0]
    # 注意：推理时如果用tensorrt加速，即.engine文件去跑，如果截图的输入不是640*640的矩形，则会报下面的错误，得修改上面的auto参数为False!但又会影响推理速度，所以修改截图为正方形
    # AssertionError: input size torch.Size([1, 3, 384, 640]) not equal to max model size (1, 3, 640, 640)
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    im = torch.from_numpy(img).to(device)
    im = im.float()
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred = model(im, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres, iou_thres)

    # added by Bo for debug
    if args.show_debug_info:
        sum_time2 += (time.time() - t1)
        print('检测 took {} seconds'.format(time.time() - t1), ', 平均检测时间为', sum_time2/(cnt+1))

    aims = []
    for i, det in enumerate(pred):  # Bo:这个循环貌似有点冗余，up主新版代码这里已经去掉了
        gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
        if len(det):
            # Rescale boxes from img_size to im0 size
            # Bo: 貌似up源码是6.1版本，而Bo下的是6.2版本，下面的方法从scale_coords改成了scale_boxes
            # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                # bbox:(tag, x_center, y_center, x_width, y_width)
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                # line = (cls, *xywh)  # label format
                line = (cls, *xywh, conf) if args.show_conf else (cls, *xywh)  # label format
                aim = ('%g ' * len(line)).rstrip() % line
                aim = aim.split(' ')
                # print(aim)  # added by Bo
                aims.append(aim)

            # print(f'检测到的个数：{len(aims)}')  # added by Bo

        if len(aims):
            if locker.lock_mode:
                # Bo: 注意，用pyqt5截图的话，移动窗体后，top_x和top_y位置变了，自瞄可能出问题！
                locker.lock(aims, (time.time()-t0))  # PID算法考虑采样时间delta_t

            if args.show_window:
                for i, det in enumerate(aims):
                    # added by Bo. unpack conf
                    if args.show_conf:
                        tag, x_center, y_center, width, height, conf = det
                    else:
                        tag, x_center, y_center, width, height = det
                    x_center, width = locker.len_x * float(x_center), locker.len_x * float(width)
                    y_center, height = locker.len_y * float(y_center), locker.len_y * float(height)
                    top_left = (int(x_center - width / 2.), int(y_center - height / 2.))
                    bottom_right = (int(x_center + width / 2.), int(y_center + height / 2.))
                    # cv2.rectangle(img0, top_left, bottom_right, (0, 255, 0), thickness=thickness)
                    # Bo: 警匪头身区分颜色标, BGR格式：https://blog.csdn.net/qq_51985653/article/details/113392665
                    if int(tag) == 0:  # t head: 铜黄
                        color = (0, 215, 255)
                    elif int(tag) == 1:  # t body: 黄色
                        color = (18, 153, 255)
                    elif int(tag) == 2:  # ct head:
                        color = (235, 206, 135)
                    else:  # ct body:
                        # color = (255, 255, 0)
                        color = (225, 105, 65)
                        # color = (230, 224, 176)
                        # color = (208, 224, 64)
                    cv2.rectangle(img0, top_left, bottom_right, color, thickness=args.thickness)

                    if args.show_label:
                        # cv2.putText(img0, tag, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 0, 0), 4)
                        # Bo: 将类别数字修改为名称,label用白色字体，用个小矩形框包住，参考了yolov5预测图片保存的代码
                        p1 = top_left
                        label = f'{names[int(tag)]} {float(conf):.2f}' if args.show_conf else names[int(tag)]
                        w, h = cv2.getTextSize(label, 0, fontScale=1.25, thickness=args.thickness)[0]  # text width, height
                        outside = p1[1] - h >= 3
                        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                        cv2.rectangle(img0, p1, p2, color, -1, cv2.LINE_AA)  # filled
                        cv2.putText(img0, label, top_left, cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), args.thickness,
                                    lineType=cv2.LINE_AA)
        else:  # len(aims) == 0，即没有目标时. Added by Bo.
            if locker.lock_mode:
                if locker.lock_strategy == 'pid':  # PID清0
                    locker.reset_pid_error()
                if locker.auto_shoot:  # 自动开枪逻辑:3秒内没有锁定目标则鼠标移动
                    if time.time() - locker.last_locked_time > 3:
                        locker.auto_turn_around()

    if args.show_window:
        if args.show_fps:
            cv2.putText(img0, "FPS:{:.1f}".format(1. / (time.time() - t0)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2,
                        (0, 0, 235), 4)
        cv2.imshow('csgo-detect', img0)
        if args.top_most:
            hwnd = win32gui.FindWindow(None, 'csgo-detect')
            CVRECT = cv2.getWindowImageRect('csgo-detect')
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        cv2.waitKey(1)

    # added by Bo for debug
    sum_time3 += (time.time() - t0)
    # if args.show_debug_info:
    if True:
        if cnt % 300 == 0:  # 每秒大概60帧，即每5秒打印一次
            print('loop took {} seconds'.format(time.time() - t0),
                  "FPS:{:.1f}".format(1. / (time.time() - t0)))
            print('平均loop时间为', sum_time3 / (cnt + 1),
                  "平均FPS:{:.1f}".format(1. / (sum_time3 / (cnt + 1))))

    t0 = time.time()

    # time.sleep(0.3)  # 加延时检查算法移动效果，要移动几次
    # time.sleep(5 / 1000)  # up主github最新代码加了这个：检测帧率控制(ms)，防止因快速拉枪导致的残影误检
    cnt += 1
    # time.sleep(0.01)  # 加10ms延迟，貌似帧数低一些，比如每秒30帧，似乎更加稳定不抖动？
