
import csgo.ghub_mouse as ghub
from math import *
import pynput
import csv
import time

mouse = pynput.mouse.Controller()


class Locker(object):
    def __init__(self, args):
        self.top_x = 0
        self.top_y = 0
        self.len_x = 0
        self.len_y = 0

        self.flag = 0  # up主的原压枪标识
        self.lock_mode = False
        self.ct_mode = False  # added by Bo
        self.show_conf = args.show_conf  # added by Bo

        self.head_first = args.head_first
        self.lock_tag = args.lock_tag
        self.lock_choice = args.lock_choice
        self.lock_sen = args.lock_sen
        self.lock_smooth = args.lock_smooth
        self.lock_smooth_bo = args.lock_smooth_bo  # added by Bo

        # PID相关
        self.kp, self.ki, self.kd = args.p_i_d
        self.lock_strategy = args.lock_strategy
        self.anti_i_flag = args.anti_flag  # added by Bo:抗积分饱和
        self.error_sum_x = 0
        self.error_sum_y = 0
        self.pre_error_x = 0
        self.pre_error_y = 0
        self.shot_time = 0  # 自动开枪逻辑: 记录射击时间，保证两枪的间隔时间
        self.last_locked_time = 0  # 自动开枪逻辑：记录最后一次存在锁定目标的时间，如果长时间没有锁定目标，则鼠标移动

# *****************************************************************************
# PID算法相关：参考B站up主Caesar的PID方法实现，但疑似error_sum项不清0的话，积分项容易累积？
# https://github.com/Caesar-s1mple/csgo-yolov5-6.2/blob/main/aim_csgo/aim_lock_pi.py
    def reset_pid_error(self):
        """
        使用PID算法没有锁定目标时，清0
        """
        self.error_sum_x = 0
        self.error_sum_y = 0
        self.pre_error_x = 0
        self.pre_error_y = 0

    def pid(self, error_x, error_y):
        # 离散形式PID
        Pout_x = self.kp * error_x
        self.error_sum_x += error_x
        Iout_x = self.ki * self.error_sum_x
        Dout_x = self.kd * (error_x - self.pre_error_x)
        self.pre_error_x = error_x

        Pout_y = self.kp * error_y
        self.error_sum_y += error_y
        Iout_y = self.ki * self.error_sum_y
        Dout_y = self.kd * (error_y - self.pre_error_y)
        self.pre_error_y = error_y

        # print("error_sum_x: ", self.error_sum_x, " error_sum_y: ", self.error_sum_x,
        #       " pre_error_x: ", self.pre_error_x, " pre_error_y: ", self.pre_error_y)

        # return int(Pout_x + Iout_x + Dout_x), error_y  # 貌似y轴方向不用PID效果也不错！
        return int(Pout_x + Iout_x + Dout_x), int(Pout_y + Iout_y + Dout_y)
    # *****************************************************************************

    def pid2(self, error_x, error_y, delta_t):  # 考虑采样时间delta_t, 抗积分饱和
        """
        delta_t:考虑采样时间的原因可以参考微信收藏：防止单次循环时间不稳定对PID移动量的影响。
        anti_i_flag:借用抗积分饱和这个类似的概念。这里的实际作用是，如果error跟上一次的error符号相反，说明已经追上目标了，
        但是积分项的累计作用可能仍有“惯性”，会继续往反方向作用，导致震荡。设置为True则之前的error_sum_x会清0。
        """
        # 离散形式PID
        Pout_x = self.kp * error_x
        if self.anti_i_flag and error_x * self.pre_error_x <= 0:  # “抗饱和”
            self.error_sum_x = error_x * delta_t  # 之前的清0，这次重新开始计算
        else:
            self.error_sum_x += (error_x * delta_t)
        Iout_x = self.ki * self.error_sum_x
        Dout_x = self.kd * (error_x - self.pre_error_x) / delta_t
        self.pre_error_x = error_x

        Pout_y = self.kp * error_y
        if self.anti_i_flag and error_y * self.pre_error_y <= 0:  # “抗饱和”
            self.error_sum_y = error_y * delta_t
        else:
            self.error_sum_y += (error_y * delta_t)
        Iout_y = self.ki * self.error_sum_y
        Dout_y = self.kd * (error_y - self.pre_error_y) / delta_t
        self.pre_error_y = error_y

        # print("error_sum_x: ", self.error_sum_x, " error_sum_y: ", self.error_sum_x,
        #       " pre_error_x: ", self.pre_error_x, " pre_error_y: ", self.pre_error_y)

        return int(Pout_x + Iout_x + Dout_x), int(Pout_y + Iout_y + Dout_y)
    # *****************************************************************************

    # def lock(aims, mouse, top_x, top_y, len_x, len_y, lock_choice, head_first):
    def lock(self, aims, delta_t):
        mouse_pos_x, mouse_pos_y = mouse.position
        # print(mouse_pos_x, mouse_pos_y)  # csgo x=1280, y=734
        # 注意截图方式为pyqt5的传入的top_x, top_y的高度位置会有偏差！！！因为窗体有标题栏，占一定高度，所以鼠标的位置并非是2560/2, 1440/2，要往下一点！！
        # 下面两个计算方式返回的位置都是1280，720，跟实际位置x=1280, y=734 ；不一致！！
        # mouse_pos_x, mouse_pos_y = 2560/2, 1440/2  # 某些FPS游戏可能鼠标并非屏幕中间？
        # mouse_pos_x, mouse_pos_y = top_x + len_x // 2, top_y + len_y // 2
        aims_copy = aims.copy()
        # 筛选要锁定的对象lock_choice；CT锁T，T锁CT
        aims_copy = [x for x in aims_copy if x[0] in self.lock_choice]  # 都是str类型
        k = 4.07 * (1 / self.lock_smooth)
        # TBD: 开启瞄准后出现剧烈抖动的原因：对于远处的目标，模型识别不稳定，特别是当准星移动过去的时候，挡住了头部，导致警匪混乱，
        # 以及鼠标移动过去后，挡住的头部不再能识别出来, head first情况下，又切换到另一个头部目标了，然后循环反复。。。
        if len(aims_copy):
            self.last_locked_time = time.time()  # bo: 自动开枪逻辑。记录最后一次存在锁定目标的时间
            dist_list = []
            tag_list = [x[0] for x in aims_copy]
            if self.head_first:  # 0:t_head, 1:t_body, 2:ct_head, 3:ct_body
                if (self.lock_tag[0] in tag_list) or (self.lock_tag[2] in tag_list):  # 有头
                    aims_copy = [x for x in aims_copy if x[0] in [self.lock_tag[0], self.lock_tag[2]]]
            for det in aims_copy:
                # added by Bo. unpack conf
                if self.show_conf:
                    _, x_c, y_c, _, _, _ = det
                else:
                    _, x_c, y_c, _, _ = det
                dist = (self.len_x * float(x_c) + self.top_x - mouse_pos_x) ** 2 + (self.len_y * float(y_c) + self.top_y - mouse_pos_y) ** 2
                dist_list.append(dist)

            det = aims_copy[dist_list.index(min(dist_list))]  # 最近的目标
            # added by Bo. unpack conf
            if self.show_conf:
                tag, x_center, y_center, width, height, _ = det
            else:
                tag, x_center, y_center, width, height = det
            x_center, width = self.len_x * float(x_center) + self.top_x, self.len_x * float(width)
            y_center, height = self.len_y * float(y_center) + self.top_y, self.len_y * float(height)
            # *****************************************************************************
            # 算法1：up主的方法，默认参数下比较平滑，但要几帧才能拉到位置，目标高速移动情况下可能较难命中(需要改变参数最好达到1帧拉枪)
            # rel_x = int(k / self.lock_sen * atan((mouse_pos_x - x_center) / 640) * 640)
            # *****************************************************************************
            # 算法2: bo的方法 (详见下文)；目前的锁定效果接近于一帧拉枪。
            rel_x = int(1041.987412 * atan((mouse_pos_x - x_center) / 480))
            # *****************************************************************************
            # *****************************************************************************
            # 详见J:\Project\yolov5\QQ群里的代码及资料\C + +FOV.h里面的计算方式！
            # 水平方向x的距离计算
            # FOV = 106.260205  # 水平FOV，貌似分辨率1280*720跟1920*1080，是一样的！
            # width = 1920  # CSGO水平分辨率
            # pixel_x = 6547  # 水平移动360度，X轴移动的像素，貌似分辨率1280*720跟1920*1080，是一样的！
            # per_pixel_rad = pixel_x / (2 * pi)  # 值为1041.987412
            # delta_abs_x = abs(delta_x)
            # sup_distance = (width / 2) / tan((FOV * pi / 180) / 2)  # bo: 1920*1080分辨率下720；1280*720分辨率下480
            # target_angle_rad = atan(delta_abs_x / sup_distance)
            # target_move = target_angle_rad * per_pixel_rad
            # *****************************************************************************
            # 垂直方向y的距离计算
            # FOV = 73.739795  # 垂直FOV，貌似分辨率1280*720跟1920*1080，是一样的！
            # height = 1080  # CSGO垂直分辨率
            # pixel_y = 3228  # 垂直移动半圈，y轴移动的像素，所以下面不用除以2了!貌似分辨率1280*720跟1920*1080，是一样的！
            # per_pixel_rad = pixel_y / pi  # 值为1027.504313
            # delta_abs_y = abs(delta_y)
            # sup_distance = (height / 2) / tan((FOV * pi / 180) / 2)  # bo: 1920*1080分辨率下720；1280*720分辨率下480
            # target_angle_rad = atan(delta_abs_y / sup_distance)
            # target_move = target_angle_rad * per_pixel_rad
            # *****************************************************************************

            if tag in [self.lock_tag[0], self.lock_tag[2]]:  # head
                # *****************************************************************************
                # 算法1：up主的方法
                # rel_y = int(k / self.lock_sen * atan((mouse_pos_y - y_center) / 640) * 640)
                # *****************************************************************************
                # 算法2: bo的方法
                rel_y = int(1027.504313 * atan((mouse_pos_y - y_center) / 480))
                # *****************************************************************************
                if self.flag:  # 如果是压枪状态就暂时不锁定了，up主新版代码后来这里其实删掉了
                    return
            elif tag in [self.lock_tag[1], self.lock_tag[3]]:  # body
                # *****************************************************************************
                # 算法1：up主的方法
                # rel_y = int(k / self.lock_sen * atan((mouse_pos_y - y_center + 1 / 6 * height) / 640) * 640)
                # *****************************************************************************
                # 算法2: bo的方法
                rel_y = int(1027.504313 * atan((mouse_pos_y - y_center + 1 / 6 * height) / 480))
                # *****************************************************************************
                if self.flag:  # 如果是压枪状态就暂时不锁定了，up主新版代码后来这里其实删掉了
                    return

            # *****************************************************************************
            # 算法2:bo的方法: 平滑移动系数后(除了大概2~3)跟up主的效果基本类似！所以除了2.5的系数
            rel_x = int(rel_x / 2.5 / self.lock_smooth_bo)
            rel_y = int(rel_y / 2.5 / self.lock_smooth_bo)
            # *****************************************************************************

            print("Before PID: ", -rel_x, -rel_y)  # for debug

            # Bo：测试自动开枪！！在PID之前的距离更接近真实距离(虽然是按FOV换算后的鼠标距离)
            # TBD：开枪考虑远近距离，提前量？
            if abs(rel_x) < 2 and abs(rel_y) < 2 and (time.time() - self.shot_time > 3):
                ghub.mouse_down()
                ghub.mouse_up()
                self.shot_time = time.time()

            # if abs(rel_x) < 3 and abs(rel_y) < 3:  # for debug
            #     print('locked!!!')

            if self.lock_strategy == 'pid':
                # rel_x, rel_y = self.pid(rel_x, rel_y)
                rel_x, rel_y = self.pid2(rel_x, rel_y, delta_t)  # 考虑采样时间及抗饱和

            print("After PID: ", -rel_x, -rel_y)  # for debug

            ghub.mouse_xy(-rel_x, -rel_y)

        else:  # len(aims_copy) == 0，即没有要锁定的目标时
            if self.lock_strategy == 'pid':  # PID清0
                self.reset_pid_error()
            if time.time() - self.last_locked_time > 3:  # 自动开枪逻辑:3秒内没有锁定目标则鼠标移动
                ghub.mouse_xy(1500, 0)

    def recoil_control(self, args):
        """
        压枪函数：暂未使用
        pynput鼠标监听相关的使用案例=》建议参考J盘项目进行学习：J:\Project\pynput_mouse
        特别注意：B站up主的视频yolov5 csgo压枪代码(yolov5-6.1重构)，即采用的下面的阻塞版本。需要修改pynput的源码才能实现压枪，具体修改办法请参考：
        J:\Project\yolov5\QQ群里的代码及资料\常见问题解决方法.pdf。
        修改前后的区别在于：next(events)是否会阻塞。
        建议除了这个项目外，尽量不要动原来的pynput的源码，以免影响到的其他调用pynput的项目(新建单独conda环境)
        """
        ak47_recoil = []
        for i in csv.reader(open('./ak47.csv', encoding='utf-8-sig')):
            ak47_recoil.append([float(x) for x in i])
        k = -args.recoil_sen
        recoil_mode = False

        with pynput.mouse.Events() as events:
            for event in events:
                if isinstance(event, pynput.mouse.Events.Click):
                    if event.button == event.button.left:  # 左键按住与松开，表示开启和结束标识。
                        if event.pressed:
                            self.flag = 1
                        else:
                            self.flag = 0
                    if event.button == eval('event.button.' + 'x1') and event.pressed:  # 侧键x1键作为控制开关
                        recoil_mode = not recoil_mode
                        print('recoil mode', 'on' if recoil_mode else 'off')

                if self.flag and recoil_mode:
                    i = 0
                    a = next(events)  # 官方版本会阻塞，B站up主的方法修改源码后这里不阻塞！
                    print('1111---Received event {}'.format(a))  # Bo加的，便于查看
                    while True:
                        ghub.mouse_xy(int(-ak47_recoil[i][0] * k), int(ak47_recoil[i][1] * k))
                        time.sleep(ak47_recoil[i][2] / 1000 - 0.01)
                        i += 1
                        if i == 30:
                            break
                        if a is not None and isinstance(a,
                                                        pynput.mouse.Events.Click) and a.button == a.button.left and not a.pressed:
                            break
                        a = next(events)
                        print('2222---Received event {}'.format(a))  # Bo加的，便于查看
                        while a is not None and not isinstance(a, pynput.mouse.Events.Click):
                            a = next(events)
                            print('3333---Received event {}'.format(a))  # Bo加的，便于查看
                    # 跳出while循环，表示压枪结束！
                    self.flag = 0

# *******************************************************************************
# 启动一个线程，并将封装有事件监听的方法名作为target参数传入
# t = Thread(target=recoil_control)
# t.start()  # 非阻塞，程序会继续往下运行
# *******************************************************************************
