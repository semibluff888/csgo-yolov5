import cv2
import numpy as np
import win32gui
import win32ui
import win32con
import mss
import win32print
import win32api

from PyQt5.QtWidgets import QApplication
import sys

from ctypes import wintypes
import ctypes


def grab_screen_win32(region):
    hwin = win32gui.GetDesktopWindow()
    left, top, x2, y2 = region
    width = x2 - left + 1  # 少取一像素，无所谓
    height = y2 - top + 1

    hwindc = win32gui.GetWindowDC(hwin)
    srcdc = win32ui.CreateDCFromHandle(hwindc)
    memdc = srcdc.CreateCompatibleDC()
    bmp = win32ui.CreateBitmap()
    bmp.CreateCompatibleBitmap(srcdc, width, height)
    memdc.SelectObject(bmp)
    memdc.BitBlt((0, 0), (width, height), srcdc, (left, top), win32con.SRCCOPY)

    signedIntsArray = bmp.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype='uint8')
    img.shape = (height, width, 4)

    srcdc.DeleteDC()
    memdc.DeleteDC()
    win32gui.ReleaseDC(hwin, hwindc)
    win32gui.DeleteObject(bmp.GetHandle())

    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


cap = mss.mss()
def grab_screen_mss(monitor):
    return cv2.cvtColor(np.array(cap.grab(monitor)), cv2.COLOR_BGRA2BGR)


def get_real_resolution():
    hDC = win32gui.GetDC(0)
    wide = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)
    high = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)
    return {"wide": wide, "high": high}


def get_screen_size():
    wide = win32api.GetSystemMetrics(0)
    high = win32api.GetSystemMetrics(1)
    return {"wide": wide, "high": high}


def get_scaling():
    real_resolution = get_real_resolution()
    screen_size = get_screen_size()
    proportion = round(real_resolution['wide'] / screen_size['wide'], 2)
    return proportion


def get_parameters():
        x, y = get_screen_size().values()
        return 0, 0, x, y


# 可能是最快的方法！！！
# 使用pyqt5,可以截取指定名称的窗口，窗口被遮挡也不影响，但该窗口不能被最小化,否则显示黑屏
# 参考链接：https://www.jb51.net/article/248103.htm
def convertQImageToMat(incomingImage):
    '''  Converts a QImage into an opencv MAT format  '''
    # Format_RGB32 = 4,存入格式为B,G,R,A 对应 0,1,2,3
    # RGB32图像每个像素用32比特位表示，占4个字节，
    # R，G，B分量分别用8个bit表示，存储顺序为B，G，R，最后8个字节保留
    incomingImage = incomingImage.convertToFormat(4)
    width = incomingImage.width()
    height = incomingImage.height()

    ptr = incomingImage.bits()
    ptr.setsize(incomingImage.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)  # Copies the data
    # arr为BGRA，4通道图片
    # return arr
    # Bo: PNG就是BGRA这样的4通道格式，A是alpha透明度，需用cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)转化成BGR
    return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)


app = QApplication(sys.argv)
screen = QApplication.primaryScreen()
def grab_screen_pyqt5(hwnd):
    printscreen = screen.grabWindow(hwnd).toImage()
    return convertQImageToMat(printscreen)  # 将获取的图像从QImage转换为BGR格式


# 貌似网上的这个自定义方法，跟现在的官方win32gui.GetWindowRect(hwnd)结果已经差不多了，或许已经修复了
# https://www.likeinlove.com/info/132.html
# https://blog.csdn.net/See_Star/article/details/103940462
def get_window_rect(hwnd):
    try:
        f = ctypes.windll.dwmapi.DwmGetWindowAttribute
    except WindowsError:
        f = None
    if f:
        rect = ctypes.wintypes.RECT()
        DWMWA_EXTENDED_FRAME_BOUNDS = 9
        f(ctypes.wintypes.HWND(hwnd),
          ctypes.wintypes.DWORD(DWMWA_EXTENDED_FRAME_BOUNDS),
          ctypes.byref(rect),
          ctypes.sizeof(rect)
          )
        return rect.left, rect.top, rect.right, rect.bottom
