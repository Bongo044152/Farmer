import numpy as np
import win32gui, win32ui, win32con
import pyautogui
from threading import Thread, Lock

class WindowCapture:

    # constructor
    def __init__(self, window_name:str = None) -> None :
        # create a thread lock object
        self.lock = Lock()

        # find the handle for the window we want to capture.
        # if no window name is given, capture the entire screen
        if window_name is None:
            self.hwnd = win32gui.GetDesktopWindow()
            self.w, self.h = pyautogui.size()
        else:
            self.hwnd = win32gui.FindWindow(None, window_name)
            if not self.hwnd:
                raise Exception('Window not found: {}'.format(window_name))

        # get the window size
        window_rect = win32gui.GetWindowRect(self.hwnd)
        if not self.w or not self.h:
            self.w = window_rect[2] - window_rect[0]
            self.h = window_rect[3] - window_rect[1]
        
        # account for the window border and titlebar and cut them off
        border_pixels = 0 # 8
        titlebar_pixels = 0 # 30
        self.w = self.w - (border_pixels * 2)
        self.h = self.h - titlebar_pixels - border_pixels
        self.cropped_x = border_pixels
        self.cropped_y = titlebar_pixels

        # set the cropped coordinates offset so we can translate screenshot
        # images into actual screen positions
        self.offset_x = window_rect[0] + self.cropped_x
        self.offset_y = window_rect[1] + self.cropped_y

    def get_screenshot(self) -> np.ndarray :

        # get the window image data
        wDC = win32gui.GetWindowDC(self.hwnd)
        dcObj = win32ui.CreateDCFromHandle(wDC)
        cDC = dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        dataBitMap.CreateCompatibleBitmap(dcObj, self.w, self.h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0, 0), (self.w, self.h), dcObj, (self.cropped_x, self.cropped_y), win32con.SRCCOPY)

        # convert the raw data into a format opencv can read
        #dataBitMap.SaveBitmapFile(cDC, 'debug.bmp')
        signedIntsArray = dataBitMap.GetBitmapBits(True)
        img = np.fromstring(signedIntsArray, dtype='uint8')
        img.shape = (self.h, self.w, 4)

        # free resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())

        # drop the alpha channel, or cv.matchTemplate() will throw an error like:
        #   error: (-215:Assertion failed) (depth == CV_8U || depth == CV_32F) && type == _templ.type() 
        #   && _img.dims() <= 2 in function 'cv::matchTemplate'
        img = img[...,:3]

        # make image C_CONTIGUOUS to avoid errors that look like:
        #   File ... in draw_rectangles
        #   TypeError: an integer is required (got type tuple)
        # see the discussion here:
        # https://github.com/opencv/opencv/issues/14866#issuecomment-580207109
        img = np.ascontiguousarray(img)
        with self.lock:
            self.screenshot = img
        return img

    # find the name of the window you're interested in.
    # once you have it, update window_capture()
    # https://stackoverflow.com/questions/55547940/how-to-get-a-list-of-the-name-of-every-open-window
    @staticmethod
    def list_window_names() -> None :
        def winEnumHandler(hwnd, ctx):
            if win32gui.IsWindowVisible(hwnd):
                print(hex(hwnd), win32gui.GetWindowText(hwnd))
        win32gui.EnumWindows(winEnumHandler, None)

    # translate a pixel position on a screenshot image to a pixel position on the screen.
    # pos = (x, y)
    # WARNING: if you move the window being captured after execution is started, this will
    # return incorrect coordinates, because the window position is only calculated in
    # the __init__ constructor.
    def get_screen_position(self, pos):
        return (pos[0] + self.offset_x, pos[1] + self.offset_y)

    # threading methods

    def start(self) -> None :
        self.stopped = False
        self.screenshot = None
        t = Thread(target=self.run)
        t.daemon = True
        t.start()

    def end(self) -> None :
        self.stopped = True

    def run(self) -> None :
        # TODO: you can write your own time/iterations calculation to determine how fast this is
        while not self.stopped:
            # get an updated image of the game
            screenshot = self.get_screenshot()
            # lock the thread while updating the results
            with self.lock:
                self.screenshot = screenshot

# 使用範例
if __name__ == '__main__':
    from img_process import Img_helper # 圖片顯示工具

    # 創建拍照精靈
    wincap = WindowCapture()
    img_from_wincap = wincap.get_screenshot() # 取得螢幕截圖

    # 註記: wincap.get_screenshot() 取得照片後， 可以透過 wincap.screenshot 來取出
    # 不需要使用 img_from_wincap = wincap.get_screenshot() 的方式額外創建一個變數
    if np.array_equal(img_from_wincap, wincap.screenshot): # 如果兩張照片相同
        print("我們其實是一樣的!")                          # 輸出: "我們其實是一樣的!"

    Img_helper.show_img(wincap.screenshot) # 顯示圖片