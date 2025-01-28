import numpy as np
import cv2 as cv
import pyautogui

if __name__ == '__main__':
    from filter.hsvfilter import HsvFilter
    from filter.edgefilter import EdgeFilter
    from windowcapture import WindowCapture
else:
    from utils.filter.hsvfilter import HsvFilter
    from utils.filter.edgefilter import EdgeFilter
    from utils.windowcapture import WindowCapture

def show_image(name: str, image: np.ndarray) -> None:
    '''
    顯示圖像並調整顯示窗口大小，防止圖像過大無法顯示。

    @params:
        name : str : 顯示圖像的窗口名稱。
        image : np.ndarray : 需要顯示的圖像數組。
    
    @return:
        None : 不返回任何內容。
    '''
    # 獲取屏幕分辨率
    screen_width, screen_height = pyautogui.size()

    # 取得圖像的寬和高
    width = image.shape[1]
    height = image.shape[0]

    # 如果圖像尺寸大於屏幕 80%，則縮小圖像
    if width > screen_width * 0.8:
        width = int(width / 5 * 4)
    if height > screen_height * 0.8:
        height = int(height / 5 * 4)
    
    dim = (width, height)
    resized_image = cv.resize(image, dim, interpolation=cv.INTER_AREA)
    
    # 設置顯示窗口的最小大小
    n_x = max(400, width)
    n_y = max(300, height)

    # 顯示圖像並調整窗口大小
    cv.imshow(name, resized_image)
    cv.resizeWindow(name, n_x, n_y)

class DEBUG:
    def __init__(self) -> None:
        pass

    # given a list of [x, y, w, h] rectangles and a canvas image to draw on, return an image with
    # all of those rectangles drawn
    def draw_rectangles(self, haystack_img, rectangles):
        # these colors are actually BGR
        line_color = (0, 255, 0) # green color
        line_type = cv.LINE_4

        for (x, y, w, h) in rectangles:
            # determine the box positions
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            # draw the box
            cv.rectangle(haystack_img, top_left, bottom_right, line_color, lineType=line_type)

        return haystack_img
    
    # given an image and an HSV filter, apply the filter and return the resulting image.
    # if a filter is not supplied, the control GUI trackbars will be used
    def apply_hsv_filter(self, original_image, hsvfilter: HsvFilter = None):

        if not hsvfilter:
            hsvfilter = HsvFilter.get_hsv_filter_from_controls()

        # convert image to HSV
        hsv = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)

        # add/subtract saturation and value
        h, s, v = cv.split(hsv)
        s = hsvfilter.shift_channel(s, hsvfilter.sAdd)
        s = hsvfilter.shift_channel(s, -hsvfilter.sSub)
        v = hsvfilter.shift_channel(v, hsvfilter.vAdd)
        v = hsvfilter.shift_channel(v, -hsvfilter.vSub)
        hsv = cv.merge([h, s, v])

        # Set minimum and maximum HSV values to display
        lower = np.array([hsvfilter.hMin, hsvfilter.sMin, hsvfilter.vMin])
        upper = np.array([hsvfilter.hMax, hsvfilter.sMax, hsvfilter.vMax])
        # Apply the thresholds
        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(hsv, hsv, mask=mask)

        # convert back to BGR for imshow() to display it properly
        img = cv.cvtColor(result, cv.COLOR_HSV2BGR)

        return img
    
    # given an image and a Canny edge filter, apply the filter and return the resulting image.
    # if a filter is not supplied, the control GUI trackbars will be used
    def apply_edge_filter(self, original_image: np.ndarray, edge_filter: EdgeFilter = None):

        if edge_filter is None:
            edge_filter = EdgeFilter.get_edge_filter_from_controls()

        kernel = np.ones((edge_filter.kernelSize, edge_filter.kernelSize), np.uint8)
        eroded_image = cv.erode(original_image, kernel, iterations=edge_filter.erodeIter)
        dilated_image = cv.dilate(eroded_image, kernel, iterations=edge_filter.dilateIter)

        # canny edge detection
        result = cv.Canny(dilated_image, edge_filter.canny1, edge_filter.canny2)

        # convert single channel image back to BGR
        img = cv.cvtColor(result, cv.COLOR_GRAY2BGR)

        return img

# 列出所有的視窗名稱
def list_windows():
    WindowCapture.list_window_names()

# "色彩" 為參數的 debug
def hsv_filter_debug(dictionary: dict = None) -> None:
    if not dictionary:
        HsvFilter.init_control_gui()
        hsvfilter = None
    else:
        hsvfilter = HsvFilter.create_by_dict(dictionary)
    try:
        debuger = DEBUG()
        wincap = WindowCapture()
        wincap.start()

        while True:
            if wincap.screenshot is None:
                continue
            
            hsv_res = debuger.apply_hsv_filter(wincap.screenshot, hsvfilter)

            cv.imshow("HSV debug result", hsv_res)

            # press 'q' with the output window focused to exit.
            # waits 1 ms every loop to process key presses
            key = cv.waitKey(1)
            if key == ord('q'):

                if hsvfilter is None:
                    hsvfilter = HsvFilter.get_hsv_filter_from_controls()
                wincap.end()
                cv.destroyAllWindows()

                import json
                dictionary  = {}
                dictionary['hMin'] = hsvfilter.hMin
                dictionary['sMin'] = hsvfilter.sMin
                dictionary['vMin'] = hsvfilter.vMin
                dictionary['hMax'] = hsvfilter.hMax
                dictionary['sMax'] = hsvfilter.sMax
                dictionary['vMax'] = hsvfilter.vMax
                dictionary['sAdd'] = hsvfilter.sAdd
                dictionary['sSub'] = hsvfilter.sSub
                dictionary['vAdd'] = hsvfilter.vAdd
                dictionary['vSub'] = hsvfilter.vSub
                print(json.dumps(dictionary, ensure_ascii=False, indent=4))

                break
        print("END.")
    except Exception as e:
        print("退出線程...")
        wincap.end()
        raise e

# "邊" 為參數的 debug
def edge_filter_debug(dictionary: dict = None) -> None:
    if not dictionary:
        EdgeFilter.init_control_gui()
        edgefilter = None
    else:
        edgefilter = EdgeFilter.create_by_dict(dictionary)
    
    try:
        debuger = DEBUG()
        wincap = WindowCapture()
        wincap.start()

        while True:
            if wincap.screenshot is None:
                continue
        
            edge_res = debuger.apply_edge_filter(wincap.screenshot, edgefilter)

            cv.imshow("EDGE debug result", edge_res)

            # press 'q' with the output window focused to exit.
            # waits 1 ms every loop to process key presses
            key = cv.waitKey(1)
            if key == ord('q'):

                if edgefilter is None:
                    edgefilter = EdgeFilter.get_edge_filter_from_controls()
                wincap.end()
                cv.destroyAllWindows()

                import json
                dictionary  = {}
                dictionary ['KernelSize'] = edgefilter.kernelSize
                dictionary ['ErodeIter'] = edgefilter.erodeIter
                dictionary ['DilateIter'] = edgefilter.dilateIter
                dictionary ['Canny1'] = edgefilter.canny1
                dictionary['Canny2'] = edgefilter.canny2
                print(json.dumps(dictionary, ensure_ascii=False, indent=4))

                break
        print("END.")
    except Exception as e:
        print("退出線程...")
        wincap.end()
        raise e

# 程式執行的地方
if __name__ == '__main__':
    e = {
        "KernelSize": 10,
        "ErodeIter": 1,
        "DilateIter": 0,
        "Canny1": 200,
        "Canny2": 500
    }
    edge_filter_debug(e)