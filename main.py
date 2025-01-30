from utils.windowcapture import WindowCapture
from utils.object_detection import ObjectDetection

import pyautogui

import cv2 as cv
from time import time

screen_width, screen_height = pyautogui.size()

wincap = WindowCapture()
wincap.start()

current_time = time()

try:
    target = ObjectDetection("./img/target2.png")
    while True:
        # wait until screenshot is ready
        if wincap.screenshot is None:
            continue

        result = target.update_background_image(wincap.screenshot).edge_filter_method(e, 0.4, 20).debug_draw_rectangles()

        print(f"FPS: {1 / (time() - current_time)}")
        current_time = time()

        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            wincap.end()
            break
except Exception as e:
    wincap.end()
    raise e
