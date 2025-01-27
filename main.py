from utils.windowcapture import WindowCapture
from utils.object_detection import ObjectDetection
from utils.DEBUG import show_image

import pyautogui

import cv2 as cv
from time import time

wincap = WindowCapture()
wincap.start()

current_time = time()

def nothing():
    pass

a = 0

try:
    while True:
        # wait until screenshot is ready
        if wincap.screenshot is None:
            continue

        target = ObjectDetection(wincap.screenshot, "target2.png", nothing)
        result = target.find().debug_draw_rectangles()

        show_image("Result", result)

        print(f"FPS: {1 / (time() - current_time)}")
        current_time = time()

        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            wincap.end()
            break
except Exception as e:
    wincap.end()
    raise e
