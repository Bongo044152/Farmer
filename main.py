from utils.windowcapture import WindowCapture
from utils.object_detection import ObjectDetection
from utils.DEBUG import show_image

import pyautogui

import cv2 as cv
from time import time

wincap = WindowCapture()
wincap.start()

current_time = time()

h = {
    "hMin": 21,
    "sMin": 62,
    "vMin": 255,
    "hMax": 23,
    "sMax": 203,
    "vMax": 255,
    "sAdd": 33,
    "sSub": 32,
    "vAdd": 255,
    "vSub": 0
}

try:
    target = ObjectDetection("./img/target.png")
    while True:
        # wait until screenshot is ready
        if wincap.screenshot is None:
            continue

        result = target.update_background_image(wincap.screenshot).find_muti(0.7, 10).debug_draw_rectangles()
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
