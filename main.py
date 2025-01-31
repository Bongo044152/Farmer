from utils.windowcapture import WindowCapture
from utils.object_detection import ObjectDetection

import pyautogui

import cv2 as cv
from time import time

screen_width, screen_height = pyautogui.size()

wincap = WindowCapture()
wincap.start()

current_time = time()


target = ObjectDetection("./img/test1.png")
while True:
    # wait until screenshot is ready
    if wincap.screenshot is None:
        continue

    result = target.update_background_image(wincap.screenshot).find().debug_draw_rectangles()

    print(f"FPS: {1 / (time() - current_time)}")
    current_time = time()

    if cv.waitKey(1) == ord('q'):
        cv.destroyAllWindows()
        wincap.end()
        break
