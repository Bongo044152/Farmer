import win32api
import win32con
import pyautogui

import threading
from queue import Queue

import time, sys, os

import json
with open('./utils/VK_CODE.json', 'r', encoding='utf-8') as f:
    global VK_CODE
    VK_CODE = json.load(f)

def _get_key_code(key: str) -> int:
    try:
        return VK_CODE[key]
    except KeyError:
        print(f"錯誤的代碼: {hex(key)} ， 請查看 utils/VK_CODE.json")
        raise ValueError(f"Invalid key: {key}")

class KeyboardController:

    def __init__(self):
        self.event_queue = Queue()
        self.stop_flag = False
        self.processing = False
        my_thread = threading.Thread(target=self.__thread_handler)
        my_thread.daemon = True
        my_thread.start()

    def stop(self) -> None:
        self.stop_flag = True

    def start(self) -> None:
        self.stop_flag = False

    def wait(self) -> None:
        while not self.event_queue.empty() or self.processing:
            time.sleep(0.1)

    def __thread_handler(self):
        while not self.stop_flag:
            my_fun = self.event_queue.get()
            self.processing = True
            my_fun()

    def press_key(self, key: str, t: float = None) -> 'KeyboardController' :
        if t:
            self.event_queue.put(lambda: self.__press_key_task(key, t))
        else:
            self.event_queue.put(lambda: self.__press_key_once(key))
        
        return self

    def __press_key_once(self, key: str) -> None :
        key_code = _get_key_code(key)
        win32api.keybd_event(key_code, 0, 0, 0)
        win32api.keybd_event(key_code, 0, win32con.KEYEVENTF_KEYUP, 0)

    def __press_key_task(self, key: str, t: float, time_step: float = 0.02) -> None :
        key_code = _get_key_code(key)

        # 50次每秒
        for _ in range(int(t * 50)):
            win32api.keybd_event(key_code, 0, 0, 0)
            time.sleep(time_step)  # 控制每次間隔，防止 cpu 過於忙碌
        win32api.keybd_event(key_code, 0, win32con.KEYEVENTF_KEYUP, 0)

        # 任務處理結束，設置為 False
        self.processing = False

# 使用範例
if __name__ == "__main__":
    # 創建物件
    keyboard = KeyboardController()

    # 按下 w 鍵
    t = 1  # 設定持續按下時間
    keyboard.press_key("w", t)
    keyboard.press_key('a') # 點擊 a 鍵 一次

    # 其他事情 -> main thread
    for _ in range(5):
        print(666)
        time.sleep(0.1)

    # 等待所有指派給 keyboard 的任務完成
    keyboard.wait()

    # 結束
    print("END")

# 