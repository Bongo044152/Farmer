import pydirectinput
import ctypes, time
import pyautogui

class MouseController:
    def __init__(self) -> None :
        pass

    def rotate_perspective(self, x_offset: int, y_offset: int, duration: float = 1) -> None :
        """
        Rotate the game character's perspective by moving the mouse.

        @prams:
        x_offset (int): Horizontal movement of the mouse.
        y_offset (int): Vertical movement of the mouse.
        duration (float): Total time for the movement (seconds).
        steps (int): Number of steps for the movement.
        """

        pydirectinput.moveRel(x_offset, y_offset, duration)

    @staticmethod
    def get_mouse_pos_by_click(right_bottom: tuple[int, int] = None, sensitive:float = 0.3) -> None :
        if not right_bottom:
            print("使用這個功能前，請確保你已經進行過校正!")
        
        # 定義 POINT 結構體
        class POINT(ctypes.Structure):
            _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

        # 取得滑鼠位置的函數
        def get_mouse_position():
            pt = POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            return pt.x, pt.y
        if right_bottom:
            screen_width, screen_height = pyautogui.size()
            scale_x = int(screen_width / right_bottom[0])
            scale_y = int(screen_height / right_bottom[1])
        else:
            scale_x = 1
            scale_y = 1

        # 主程式循環，等待滑鼠點擊
        while True:
            # 檢查滑鼠左鍵是否被按下
            if ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000:  # 0x01 是左鍵
                x, y = get_mouse_position()
                x *= scale_x
                y *= scale_y
                print(f"滑鼠點擊位置：({x}, {y})")
                time.sleep(sensitive)  # 防止多次觸發
            elif ctypes.windll.user32.GetAsyncKeyState(0x04) & 0x8000:  # 0x04 是中鍵
                break

# 使用範例
if __name__ == "__main__":
    # 使用前記得校準
    MouseController.get_mouse_pos_by_click()