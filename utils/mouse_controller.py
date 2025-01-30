import pydirectinput
import ctypes, time
import pyautogui

class MouseController:
    def __init__(self) -> None :
        pass

    def rotate_perspective(self, x_offset: int, y_offset: int, duration: float = 1, steps: int = 100) -> None :
        """
        Rotate the game character's perspective by moving the mouse.

        @prams:
        x_offset (int): Horizontal movement of the mouse.
        y_offset (int): Vertical movement of the mouse.
        duration (float): Total time for the movement (seconds).
        steps (int): Number of steps for the movement.
        """
        step_x = x_offset // steps
        step_y = y_offset // steps
        step_time = duration / steps

        for _ in range(steps):
            pydirectinput.moveRel(step_x, step_y)  # Relative mouse movement
            time.sleep(step_time)

    @staticmethod
    def get_mouse_pos_by_click(right_bottom: tuple[int, int] = None ):
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
                time.sleep(0.1)  # 防止多次觸發
            elif ctypes.windll.user32.GetAsyncKeyState(0x04) & 0x8000:  # 0x04 是中鍵
                break

if __name__ == "__main__":
    MouseController.get_mouse_pos_by_click((2559, 1599))