import cv2 as cv
import numpy as np
import pyautogui, os

class Img_helper:
    def __init__(self) -> None:
        pass

    @staticmethod
    def show_img(image: np.ndarray, name: str = 'Image', once: bool = True) -> None:
        '''
        顯示圖像並調整顯示窗口大小，防止圖像過大無法顯示。

        @params:
            name : str : 顯示圖像的窗口名稱。
            image : np.ndarray : 需要顯示的圖像數組。
            once : bool : 是否只顯示一次並等待關閉窗口。預設為 True。
        
        @return:
            None : 不返回任何內容。
        '''
        screen_width, screen_height = pyautogui.size()

        width = image.shape[1]
        height = image.shape[0]

        # 如果圖像尺寸大於屏幕 80%，則縮小圖像
        if width > screen_width * 0.8 or height > screen_height * 0.8:
            width = int(width / 5 * 4)
            height = int(height / 5 * 4)
            dim = (width, height)
            image = cv.resize(image, dim, interpolation=cv.INTER_AREA)
        # 設置顯示窗口的最小大小
        elif width < 400 or height < 300:
            if width > height:
                scale = 400 / width
            else:
                scale = 300 / height
            width = int(width * scale)
            height = int(height * scale)
            dim = (width, height)
            image = cv.resize(image, dim, interpolation=cv.INTER_AREA)

        # 顯示圖像並調整窗口大小
        cv.imshow(name, image)
        if once:
            cv.waitKey(0)
            cv.destroyAllWindows()

    @staticmethod
    def save_img(img: np.ndarray, name: str) -> None:
        '''
        將圖像保存到指定路徑。

        @params:
            img : np.ndarray : 需要保存的圖像數組。
            name : str : 保存圖像的文件名。

        @return:
            None : 不返回任何內容。
        '''
        cv.imwrite(name, img)

    @staticmethod
    def load_img(path: str) -> np.ndarray:
        '''
        從指定路徑讀取圖像。

        @params:
            name : str : 圖像文件的路徑。

        @return:
            img : np.ndarray : 讀取到的圖像數組，形式為 BGR。
        '''
        if not os.path.isabs(path):
            path = os.path.abspath(path)

        if not os.path.exists(path):
            raise Exception(f"Image not found : {path}")
        else:
            path = os.path.abspath(path)
            img = cv.imread(path, cv.IMREAD_COLOR_BGR)
            return img


if __name__ == "__main__":
    # 範例一: 仔入圖像
    my_img = Img_helper.load_img("./img/test2.png")

    # 範例二: 顯示圖像
    Img_helper.show_img(my_img, name="My Image", once=True) # once=True 表示顯示一次並等待關閉窗口，差別救世會等待你關閉窗口才會執行後續程序

    # 範例三: 保存圖像
    Img_helper.save_img(my_img, "./img/test2_copy.png")