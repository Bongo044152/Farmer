import cv2 as cv
import numpy as np
import pyautogui

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
        n_x = max(400, width)
        n_y = max(300, height)

        # 顯示圖像並調整窗口大小
        cv.imshow(name, image)
        cv.resizeWindow(name, n_x, n_y)
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
        img = cv.imread(path, cv.IMREAD_COLOR_BGR)
        if img.any() == None:
            raise Exception(f"Image not found : {path}")
        return img


if __name__ == "__main__":
    image = Img_helper.load_img("./img/test1.png")
    print("Hello World !")
    Img_helper.show_img("test", image)
    print("this is a string for testing ...")