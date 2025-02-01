import numpy as np
import cv2 as cv
import pytesseract
import os

tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 確認文件是否存在
if not os.path.exists(tesseract_path):
    print(f"請確保你使用的是 windows 系統或者你有下載過 tesseract-ocr")
    print(f"tesseract-ocr 下載地址: https://github.com/UB-Mannheim/tesseract/wiki")
    print(f"或者直接下載點擊此連結下載: https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe")
    exit(-1)

pytesseract.pytesseract.tesseract_cmd = tesseract_path


class OCR_detection:
    '''
    OCR 檢測器，用於從圖片中提取指定文本的位置和大小資訊
    '''
    def __init__(self, background_img: np.ndarray) -> None:
        '''
        @params:
            - background_img (np.ndarray): 背景圖像
        '''
        self.ROI = False
        self.background_img = background_img

    def enable_ROI(self, left_top: tuple[int, int], right_bottom: tuple[int, int]) -> 'OCR_detection':
        '''
        ROI : Region of Interest

        指定感興趣的區域，只針對感興趣的區域進行搜尋。
        @params:
            - left_top (tuple): 左上角座標 (x, y)
            - right_bottom (tuple): 右下角座標 (x, y)

        @return:
            - self (OCR_detection)
        '''
        self.ROI = True

        # 確保 ROI 大小大於目標圖像大小
        roi_width = right_bottom[0] - left_top[0]
        roi_height = right_bottom[1] - left_top[1]

        bac_h, bac_w = self.background_img.shape[:2]

        if roi_width > bac_w or roi_height > bac_h:
            raise ValueError(f"ROI size must be smaller than background image size. "
                            f"Required: width <= {bac_w}, height <= {bac_h}, "
                            f"but got: width = {roi_width}, height = {roi_height}")
        
        self.ROIleft = left_top[0]
        self.ROItop = left_top[1]

        self.original_background_img = self.background_img.copy()
        self.background_img = self.background_img[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]]

        return self

    def get_text_xywh(self,target_str: str, conf_threshold: int = 65, language:str = 'chi_tra') -> tuple[int, int, int, int]:
        '''
        使用 Tesseract OCR 從圖片中提取指定文本 (target_str) 的位置和大小資訊，並返回其邊界框 (x, y, w, h)。
        
        @params:
            - target_str (str): 目標文字字符串，用來在圖片中搜尋
            - conf_threshold (int): 設定 OCR 識別信心分數的閾值，低於此分數的文字會被排除。預設為 65
            - language (str): 使用的語言模型，預設為 'chi_tra' (繁體中文)，更多支援請參考文檔: https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc#languages-and-scripts

        @reaturn:
            - tuple: 返回一個四元組 (x, y, w, h)，分別表示目標文字在圖片中的位置和大小
                如果未找到目標文字，則返回 (0, 0, 0, 0)
        '''
        gray = cv.cvtColor(self.background_img, cv.COLOR_BGR2GRAY)
        # 二值化處理提高準確度
        _, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
        data = pytesseract.image_to_data(binary, lang=language, output_type=pytesseract.Output.DICT)


        # 檢查機制 : 至少找到一個 target_str 文字
        found_target = False
        for char in data['text']:
            if char in target_str:
                found_target = True
                break
        if not found_target:
            print(f"Text '{target_str}' not found.")
            return (0, 0, 0, 0)

        # 過濾訊息
        del data['level']
        del data['page_num']
        del data['par_num']
        del data['block_num']
        del data['line_num']
        del data['word_num']
        
        to_delete = []
        
        for i in range(len(data['text']) - 1, -1, -1):
            char = data['text'][i]
            score = data['conf'][i]

            if len(char) == 0:
                to_delete.append(i)
            elif char in target_str:
                pass
            elif score < conf_threshold or char not in target_str:
                to_delete.append(i)

        # 刪除標記的索引
        for i in to_delete:
            del data['text'][i]
            del data['width'][i]
            del data['top'][i]
            del data['height'][i]
            del data['left'][i]
            del data['conf'][i]
        
        # get text [x, y, w, h]
        left = min(data['left']) - 2
        top = min(data['top'])

        right = 0
        bottom = 0
        for i in range(len(data['text'])):
            right = max(right, data['left'][i] + data['width'][i])
            bottom = max(bottom, data['top'][i] + data['height'][i])
        
        width = right - left
        height = bottom - top

        if self.ROI:
            left += self.ROIleft
            top += self.ROItop

        return left, top, width, height
    
    def draw_rectangle(self, x: int, y: int, w: int, h: int):
        '''
        繪製出目標的所在位置，並返回圖片。

        @params:
            - x (int): 目標的左上角 x 座標
            - y (int): 目標的左上角 y 座標
            - w (int): 目標的寬度
            - h (int): 目標的高度

        @return:
            - np.ndarray: 繪製出目標的所在位置的圖片

        @raise:
            - ValueError: 如果座標或尺寸錯誤或超出圖片範圍 (不合理的數值)
        '''
        background_img = self.background_img.copy()

        # 檢查座標是否錯誤
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            raise ValueError("座標或尺寸錯誤")
        elif x + w > background_img.shape[1] or y + h > background_img.shape[0]:
            raise ValueError("座標或尺寸超出圖片範圍")
        
        return cv.rectangle(
            background_img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
            cv.LINE_4
        )
    
    def detect_text(self, target_str: str, conf_threshold: int = 70, language: str = 'chi_tra'):
        '''
        檢測範圍內是否有出現特定文字/目標文字。

        @params:
            - target_str (str): 目標文字
            - conf_threshold (int): 識別信心門檻，預設為 70
            - language (str): 使用的語言模型，預設為 'chi_tra' (繁體中文)，更多支援請參考文檔: https://github.com/tesseract-ocr/tesseract/blob/main/doc/tesseract.1.asc#languages-and-scripts

        @return:
            - bool: 如果找到目標文字，則返回 True，否則返回 False
        '''
        gray = cv.cvtColor(self.background_img, cv.COLOR_BGR2GRAY)
        _, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)

        data = pytesseract.image_to_data(binary, lang=language, output_type=pytesseract.Output.DICT)
        
        total = len(target_str)
        points = 0

        # 檢查每個識別的單詞
        for i in range(len(data['text'])):
            word = data['text'][i].strip()
            
            # 檢查該單詞是否包含在目標字符串中
            if word and any(char in target_str for char in word):
                points += 1

        if points / total > conf_threshold / 100:
            return True
        else:
            return False

# 使用範例
if __name__ == '__main__':
    import time

    def example_1():
        '''使用範例 1: 搜尋特定文字'''

        # 創建幫忙獲取圖片的輔助精靈
        from windowcapture import WindowCapture
        wincap = WindowCapture()
        time.sleep(1.5)
        img = wincap.get_screenshot() # 獲取圖片

        # 提供掃描/偵測的區域
        ocr = OCR_detection(img)

        # 使用 ROI
        left_top = (0,0)
        right_bottom = (img.shape[1]//4, img.shape[0]//4)
        ocr.enable_ROI(left_top, right_bottom)

        # 搜尋關鍵字: "資源回收筒"
        print("嘗試搜尋螢幕左上角是否有出現關鍵字: '資源回收筒'")
        if ocr.detect_text('資源回收筒'):
            print('find')
        else:
            print('not find')
    
    def example_2():
        '''使用範例 2: 搜尋特定文字並取得座標'''

        # 獲取圖片
        from img_process import Img_helper
        my_image = Img_helper.load_img('./img/test2.png') # 仔入圖片
        Img_helper.show_img(my_image) # 顯示圖片

        # 提供掃描/偵測的區域
        ocr = OCR_detection(my_image)

        # 搜尋關鍵字: "資源回收筒"
        x, y, w, h = ocr.get_text_xywh('資源回收筒')
        return_img = ocr.draw_rectangle(x, y, w, h) # 繪製區域
        Img_helper.show_img(return_img) # 顯示圖片

    example_1()