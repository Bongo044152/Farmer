import numpy as np
import cv2 as cv
import pytesseract
import os

tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 確認文件是否存在
if os.path.exists(tesseract_path):
    print(f"文件 '{tesseract_path}' 存在。")
else:
    print(f"請確保你使用的是 windows 系統或者你有下載過 tesseract-ocr")
    print(f"tesseract-ocr 下載地址: https://github.com/UB-Mannheim/tesseract/wiki")
    print(f"或者直接下載點擊此連結下載: https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe")
    exit(-1)

pytesseract.pytesseract.tesseract_cmd = tesseract_path


class OCR_detection:
    def __init__(self, background_img: np.ndarray) -> None:
        self.ROI = False
        self.background_img = background_img

    def enable_ROI(self, left_top: tuple[int, int], right_bottom: tuple[int, int]) -> 'OCR_detection':
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

    def get_text_xywh(self,target_str: str, conf_threshold: int = 65, language:str = 'chi_tra'):
        gray = cv.cvtColor(self.background_img, cv.COLOR_BGR2GRAY)
        # 二值化處理提高準確度
        _, binary = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
        data = pytesseract.image_to_data(binary, lang=language, output_type=pytesseract.Output.DICT)

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
    
    def detect_text(self, target_str: str, conf_threshold: int = 70, language: str = 'chi_tra'):
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
    
if __name__ == '__main__':
    from windowcapture import WindowCapture
    import time

    time.sleep(1.5)
    wincap = WindowCapture()
    img = wincap.get_screenshot()
    print(img.shape)
    ocr = OCR_detection(img)
    left_top = (img.shape[1]//4 * 3, img.shape[0]//2)
    right_bottom = (img.shape[1], img.shape[0])
    if ocr.enable_ROI(left_top, right_bottom).detect_text('資源回收筒'):
        print('find')
    else:
        print('not find')