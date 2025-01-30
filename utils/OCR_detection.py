import numpy as np
import cv2 as cv
import pytesseract
import os

tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 確認文件是否存在
if os.path.exists(tesseract_path):
    print(f"文件 '{tesseract_path}' 存在。")
else:
    print(f"文件 '{tesseract_path}' 不存在。")

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

        bac_w, bac_h = self.background_img.shape[:2]

        if roi_width > bac_w or roi_height > bac_h:
            raise ValueError(f"ROI size must be smaller than background image size. "
                            f"Required: width <= {bac_w}, height <= {bac_w}, "
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

        print(data)

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
        height = buttom - top

        print(data)

        if self.ROI:
            left += self.ROIleft
            top += self.ROItop

        return left, top, width, height
    
if __name__ == '__main__':
    from windowcapture import WindowCapture
    from img_process import Img_helper
    import time
    time.sleep(2)
    wincap = WindowCapture()
    img = wincap.get_screenshot()
    ocr = OCR_detection(img)
    left_top = (0, 0)
    right_bottom = (img.shape[1]//4, img.shape[0]//4)
    left, top, width, height = ocr.enable_ROI(left_top, right_bottom).get_text_xywh('資源回收筒')
    right = left + width
    buttom = top + height
    img = cv.rectangle(img, (left, top), (right, buttom), (0, 255, 0), 2)
    Img_helper.show_img(img)