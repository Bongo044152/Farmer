# 王農民的自動化打遊戲工具包

本專案用於幫助王農民自動化打遊戲，並提供相關的程式碼、工具和資源，主要用於私人用途，因此不提供詳細說明。

<font color:"red">注意! 僅支持 windows 系統</font>

## 功能

主要內容都位於 utils 資料夾中:

- OCR 檢測 -> ./utils/OCR_detection.py
- 物件檢測 -> ./utils/Object_detection.py
- 螢幕截圖 -> ./utils/windowcapture.py
- 圖像處理 -> ./utils/image_process.py ( 與 cv2 功能相同 )
- 鼠標控制 -> ./utils/mouse_control.py
<!-- - 運動控制 -> ./utils/Motion_control.py -->

## 使用

建議使用 python 13 或以上版本。

- 建構指令 :
```shell
git clone git@github.com:Bongo044152/Farmer.git
python -m venv .venv # 可選，建立 python 的虛擬環境
pip install -r requirements.txt
```

- 安裝所需資源: https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe ( OCR辨識所需資源 : tesseract )

## 資源

- [OpenCV docs](https://docs.opencv.org/4.x/index.html)
- [tesseract](https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0)