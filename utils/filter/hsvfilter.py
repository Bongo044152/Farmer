import cv2 as cv
import numpy as np

class HsvFilter:

    TRACKBAR_WINDOW = "Trackbars_Hsv"

    def __init__(self, hMin=None, sMin=None, vMin=None, hMax=None, sMax=None, vMax=None, 
                    sAdd=None, sSub=None, vAdd=None, vSub=None):
        self.hMin = hMin
        self.sMin = sMin
        self.vMin = vMin
        self.hMax = hMax
        self.sMax = sMax
        self.vMax = vMax
        self.sAdd = sAdd
        self.sSub = sSub
        self.vAdd = vAdd
        self.vSub = vSub

    @staticmethod
    def create_by_dict(dictionary: dict) -> 'HsvFilter':
        """
        根據字典中的資料創建 HsvFilter 實例。

        :param dictionary: 包含 HSV 範圍和其他參數的字典
        :return: 返回創建的 HsvFilter 實例
        :raises KeyError: 如果字典中缺少必需的鍵（例如 'hMin', 'sMin', 'vMin' 等）
        """
        try:
            return HsvFilter(
                hMin=dictionary['hMin'],
                sMin=dictionary['sMin'],
                vMin=dictionary['vMin'],
                hMax=dictionary['hMax'],
                sMax=dictionary['sMax'],
                vMax=dictionary['vMax'],
                sAdd=dictionary['sAdd'],
                sSub=dictionary['sSub'],
                vAdd=dictionary['vAdd'],
                vSub=dictionary['vSub']
            )
        except KeyError as e:
            print(f"KeyError: Missing key {e.args[0]} in the dictionary")
            raise e

    # create gui window with controls for adjusting arguments in real-time
    @staticmethod
    def init_control_gui() -> None :
        cv.namedWindow(HsvFilter.TRACKBAR_WINDOW, cv.WINDOW_NORMAL)
        cv.resizeWindow(HsvFilter.TRACKBAR_WINDOW, 350, 700)

        # required callback. we'll be using getTrackbarPos() to do lookups
        # instead of using the callback.
        def nothing(position):
            pass

        # create trackbars for bracketing.

        # OpenCV scale for HSV is H: 0-179, S: 0-255, V: 0-255
        cv.createTrackbar('HMin', HsvFilter.TRACKBAR_WINDOW, 0, 179, nothing)
        cv.createTrackbar('SMin', HsvFilter.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VMin', HsvFilter.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('HMax', HsvFilter.TRACKBAR_WINDOW, 0, 179, nothing)
        cv.createTrackbar('SMax', HsvFilter.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VMax', HsvFilter.TRACKBAR_WINDOW, 0, 255, nothing)
        # Set default value for Max HSV trackbars
        cv.setTrackbarPos('HMax', HsvFilter.TRACKBAR_WINDOW, 179)
        cv.setTrackbarPos('SMax', HsvFilter.TRACKBAR_WINDOW, 255)
        cv.setTrackbarPos('VMax', HsvFilter.TRACKBAR_WINDOW, 255)

        # trackbars for increasing/decreasing saturation and value
        cv.createTrackbar('SAdd', HsvFilter.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('SSub', HsvFilter.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VAdd', HsvFilter.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VSub', HsvFilter.TRACKBAR_WINDOW, 0, 255, nothing)

    # returns an HSV filter object based on the control GUI values
    @staticmethod
    def get_hsv_filter_from_controls() -> 'HsvFilter':
        # Get current positions of all trackbars
        hsv_filter = HsvFilter()
        hsv_filter.hMin = cv.getTrackbarPos('HMin', HsvFilter.TRACKBAR_WINDOW)
        hsv_filter.sMin = cv.getTrackbarPos('SMin', HsvFilter.TRACKBAR_WINDOW)
        hsv_filter.vMin = cv.getTrackbarPos('VMin', HsvFilter.TRACKBAR_WINDOW)
        hsv_filter.hMax = cv.getTrackbarPos('HMax', HsvFilter.TRACKBAR_WINDOW)
        hsv_filter.sMax = cv.getTrackbarPos('SMax', HsvFilter.TRACKBAR_WINDOW)
        hsv_filter.vMax = cv.getTrackbarPos('VMax', HsvFilter.TRACKBAR_WINDOW)
        hsv_filter.sAdd = cv.getTrackbarPos('SAdd', HsvFilter.TRACKBAR_WINDOW)
        hsv_filter.sSub = cv.getTrackbarPos('SSub', HsvFilter.TRACKBAR_WINDOW)
        hsv_filter.vAdd = cv.getTrackbarPos('VAdd', HsvFilter.TRACKBAR_WINDOW)
        hsv_filter.vSub = cv.getTrackbarPos('VSub', HsvFilter.TRACKBAR_WINDOW)
        return hsv_filter
    
    # apply adjustments to an HSV channel
    # https://stackoverflow.com/questions/49697363/shifting-hsv-pixel-values-in-python-using-numpy
    def shift_channel(self, c, amount):
        if amount > 0:
            lim = 255 - amount
            c[c >= lim] = 255
            c[c < lim] += amount
        elif amount < 0:
            amount = -amount
            lim = amount
            c[c <= lim] = 0
            c[c > lim] -= amount
        return c
    
    # given an image and an HSV filter, apply the filter and return the resulting image.
    # if a filter is not supplied, the control GUI trackbars will be used
    def apply_hsv_filter(self, original_image: np.ndarray) -> np.ndarray:
        # convert image to HSV
        hsv = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)

        # add/subtract saturation and value
        h, s, v = cv.split(hsv)
        s = self.shift_channel(s, self.sAdd)
        s = self.shift_channel(s, -self.sSub)
        v = self.shift_channel(v, self.vAdd)
        v = self.shift_channel(v, -self.vSub)
        hsv = cv.merge([h, s, v])

        # Set minimum and maximum HSV values to display
        lower = np.array([self.hMin, self.sMin, self.vMin])
        upper = np.array([self.hMax, self.sMax, self.vMax])
        # Apply the thresholds
        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(hsv, hsv, mask=mask)

        # convert back to BGR for imshow() to display it properly
        img = cv.cvtColor(result, cv.COLOR_HSV2BGR)

        return img

if __name__ == '__main__':
    pass