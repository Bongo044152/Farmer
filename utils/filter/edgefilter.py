import cv2 as cv
import numpy as np

# custom data structure to hold the state of a Canny edge filter
class EdgeFilter:

    TRACKBAR_WINDOW = "Trackbars_Edge"

    def __init__(self, kernelSize=None, erodeIter=None, dilateIter=None, canny1=None, 
                    canny2=None):
        self.kernelSize = kernelSize
        self.erodeIter = erodeIter
        self.dilateIter = dilateIter
        self.canny1 = canny1
        self.canny2 = canny2

    # create gui window with controls for adjusting arguments in real-time
    @staticmethod
    def init_control_gui() -> None :
        cv.namedWindow(EdgeFilter.TRACKBAR_WINDOW, cv.WINDOW_NORMAL)
        cv.resizeWindow(EdgeFilter.TRACKBAR_WINDOW, 350, 350)

        # required callback. we'll be using getTrackbarPos() to do lookups
        # instead of using the callback.
        def nothing(position):
            pass

        # create trackbars for bracketing.

        # trackbars for edge creation
        cv.createTrackbar('KernelSize', EdgeFilter.TRACKBAR_WINDOW, 1, 30, nothing)
        cv.createTrackbar('ErodeIter', EdgeFilter.TRACKBAR_WINDOW, 1, 5, nothing)
        cv.createTrackbar('DilateIter', EdgeFilter.TRACKBAR_WINDOW, 1, 5, nothing)
        cv.createTrackbar('Canny1', EdgeFilter.TRACKBAR_WINDOW, 0, 200, nothing)
        cv.createTrackbar('Canny2', EdgeFilter.TRACKBAR_WINDOW, 0, 500, nothing)
        # Set default value for Canny trackbars
        cv.setTrackbarPos('KernelSize', EdgeFilter.TRACKBAR_WINDOW, 5)
        cv.setTrackbarPos('Canny1', EdgeFilter.TRACKBAR_WINDOW, 100)
        cv.setTrackbarPos('Canny2', EdgeFilter.TRACKBAR_WINDOW, 200)

    @staticmethod
    def create_by_dict(dictionary: dict) -> 'EdgeFilter':
        """
        根據字典中的資料創建 EdgeFilter 實例。

        :param dictionary: 包含 EdgeFilter 參數的字典，應包含 'kernelSize', 'erodeIter', 
                           'dilateIter', 'canny1', 'canny2' 這些鍵。
        :return: 返回創建的 EdgeFilter 實例。
        :raises KeyError: 如果字典中缺少必需的鍵（例如 'kernelSize', 'erodeIter', 'dilateIter' 等）。
        """
        try:
            return EdgeFilter(
                kernelSize=dictionary['KernelSize'],
                erodeIter=dictionary['ErodeIter'],
                dilateIter=dictionary['DilateIter'],
                canny1=dictionary['Canny1'],
                canny2=dictionary['Canny2']
            )
        except KeyError as e:
            print(f"KeyError: Missing key '{e.args[0]}' in the dictionary")
            raise e
    
    # returns a Canny edge filter object based on the control GUI values
    @staticmethod
    def get_edge_filter_from_controls() -> 'EdgeFilter' :
        # Get current positions of all trackbars
        edge_filter = EdgeFilter()
        edge_filter.kernelSize = cv.getTrackbarPos('KernelSize', EdgeFilter.TRACKBAR_WINDOW)
        edge_filter.erodeIter = cv.getTrackbarPos('ErodeIter', EdgeFilter.TRACKBAR_WINDOW)
        edge_filter.dilateIter = cv.getTrackbarPos('DilateIter', EdgeFilter.TRACKBAR_WINDOW)
        edge_filter.canny1 = cv.getTrackbarPos('Canny1', EdgeFilter.TRACKBAR_WINDOW)
        edge_filter.canny2 = cv.getTrackbarPos('Canny2', EdgeFilter.TRACKBAR_WINDOW)
        return edge_filter
    
    # given an image and a Canny edge filter, apply the filter and return the resulting image.
    # if a filter is not supplied, the control GUI trackbars will be used
    def apply_edge_filter(self, original_image: np.ndarray):

        kernel = np.ones((self.kernelSize, self.kernelSize), np.uint8)
        eroded_image = cv.erode(original_image, kernel, iterations=self.erodeIter)
        dilated_image = cv.dilate(eroded_image, kernel, iterations=self.dilateIter)

        # canny edge detection
        result = cv.Canny(dilated_image, self.canny1, self.canny2)

        # convert single channel image back to BGR
        img = cv.cvtColor(result, cv.COLOR_GRAY2BGR)

        return img

if __name__ == '__main__':
    pass