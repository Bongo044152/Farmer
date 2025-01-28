# 外工具
import cv2 as cv
import numpy as np

# 內工具
if __name__ == '__main__':
    from filter.hsvfilter import HsvFilter
    from filter.edgefilter import EdgeFilter
else:
    from utils.filter.hsvfilter import HsvFilter
    from utils.filter.edgefilter import EdgeFilter

# 額外需求

## 多線程
import concurrent.futures
from threading import Lock

## ???
from typing import Callable
import os

class ObjectDetection:

    def __init__(self, target_image_path: str, method: int = cv.TM_CCOEFF_NORMED) -> None:
        # 線程保護
        self.lock = Lock()

        # 如果路徑是相對路徑，將其轉換為絕對路徑
        if not os.path.isabs(target_image_path):
            target_image_path = os.path.join(os.getcwd(), target_image_path)

        # 檢查文件是否存在
        if not os.path.exists(target_image_path):
            raise FileNotFoundError(f"The file does not exist: {target_image_path}")

        self.neddle_image = cv.imread(target_image_path, cv.IMREAD_COLOR_BGR)
        self.needle_w = self.neddle_image.shape[1]
        self.needle_h = self.neddle_image.shape[0]

        self.method = method
        self.rectangles = []

        # Initialize thread pool executor
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.futures = []  # 用來存放每個線程的 future 物件
    
    def update_background_image(self, background_image: np.ndarray) -> 'ObjectDetection':
        self.screenshot = background_image
        self.rectangles.clear()
        return self

    def hsl_filter_method(self, hsl_parms: dict, tolerance: float = 0.7, max_results: int = 10) -> 'ObjectDetection':
        future = self.executor.submit(self._hsl_filter_task, hsl_parms, tolerance, max_results)
        self.futures.append(future)
        return self

    def _hsl_filter_task(self, hsl_parms: dict, tolerance: float, max_results: int) -> None:
        hsvfilter = HsvFilter.create_by_dict(hsl_parms)
        background_image = hsvfilter.apply_hsv_filter(self.screenshot)
        # neddle_image = hsvfilter.apply_hsv_filter(self.neddle_image)
        neddle_image = self.neddle_image

        with self.lock:
            self.rectangles.extend(self.process_rectangles(background_image, neddle_image, tolerance, max_results))

    def edge_filter_method(self, edge_parms: dict, tolerance: float = 0.7, max_results: int = 10) -> 'ObjectDetection':
        future = self.executor.submit(self._edge_filter_task, edge_parms, tolerance, max_results)
        self.futures.append(future)
        return self

    def _edge_filter_task(self, edge_parms: dict, tolerance: float, max_results: int) -> None:
        edgefilter = EdgeFilter.create_by_dict(edge_parms)
        background_image = edgefilter.apply_edge_filter(self.screenshot)
        # neddle_image = edgefilter.apply_edge_filter(self.neddle_image)
        neddle_image = self.neddle_image

        with self.lock:
            self.rectangles.extend(self.process_rectangles(background_image, neddle_image, tolerance, max_results))
    
    def find(self, tolerance: float = 0.85) -> 'ObjectDetection':
        future = self.executor.submit(self._find_task, tolerance)
        self.futures.append(future)
        return self

    def _find_task(self, tolerance: float) -> None:
        try:
            result = cv.matchTemplate(self.screenshot, self.neddle_image, self.method)
            # get best match position
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
            if max_val > tolerance:
                with self.lock:
                    self.rectangles = [
                        [max_loc[0], max_loc[1], self.needle_w, self.needle_h]
                    ]
            else:
                print(f"精確值過低! {max_val}，低於期望 {tolerance}，因此不採用!")
        except Exception as e:
            print(e)


    def find_muti(self, tolerance:float = 0.8, max_results: int = 5) -> 'ObjectDetection' :
        future = self.executor.submit(self._find_muti_task, tolerance, max_results)
        self.futures.append(future)

        return self
    
    def _find_muti_task(self, tolerance:float = 0.8, max_results: int = 5):
        try:
            with self.lock:
                self.rectangles.extend(self.process_rectangles(self.screenshot, self.neddle_image , tolerance, max_results))
        except Exception as e:
            print(e)


    def process_rectangles(self, background_image: np.ndarray, neddle_image: np.ndarray, tolerance: float=0.7, max_results: int=10) -> list:

        def in_order_rect(match_res: np.ndarray, rectangles: list[list[int, int, int, int]]) -> list[int, int, int, int]:
            '''
            in order the result from searched locations
            '''
            rectangles = [(rect, match_res[rect[1],rect[0]]) for rect in rectangles] ## 注意! match_res[y, x]
            rectangles = sorted(rectangles, key=lambda x : x[1], reverse=True)
            return [i[0] for i in rectangles]
        
        # run the OpenCV algorithm
        result = cv.matchTemplate(background_image, neddle_image, self.method)

        # Get the all the positions from the match result that exceed our threshold
        locations = np.where(result >= tolerance)
        locations = list(zip(*locations[::-1]))

        # if we found no results, return now. this reshape of the empty array allows us to 
        # concatenate together results without causing an error
        if not locations:
            return np.array([], dtype=np.int32).reshape(0, 4)

        # You'll notice a lot of overlapping rectangles get drawn. We can eliminate those redundant
        # locations by using groupRectangles().
        # First we need to create the list of [x, y, w, h] rectangles
        rectangles = []
        for loc in locations:
            rect = [int(loc[0]), int(loc[1]), self.needle_w, self.needle_h]
            # Add every box to the list twice in order to retain single (non-overlapping) boxes
            rectangles.append(rect)
            rectangles.append(rect)
        # Apply group rectangles.
        # The groupThreshold parameter should usually be 1. If you put it at 0 then no grouping is
        # done. If you put it at 2 then an object needs at least 3 overlapping rectangles to appear
        # in the result. I've set eps to 0.5, which is:
        # "Relative difference between sides of the rectangles to merge them into a group."
        rectangles, weights = cv.groupRectangles(rectangles, groupThreshold=1, eps=0.5)

        # sorted method
        rectangles = in_order_rect(result, rectangles)

        # 篩選最佳的前 max_results 結果
        if len(rectangles) > max_results:
            rectangles = rectangles[:max_results]

        rectangles = [rect.tolist() for rect in rectangles]
        return rectangles

    def end(self):
        concurrent.futures.wait(self.futures)
        self.futures.clear()  # 清空 futures，防止再次使用

    ########################################### GET RESULT ################################################
    
    # given a list of [x, y, w, h] rectangles and a canvas image to draw on, return an image with
    # all of those rectangles drawn
    def debug_draw_rectangles(self) -> np.ndarray:

        # 等待所有異步線程執行完成
        self.end()

        bacground_image = self.screenshot.copy()

        # these colors are actually BGR
        line_color = (0, 255, 0)
        line_type = cv.LINE_4

        for (x, y, w, h) in self.rectangles:
            # determine the box positions
            top_left = (x, y)
            bottom_right = (x + w, y + h)
            # draw the box
            cv.rectangle(bacground_image, top_left, bottom_right, line_color, lineType=line_type)

        return bacground_image

    # given a list of [x, y] positions and a canvas image to draw on, return an image with all
    # of those click points drawn on as crosshairs
    def debug_draw_crosshairs(self) -> np.ndarray:

        # 等待所有異步線程執行完成
        self.end()

        if not self.rectangles:
            print("請確保事件的數理順序正確!")
            exit(-1)

        bacground_image = self.screenshot.copy()

        # these colors are actually BGR
        marker_color = (255, 0, 255)
        marker_type = cv.MARKER_TILTED_CROSS

        for (center_x, center_y) in self.get_result():
            # draw the center point
            cv.drawMarker(bacground_image, (center_x, center_y), marker_color, marker_type, 25, 3)
        return bacground_image
    
    @property
    def get_result(self, tolerance: float = 0.7, max_results: int = 10) -> list[list[int, int]]:
        '''
        @return : get click point / center point of elements =>
            [
                [x1,y1],
                [x2,y2],
                ...
            ]
        '''

        # 等待所有異步線程執行完成
        self.end()

        with self.lock:
            if self.rectangles:
                # duplicate twice
                self.rectangles.extend(self.rectangles)
                self.rectangles = self.process_rectangles(self.screenshot, self.neddle_image , tolerance, max_results)

            # get the results of the rectangles as center points
            results = []
            for (x, y, w, h) in self.rectangles:
                results.append([
                    x + w // 2, 
                    y + h // 2
                ])
        return results
    
    @property
    def get_res_item_len(self) -> int :
        return len(self.rectangles)