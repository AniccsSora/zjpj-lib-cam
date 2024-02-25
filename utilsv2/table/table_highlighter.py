import cv2
import numpy as np
from utilsv2.gen_utils import *


def read_image(image_path: str) -> np.ndarray:
    """讀取圖像"""
    return cv2.imread(image_path)


def convert_to_gray(image: np.ndarray) -> np.ndarray:
    """轉換圖像為灰度圖"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detect_edges(gray_image: np.ndarray) -> np.ndarray:
    """使用 Canny 方法檢測邊緣"""
    return cv2.Canny(gray_image, 50, 150, apertureSize=3)


def detect_Hough_lines(edges: np.ndarray) -> np.ndarray:
    """使用 Hough 變換檢測直線"""
    return cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=50)

def detect_Hough_lines_v2(edges: np.ndarray) -> np.ndarray:
    """使用 Hough 變換檢測直線
    但是會將圖片放大至 1000 up 再次做計算
    """
    edges = resize_max_res(edges, 1000)
    return detect_Hough_lines(edges)


def draw_lines(image, lines, color=(0, 255, 0), thickness=2):
    """在圖像上繪製直線"""
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    return image


def erode_dilate_image(image: np.ndarray,
                       erode_iterations: int = 1,
                       dilate_iterations: int = 1,
                       first_operation: str = 'erode') -> np.ndarray:
    """
    對圖片進行等量的侵蝕和膨脹操作。

    :param image: 輸入的圖片（NumPy 數組）。
    :param iterations: 侵蝕和膨脹操作的次數。
    :param first_operation: 首先執行的操作，可以是 'erode'(侵蝕) 或 'dilate'(膨脹)。
    :return: 處理後的圖片。
    # example
    #做個膨脹後侵蝕
    erode_iter = 3   # '侵蝕' 操作的次數
    dilate_iter = 3  # '膨脹' 操作的次數
    edges_dilate_erode = erode_dilate_image(edges,
                                            erode_iterations=erode_iter,
                                            dilate_iterations=dilate_iter,
                                            first_operation='dilate')

    """
    image = image.copy()
    # 定義結構元素
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 根據參數選擇先進行哪種操作
    if first_operation == 'erode':
        image = cv2.erode(image, kernel, iterations=erode_iterations)
        image = cv2.dilate(image, kernel, iterations=dilate_iterations)
    elif first_operation == 'dilate':
        image = cv2.dilate(image, kernel, iterations=dilate_iterations)
        image = cv2.erode(image, kernel, iterations=erode_iterations)
    else:
        raise ValueError("first_operation must be 'erode' or 'dilate'")

    return image


def filter_hough_lines(hough_lines: np.ndarray, filter_q)-> np.ndarray:
    """過濾 Hough 變換檢測到的直線"""
    filtered_lines = []
    for line in hough_lines:
        x1, y1, x2, y2 = line[0]
        # cals p1 & p2 distance
        distance = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        filtered_lines.append(distance)
    #
    # 用 matplotlib 來繪製各長度區間直方圖

    filtered_lines = sorted(filtered_lines)
    remove_len = np.percentile(filtered_lines, filter_q)
    #print("remove length: ", remove_len)

    filtered_lines_2 = []
    for line in hough_lines:
        x1, y1, x2, y2 = line[0]
        # cals p1 & p2 distance
        distance = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        if distance > remove_len:
            filtered_lines_2.append(line[0])

    # list to reshape like input hough_lines's ndarray.
    ret = np.array(filtered_lines_2, dtype=np.int32).reshape((-1, 1, 4))
    return ret


if __name__ == "__main__":
    image_path = './images/nobody_lib_table.jpg'
    image = read_image(image_path)
    gray_image = convert_to_gray(image)
    # 邊緣檢測
    edges = detect_edges(gray_image)
    # hough 找出線段
    hough_line = detect_Hough_lines(edges)
    # 過濾後的 hough 線段組
    hough_line_filter = filter_hough_lines(hough_line, filter_q=30)
    # 繪製直線
    hough_line_image = draw_lines(np.zeros_like(image, dtype=np.uint8), hough_line)
    # 繪製直線 (過濾後)
    hough_line_image_filter = draw_lines(np.zeros_like(image, dtype=np.uint8), hough_line_filter)
    #
    image_with_lines = draw_lines(image.copy(), hough_line)
    blank_image = np.zeros_like(image, dtype=np.uint8)
    resized_image = resize_max_res(image_with_lines, 800)

    # 顯示每一步的結果
    cv2.imshow('Original Image', resize_max_res(image))
    cv2.imshow('Gray Image', resize_max_res(gray_image))
    cv2.imshow('Hough Line', resize_max_res(hough_line_image))
    cv2.imshow('Edges', resize_max_res(edges))
    cv2.imshow('Lines Detected', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plot_images(
        images=[image, gray_image,
                edges, hough_line_image_filter,
                hough_line_image, image_with_lines,
                ],
        titles=[f'原始圖像_{image.ndim}', f'灰度圖_{gray_image.ndim}',
                '邊緣檢測', 'hough_line_image_filter',
                'Hough Lines', '檢測到的直線',
                ]
    )

