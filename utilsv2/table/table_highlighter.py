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
    return cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)


def draw_lines(image, lines):
    """在圖像上繪製直線"""
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image


def resize_image(image, scale_percent=50):
    """調整圖像大小"""
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    image_path = './images/nobody_lib_table.jpg'
    image = read_image(image_path)
    gray_image = convert_to_gray(image)
    edges = detect_edges(gray_image)
    hough_line = detect_Hough_lines(edges)
    image_with_lines = draw_lines(image.copy(), hough_line)
    resized_image = resize_max_res(image_with_lines, 800)

    # 顯示每一步的結果
    cv2.imshow('Original Image', resize_max_res(image))
    cv2.imshow('Gray Image', resize_max_res(gray_image))
    cv2.imshow('Edges', resize_max_res(edges))
    cv2.imshow('Lines Detected', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plot_images(
        [image, gray_image, edges, image_with_lines],
        [f'原始圖像_{image.ndim}', f'灰度圖_{gray_image.ndim}', '邊緣檢測', '檢測到的直線']
    )

