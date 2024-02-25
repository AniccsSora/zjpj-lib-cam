import cv2
import numpy as np
from matplotlib.font_manager import fontManager
import matplotlib
import matplotlib.pyplot as plt
from typing import List
import math


# 重新設定字體，先確定可以使用的字體
_useable_fonts = [_ for _ in sorted(fontManager.get_font_names())]
if 'Microsoft JhengHei' in _useable_fonts:
    matplotlib.rc('font', family='Microsoft JhengHei')

def resize_max_res(image: np.ndarray, max_resolution: int = 800) -> np.ndarray:
    """
     調整圖片的大小，以限制最大一邊的解析度，並保持長寬比例。如果最大邊小於 max_resolution，
     則將最大邊調整至 max_resolution，同時保持長寬比。

     :param image: 輸入圖片（NumPy 數組）
     :param max_resolution: 最大解析度（一個整數，表示最大一邊的像素數）
     :return: 調整大小後的圖片
     """
    original_height, original_width = image.shape[:2]
    max_side = max(original_height, original_width)

    # 判斷是否需要放大
    if max_side < max_resolution:
        # 放大至 max_resolution，同時保持長寬比
        scale_factor = max_resolution / max_side
    else:
        # 縮小至 max_resolution，同時保持長寬比
        scale_factor = max_resolution / max_side

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image
    # """
    # 調整圖片的大小，以限制最大一邊的解析度，並保持長寬比例。
    #
    # :param image: 輸入圖片（NumPy 數組）
    # :param max_resolution: 最大解析度（一個整數，表示最大一邊的像素數）
    # :return: 調整大小後的圖片
    # """
    # # 獲取原始圖片的寬度和高度
    # original_height, original_width = image.shape[:2]
    #
    # # 確定調整大小的目標寬度和高度
    # if original_width > original_height:
    #     # 如果寬度大於高度，將寬度調整為最大解析度，並保持長寬比例
    #     new_width = max_resolution
    #     new_height = int(original_height * (max_resolution / original_width))
    # else:
    #     # 如果高度大於寬度，將高度調整為最大解析度，並保持長寬比例
    #     new_height = max_resolution
    #     new_width = int(original_width * (max_resolution / original_height))
    #
    # # 使用 OpenCV 的 resize 函數進行調整大小
    # resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    #
    # return resized_image


def plot_images(images: List[np.ndarray],
                titles: List[str],
                images_per_row: int = 2,
                figsize=(10, 10)) -> None:
    """繪製多個圖像

    : USE CASE
    image_rgb = cv2.cvtColor(resize_max_res(image, 800), cv2.COLOR_BGR2RGB)
    gray_rgb = cv2.cvtColor(resize_max_res(gray_image, 800), cv2.COLOR_GRAY2RGB)
    edges_rgb = cv2.cvtColor(resize_max_res(edges, 800), cv2.COLOR_GRAY2RGB)
    lines_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

     plot_images(
        [image_rgb, gray_rgb, edges_rgb, lines_rgb],  #
        ['原始圖像', '灰度圖', '邊緣檢測', '檢測到的直線']
    )
    """

    n = len(images)
    nrows = math.ceil(n / images_per_row)
    ncols = images_per_row

    plt.figure(figsize=figsize)
    for i in range(n):
        plt.subplot(nrows, ncols, i + 1)
        # check if it is a gray image
        if images[i].ndim == 2 or (images[i].shape[-1] != 3):
            cmap = 'gray'
        else:
            cmap = None
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])  # hide tick
    plt.tight_layout()

    if matplotlib.get_backend() != 'agg':
        plt.show()