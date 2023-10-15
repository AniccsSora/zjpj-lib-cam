import cv2
import pickle
import numpy as np
from pathlib import Path

def resize_image_with_max_resolution(image, max_resolution):
    """
    調整圖片的大小，以限制最大一邊的解析度，並保持長寬比例。

    :param image: 輸入圖片（NumPy 數組）
    :param max_resolution: 最大解析度（一個整數，表示最大一邊的像素數）
    :return: 調整大小後的圖片
    """
    # 獲取原始圖片的寬度和高度
    original_height, original_width = image.shape[:2]

    # 確定調整大小的目標寬度和高度
    if original_width > original_height:
        # 如果寬度大於高度，將寬度調整為最大解析度，並保持長寬比例
        new_width = max_resolution
        new_height = int(original_height * (max_resolution / original_width))
    else:
        # 如果高度大於寬度，將高度調整為最大解析度，並保持長寬比例
        new_height = max_resolution
        new_width = int(original_width * (max_resolution / original_height))

    # 使用 OpenCV 的 resize 函數進行調整大小
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    return resized_image