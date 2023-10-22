import cv2
import pickle
import numpy as np
from pathlib import Path


def point_in_polygon(point_xy: tuple[float, float], polygon_list: np.ndarray) -> bool:
    """
    判斷點是否在多邊形內，使用 0~1 座標系
    :param point_xy: 座標 [0, 1]
    :param polygon_list: 多邊形座標組=.
    :return: boolean
    """
    x, y = point_xy
    assert 0 <= x <= 1, f"x 座標必須在 0~1 之間，目前給定 x = {x}"
    assert 0 <= y <= 1, f"y 座標必須在 0~1 之間，目前給定 y = {y}"
    n = len(polygon_list)
    inside = False

    p1x, p1y = polygon_list[0]
    for i in range(n + 1):
        p2x, p2y = polygon_list[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
        p1x, p1y = p2x, p2y

    return inside

def load_table_N_to_data_struct(table_N_path: Path, verbose=False) -> list:
    """
    load table_N.pic to data struct
    :param table_N_path: Path
    :return: list
    """
    # 打開.txt文件以讀取模式
    with open(f'{table_N_path}', 'rb') as file:
        loaded_data = pickle.load(file)
    # 載入的數據
    if verbose:
        print("Loaded Data:", np.array(loaded_data))
    return loaded_data

def load_polygon_to_data_struct(polygon_path: Path, verbose=False) -> list:
    """
    load polygon.pic to data struct
    :param polygon_path: Path
    :return: list
    """
    # 打開.txt文件以讀取模式
    with open(f'{polygon_path}', 'rb') as file:
        loaded_data = pickle.load(file)
    # 載入的數據
    if verbose:
        print("Loaded Data:", np.array(loaded_data))
    return loaded_data


def load_pic_dot_chair(chair_N_path: Path, verbose=False) -> list:
    """
    load pickle 的資料，並 denormalization 後，畫在圖片上~
    :return:
    """
    chair_list = []
    # load pickle
    with open(chair_N_path, 'rb') as file:
        chair_list = pickle.load(file)
    # 載入的數據
    # print(chair_list)

    # load image
    image_root = chair_N_path.parent.joinpath("images")  # images
    # assert image_root.exists(), f"image_root not exists \"{image_root}\""
    # image = list(image_root.glob("*.*"))[0]
    # image = cv2.imread(str(image))
    # image = resize_image_with_max_resolution(image, 800)
    #
    # w, h = image.shape[1::-1]
    #
    # de_normalize_chair_list = []
    # # de-normalize
    # for chairs in chair_list:
    #     de_normalize_chair_list.append([(int(x * w), int(y * h)) for x, y in chairs])
    #
    #
    # # draw chair point on image
    # for chairs in de_normalize_chair_list:
    #     for chair in chairs:
    #         cv2.circle(image, chair, 2, (0, 0, 255), -1)
    # # show image
    # cv2.imshow('Image', image)
    # cv2.waitKey(0)
    return chair_list


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