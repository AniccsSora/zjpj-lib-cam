import cv2
import pickle
import numpy as np
from pathlib import Path


def calculate_polygon_center(points):
    if len(points) < 3:
        raise ValueError("多邊形至少需要3個點")

    # 檢查每個點的值域是否在 0 <= x <= 1
    for x, y in points:
        if not (0 <= x <= 1) or not (0 <= y <= 1):
            raise ValueError("點的值域必須在 0 到 1 之間")

    # 初始化中心座標的總和
    total_x, total_y = 0, 0

    # 獲得多邊形的邊數
    num_points = len(points)

    # 計算多邊形的面積
    area = 0
    for i in range(num_points):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % num_points]
        area += (x1 * y2 - x2 * y1)

    area *= 0.5

    # 計算多邊形的中心座標
    for i in range(num_points):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % num_points]
        factor = x1 * y2 - x2 * y1
        total_x += (x1 + x2) * factor
        total_y += (y1 + y2) * factor

    total_x /= (6 * area)
    total_y /= (6 * area)

    return total_x, total_y


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


def pixel_2_normalize(coordinates: np.ndarray, width, height):
    """
    將座標從像素座標轉換為 0~1 的座標系
    :param coordinates:  ndarray, shape (n, 2)
    :param width:
    :param height:
    :return:
    """
    if not isinstance(coordinates, np.ndarray) or coordinates.shape[1] != 2:
        raise ValueError("coordinates shape must like (n, 2)! n is points number.")
    x_norm, y_norm = coordinates[:, 0].copy(), coordinates[:, 1].copy()

    pixel_coordinates = None
    if coordinates.dtype == np.int32:
        x_pixel = (x_norm.astype(np.float32) / width)
        y_pixel = (y_norm.astype(np.float32) / height)
        pixel_coordinates = np.column_stack((x_pixel, y_pixel))
    else:
        raise ValueError(f"Unhandle type:{coordinates.dtype}, "
                         f"This is for [pixel (int)] -> [norm (float)],\nbut your input data is: \n{coordinates[:2]}...")

    if pixel_coordinates is None:
        raise ValueError("pixel_coordinates is None")
    return pixel_coordinates

def normalize_2_pixel(coordinates: np.ndarray, width, height):
    """
    將座標從 0~1 的座標系轉換為像素座標
    :param coordinates:  ndarray, shape (n, 2)
    :param width:
    :param height:
    :return:
    """
    if not isinstance(coordinates, np.ndarray) or coordinates.shape[1] != 2:
        raise ValueError("coordinates shape must like (n, 2)! n is points number.")
    x_norm, y_norm = coordinates[:, 0].copy(), coordinates[:, 1].copy()
    pixel_coordinates = None
    if (coordinates.dtype == np.float32) or \
            (coordinates.dtype == np.float64):
        x_pixel = (x_norm * width).astype(np.int32)
        y_pixel = (y_norm * height).astype(np.int32)
        pixel_coordinates = np.column_stack((x_pixel, y_pixel))
    else:
        raise ValueError(f"Unhandle type:{coordinates.dtype}, "
                         f"This is for [norm (float)] -> [pixel (int)],\nbut your input data is: \n{coordinates[:2]}...")
    if pixel_coordinates is None:
        raise ValueError("pixel_coordinates is None")
    return pixel_coordinates

def equally_divided_line_segments(point1: np.ndarray, point2: np.ndarray,
                                  num_segments: int, include_endp=True) -> np.ndarray:
    """
    給我兩的點，告訴我切幾段，回傳中間的點座標
    :param point1:
    :param point2:
    :param num_segments: 切幾段
    :param include_endp: 是否也回傳端點
    :return:
    """
    # load points as np.ndarray
    p1, p2 = np.array(point1, dtype=np.float64), np.array(point2, dtype=np.float64)
    N = num_segments  # 切幾段
    #===============================
    # check they just a points
    assert p1.ndim == 1 and p2.ndim == 1, f"p1.ndim = {p1.ndim}, p2.ndim = {p2.ndim}"
    # check size
    assert p1.size == 2 and p2.size == 2, f"p1.size = {p1.size}, p2.size = {p2.size}"

    line_vector = p2 - p1

    dividing_points = np.array([p1 + (i / N) * line_vector for i in range(1, N)])

    # round then
    dividing_points = np.round(dividing_points).astype(np.int32)

    #
    if include_endp:
        dividing_points = np.concatenate([np.array([p1], dtype=np.int32),
                                          dividing_points,
                                          np.array([p2], dtype=np.int32)],
                                         axis=0)
    return dividing_points

def divided_square_and_cals_slice_linesPair(square:np.ndarray, number:int):
    """
    :param square: 預期至少四邊形即可~
    :param number: 要接成幾個座位，因為是接桌子，故要用 2 的倍數
    :return:
    """
    assert number % 2 == 0
    assert square.shape[0] == 4  # 預期是四邊型
    # 長邊要切幾刀
    long_side_cut = (number//2) - 1
    # 短邊要切幾刀
    short_side_cut = 1

    # 兩個長邊
    long_side1 = equally_divided_line_segments(square[0], square[1], long_side_cut+1, include_endp=False)
    long_side2 = equally_divided_line_segments(square[2], square[3], long_side_cut+1, include_endp=False)
    long_side2 = long_side2[::-1]  # 反轉
    # 兩個短邊
    short_side1 = equally_divided_line_segments(square[1], square[2], short_side_cut+1, include_endp=False)
    short_side2 = equally_divided_line_segments(square[3], square[0], short_side_cut+1, include_endp=False)
    short_side2 = short_side2[::-1]  # 反轉
    # 合併
    #  #長邊點組
    merge_long_side_P_pair = [i for i in zip(long_side1, long_side2)]
    #  # 短邊點組
    merge_short_side_P_pair = [i for i in zip(short_side1, short_side2)]

    res = merge_long_side_P_pair + merge_short_side_P_pair
    if long_side_cut+short_side_cut != len(res):
        raise ValueError("長邊切的刀數({}) + 短邊切的刀數({}) != 總回傳線段數{}".format(
            long_side_cut, short_side_cut, len(res)
        ))
    return res

