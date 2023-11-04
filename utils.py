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
    if isinstance(coordinates, list):
        coordinates = np.array(coordinates, dtype=np.float64)

    if not isinstance(coordinates, np.ndarray) or coordinates.shape[1] != 2:
        raise ValueError("coordinates shape must like (n, 2)! n is points number.")
    x_norm, y_norm = coordinates[:, 0].copy(), coordinates[:, 1].copy()
    pixel_coordinates = None
    if (coordinates.dtype == np.float32) or \
            (coordinates.dtype == np.float64):
        x_pixel = (x_norm * width).astype(np.int64)
        y_pixel = (y_norm * height).astype(np.int64)
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


def calc_parallelogram_location_points(parallelogram:np.ndarray, sit_number:int, assert_check = True)->np.ndarray:
    """
    回傳，所有棋盤方格 "點位"(包括座位區，定位點，桌子(角落點, 棋盤點))
    :param parallelogram:
    :param sit_number: 整個桌面要有幾個座位
    :return: {"sit_points": [np.ndarray ...],
              "table_points": [np.ndarray ...]
            }
    """

    assert parallelogram.shape[0] == 4  # 預期是四邊型
    assert sit_number % 2 == 0 and sit_number >= 2
    assert parallelogram.dtype == np.float32 or parallelogram.dtype == np.float64  # 請傳入正歸化座標

    # 檢查 邊1, 與邊2 是否垂直?
    vector_a = np.array(parallelogram[1] - parallelogram[0])  # 替換 a1, a2, a3 為實際的數值
    vector_b = np.array(parallelogram[-1] - parallelogram[0])
    # 計算點積
    dot_product = np.dot(vector_a, vector_b)
    if assert_check:
        assert np.isclose(dot_product, 0, atol=0.001), "兩向量不垂直，無法計算,內積為:{}".format(dot_product)
    # ===================== pre-checker END =====================
    # horizon vector
    x_vector = vector_a
    if assert_check:
        assert np.isclose(x_vector[1], 0, atol=0.01), "水平向量 具有 y 分量, x_vector = {}".format(x_vector)
    # vertical vector
    y_vector = vector_b
    if assert_check:
        assert np.isclose(y_vector[0], 0, atol=0.01), "垂直向量 具有 x 分量, y_vector = {}".format(y_vector)

    # 水平位置一個座位的方向向量
    x_sit_unit_vector = x_vector.astype(np.float64) / (sit_number // 2)

    # 垂直位置一個座位的方向向量
    y_sit_unit_vector = y_vector.astype(np.float64) / 2

    # 檢測 單位向量 位移 數次後，可以走到 另一個桌子角落。
    # 左上原點 -> 右上角落
    lt = parallelogram[0]
    rt = parallelogram[1]
    _x_delta = x_sit_unit_vector * (sit_number//2)
    if assert_check:
        assert np.isclose(np.linalg.norm(lt + _x_delta - rt), 0), \
            "左上四邊形原點:{}，加上 x 方向向量，走 {} 步，無法到達右上角落:{}".\
            format(lt, sit_number//2, rt)
    # 左上原點 -> 左下角落
    # lt = parallelogram[0]
    lb = parallelogram[3]
    _y_delta = y_sit_unit_vector * 2   # 垂直永遠是 2 個單位，因為只切一刀
    if assert_check:
        assert np.isclose(np.linalg.norm(lt + _y_delta - lb), 0), \
            "左上四邊形原點:{}，加上 y 方向向量，走 {} 步，無法到達左下角落:{}".\
            format(lb, 2, lb)

    # build return dataform
    result_dict = {
        'sit_points': [],
        'table_points': [],
    }

    # 左上的座位點，位移
    sit_lf = lt - y_sit_unit_vector
    for i in range((sit_number//2)+1):
        result_dict['sit_points'].append(sit_lf + (x_sit_unit_vector*i))
    # 左上方座位點位移 N 的位置後會抵達，右上角落- y_sit_unit_vector 位置
    if assert_check:
        assert np.isclose(np.linalg.norm(result_dict['sit_points'][-1] + y_sit_unit_vector - rt), 0), \
            "  左上方座位點位移 N 的位置後會抵達，右上角落- y_sit_unit_vector 位置\n" \
            "  椅子右上方 = result_dict['sit_points'][-1]:{}\n" \
            "  桌子右上角落 = rt:{}\n" \
            "  y 單位向量(1分隔向量)\n".format(result_dict['sit_points'][-1], rt, y_sit_unit_vector)
    # 上排驗證完畢，直接走4個 y 位移量即可抵達下排座位
    bottom_sits = np.array(result_dict['sit_points']) + (y_sit_unit_vector * 4)
    result_dict['sit_points'] += list(bottom_sits)


    # 座位計算完畢，來整個桌子點
    for i in range((sit_number // 2) + 1):
        result_dict['table_points'].append(lt + (x_sit_unit_vector * i))
    # 下面兩排也算出來，整排位移 y_sit_unit_vector ，兩遍的點 + 近來
    _topper_table_points = np.array(result_dict['table_points'])
    for i in range(1, 2+1):
        _shift_table_points = _topper_table_points + (y_sit_unit_vector * i)
        result_dict['table_points'] += list(_shift_table_points)
    #
    _debug = 0
    if _debug:
        # debug section
        # 印出點位用來 debug
        # 建立一個 800, 600 空畫布
        debug_image = np.zeros((600, 800, 3), np.uint8)
        # 畫出四邊形
        for i in range(4):
            cv2.line(debug_image, tuple(parallelogram[i]), tuple(parallelogram[(i+1)%4]), (0, 255, 0), 1)
        # 畫出點位 [table_points]
        for i in range(len(result_dict['table_points'])):
            cv2.circle(debug_image, tuple(result_dict['table_points'][i].astype(np.int32)),
                       2, (255, 255, 0), -1)
        #
        # 繪製座位點
        for i in range(len(result_dict['sit_points'])):
            cv2.circle(debug_image, tuple(result_dict['sit_points'][i].astype(np.int32)),
                       2, (255, 0, 255), -1)
        cv2.imshow("debug_image", debug_image)
        cv2.waitKey(0)
        # debug section
    # ===================== debug section End=====================
    # 椅子
    assert len(result_dict['sit_points']) == ((sit_number//2)+1) * 2
    # 桌子
    assert len(result_dict['table_points']) == ((sit_number//2)+1) * 3
    # 檢查計算完畢的點位數量是對的。

    return result_dict


def sort_poins_is_clockwise_and_leftTop_mostClose_leftTop(points:np.ndarray):
    """
    檢查傳入的點排序是順時針，且最靠近左上角的點是第一個
    """
    if isinstance(points, list):
        points = np.array(points)

    assert isinstance(points, np.ndarray)
    original_dtype = points.dtype

    # 計算每個點到原點的歐幾里得距離
    distances = np.sqrt((points[:, 0] - 0) ** 2 + (points[:, 1] - 0) ** 2)

    # 找到最靠近原點的點的索引
    starting_index = np.argmin(distances)

    # 重新排列點的順序，使得最靠近原點的點是第一個
    points = np.roll(points, -starting_index, axis=0)

    # 計算相對於起始點的極角
    angles = np.arctan2(points[:, 1] - points[0, 1], points[:, 0] - points[0, 0])

    # 對角度進行排序（由於y往下遞增，angles本身就是順時針排序）
    sorted_indices = np.argsort(angles)

    # 排序點，使其順序是順時針方向
    sorted_points = points[sorted_indices]

    # 平行四邊形不能用這種方式確認
    # assert np.linalg.norm(points[0] - points[1]) > np.linalg.norm(points[0] - points[-1]), \
    # "順時針的第一個向量，大於 逆時針的第一個向量" \
    # "np.linalg.norm(points[0] - points[1]):{}\n" \
    # "np.linalg.norm(points[0] - points[-1]):{}\n" \
    # .format(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[0] - points[-1]))


    return sorted_points.astype(original_dtype)


