import cv2
import pickle
import numpy as np
from pathlib import Path
import random
import copy



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

def centroid(vertices):
    x, y = 0, 0
    n = len(vertices)
    signed_area = 0
    for i in range(len(vertices)):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        # shoelace formula
        area = (x0 * y1) - (x1 * y0)
        signed_area += area
        x += (x0 + x1) * area
        y += (y0 + y1) * area
    signed_area *= 0.5
    x /= 6 * signed_area
    y /= 6 * signed_area
    return x, y

def draw_norm_polygon_on_image(image: np.ndarray, polygon: np.ndarray, color, thickness, inward=True):
    polygon = polygon.copy()
    center = None
    if 0:
        # 端點內縮
        center = calculate_polygon_center(polygon)
        # add (point -> center vector) to each point
        for i in range(len(polygon)):
            if 0:
                polygon[i] += (center - polygon[i]) * 0.06
            else:
                _p2c_vec = np.linalg.norm(center - polygon[i])
                polygon[i] += (center - polygon[i]) * (0.06 )
    #
    polygon[:, 0] *= image.shape[1]
    polygon[:, 1] *= image.shape[0]

    if inward:
        center = np.array(centroid(polygon))
        for i in range(len(polygon)):
            vec_len = np.linalg.norm(center - polygon[i])
            ratio = thickness/vec_len
            polygon[i] += (center - polygon[i]) * (ratio)*1.05 # 越>1 框

    cv2.polylines(image, [polygon.astype(np.int32)], True, color, thickness)


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
        'x_unit_vector': x_sit_unit_vector,
        'y_unit_vector': y_sit_unit_vector,
        'sit_group_points': [],  # 每個 '橫排'or'列' 的座位點, 基本上座位就是兩排
        'table_group_points': [],  # 每個 '橫排'or'列' 的桌子點, 基本上桌子就是三排
    }

    # 左上的座位點，位移
    sit_lf = lt - y_sit_unit_vector
    _sit_group_points_tmp = []
    for i in range((sit_number//2)+1):
        result_dict['sit_points'].append(sit_lf + (x_sit_unit_vector*i))
        _sit_group_points_tmp.append(sit_lf + (x_sit_unit_vector*i))
    result_dict['sit_group_points'].append(_sit_group_points_tmp)
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
    result_dict['sit_group_points'].append(list(bottom_sits.copy()))

    # 座位計算完畢，來整個桌子點
    _table_group_points_tmp = []
    for i in range((sit_number // 2) + 1):
        result_dict['table_points'].append(lt + (x_sit_unit_vector * i))
        _table_group_points_tmp.append(lt + (x_sit_unit_vector * i))
    result_dict['table_group_points'].append(_table_group_points_tmp)
    #
    # 下面兩排也算出來，整排位移 y_sit_unit_vector ，兩遍的點 + 近來
    _topper_table_points = np.array(result_dict['table_points'])
    for i in range(1, 2+1):
        _shift_table_points = _topper_table_points + (y_sit_unit_vector * i)
        result_dict['table_points'] += list(_shift_table_points)
        result_dict['table_group_points'].append(list(_shift_table_points.copy()))
    #
    _debug = 0
    if _debug:
        _debug_w, _debug_h = 800, 600  # # 建立一個 800, 600 空畫布
        _debug_parallelogram = np.zeros(parallelogram.shape, np.int32)
        if parallelogram.dtype == np.float32 or\
                parallelogram.dtype == np.float64:
            _debug_parallelogram[:, 0] = parallelogram[:, 0] * _debug_w
            _debug_parallelogram[:, 1] = parallelogram[:, 1] * _debug_h

        # debug section
        # 印出點位用來 debug
        debug_image = np.zeros((_debug_h, _debug_w, 3), np.uint8)
        # 畫出四邊形
        for i in range(4):
            cv2.line(debug_image, tuple(_debug_parallelogram[i]), tuple(_debug_parallelogram[(i+1)%4]), (0, 255, 0), 1)
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
        cv2.destroyAllWindows()
        # debug section
    # ===================== debug section End=====================
    # 椅子
    assert len(result_dict['sit_points']) == ((sit_number//2)+1) * 2
    # 桌子
    assert len(result_dict['table_points']) == ((sit_number//2)+1) * 3
    # 檢查計算完畢的點位數量是對的。

    # transform to np.ndarray
    result_dict['sit_points'] = np.array(result_dict['sit_points'])
    result_dict['table_points'] = np.array(result_dict['table_points'])
    result_dict['sit_group_points'] = np.array(result_dict['sit_group_points'])
    result_dict['table_group_points'] = np.array(result_dict['table_group_points'])


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


def long_side_always_first(points:np.ndarray)->np.ndarray:
    if isinstance(points, list):
        points = np.array(points)
    if isinstance(points, np.ndarray):
        pass
    else:
        raise ValueError("points must be list or np.ndarray, but input type:{}".format(type(points)))

    origin_sort_points = points.copy()
    assert isinstance(origin_sort_points, np.ndarray)
    assert isinstance(points, np.ndarray)

    assert_long_vector = np.linalg.norm(points[0] - points[1])
    assert_short_vector = np.linalg.norm(points[0] - points[-1])
    if assert_long_vector < assert_short_vector:
        # swap
        points[1], points[-1] = origin_sort_points[-1], origin_sort_points[1]
    return points


def make_sit_table_binding_to_field_binding(loc_points: dict, field_debug_show=False) -> list:
    """
    用定位點 來計算 椅子與桌子的綁定關係
    :param loc_points:
    :param field_debug_show:
    :return:
    """

    # 小小建構子
    def New_my_dict():
        return {'sit_field_region': [], 'table_field_region': [], 'whole_sit_region': []}
    res = []
    #
    _sit_field_debug = field_debug_show  # debug
    if _sit_field_debug:
        _debug_w, _debug_h = 800, 600
        _debug_image = np.zeros((_debug_h, _debug_w, 3), np.uint8)

    _table_field_debug = field_debug_show  # debug
    if _table_field_debug:
        _debug_w, _debug_h = 800, 600
        _debug_image = np.zeros((_debug_h, _debug_w, 3), np.uint8)

    _bind_whole_sit_debug = field_debug_show  # debug
    if _bind_whole_sit_debug:
        _debug3_w, _debug3_h = 800, 600
        _debug_image3 = np.zeros((_debug3_h, _debug3_w, 3), np.uint8)

    for sit_group in loc_points['sit_group_points']:
        for table_group in loc_points['table_group_points']:
            for _y_unit_v in [1*loc_points['y_unit_vector'], -1*loc_points['y_unit_vector']]:
                if np.isclose(sit_group + _y_unit_v, table_group).all():
                    # 當整排的椅子， shift 上 or 下 一個 y-axis unit_vector 可以重和 桌子的時候
                    #
                    # 開始編織座位區域
                    # 座位點(sit_group) 與 桌子點(table_group) 與向量對應
                    st_binding = [_ for _ in zip(sit_group, table_group)]
                    for idx in range(len(st_binding)-1):
                        # 編號: 按照 順/逆 時方向
                        # 椅子起點, 左值都是點
                        sit_1 = st_binding[idx][0]  # [當前][椅子]
                        sit_2 = st_binding[idx+1][0]  # [下一個][椅子]
                        # 桌子，注意這得是 順時方向故
                        table_3 = st_binding[idx+1][1]  # [下一個][桌子]
                        # 檢查
                        assert np.isclose(table_3, sit_2+_y_unit_v).all(), \
                            "   椅子到桌子的跳躍點異常\n" \
                            "   椅子起點:{} -> 椅子終點:{}, 向量: {}\n" \
                            "   failed here >> \"下個桌子越點3 :{} != {}:(椅子2 加 y-unit)\"\n" \
                            "   Y 單位向量: {}\n". \
                            format(sit_1, sit_2, sit_2-sit_1, table_3, sit_2+_y_unit_v,_y_unit_v)
                        #
                        table_4 = st_binding[idx][1]  # [當前][桌子]
                        # 檢查
                        assert np.isclose(table_4, sit_1+_y_unit_v).all(), \
                            "   椅子到桌子的跳躍點異常\n" \
                            "   椅子1起點:{} -> 桌子4終點:{}, 向量: {}\n" \
                            "   failed here >> \"當前桌子越點4 :{} != {}:(椅子1 加 y-unit)\"\n" \
                            "   Y 單位向量: {}\n". \
                            format(sit_1, table_4, table_4 - sit_1, table_4, sit_1 + _y_unit_v, _y_unit_v)

                        #sit_field_region_tmp.append(np.array([sit_1, sit_2, table_3, table_4]))
                        # 畫出來
                        if _sit_field_debug:
                            print(np.array([sit_1, sit_2, table_3, table_4]))
                            _polygen = np.array([sit_1, sit_2, table_3, table_4])
                            _polygen[:, 0] = _polygen[:, 0] * _debug_w
                            _polygen[:, 1] = _polygen[:, 1] * _debug_h
                            _polygen = _polygen.astype(np.int32)
                            _debug_image = np.zeros((_debug_h, _debug_w, 3), np.uint8)
                            cv2.polylines(_debug_image, [_polygen.astype(np.int32)], True, (255, 255, 255), 1)
                            cv2.putText(_debug_image, str(1), (_polygen[0][0], _polygen[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 125, 255), 1)
                            cv2.putText(_debug_image, str(2), (_polygen[1][0], _polygen[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 125, 255), 1)
                            cv2.putText(_debug_image, str(3), (_polygen[2][0], _polygen[2][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 125, 255), 1)
                            cv2.putText(_debug_image, str(4), (_polygen[3][0], _polygen[3][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 125, 255), 1)
                            cv2.imshow("[debug] Sit field, single.", _debug_image)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                        #
                        # =============================================
                        # 接著 順著此  _y_unit_v  bind 過去 桌子 field
                        # 上面的假設是對的這邊就不需要檢查。
                        _table_1 = st_binding[idx][1]  # [當前][桌子]
                        _table_2 = st_binding[idx+1][1]  # [下一個][桌子]
                        _table_3 = st_binding[idx+1][1] +  _y_unit_v # [下一個][桌子] + _y_unit_v
                        _table_4 = st_binding[idx][1] + _y_unit_v  # [當前][桌子] + _y_unit_v

                        tmp_dict = New_my_dict()
                        tmp_dict['sit_field_region'] = np.array([sit_1, sit_2, table_3, table_4])
                        tmp_dict['table_field_region'] = np.array([_table_1, _table_2, _table_3, _table_4])

                        # 畫出來
                        if _table_field_debug:
                            _polygen = np.array([_table_1, _table_2, _table_3, _table_4])
                            _polygen[:, 0] = _polygen[:, 0] * _debug_w
                            _polygen[:, 1] = _polygen[:, 1] * _debug_h
                            _polygen = _polygen.astype(np.int32)
                            _debug_image = np.zeros((_debug_h, _debug_w, 3), np.uint8)
                            cv2.polylines(_debug_image, [_polygen.astype(np.int32)], True, (0, 255, 255), 1)
                            cv2.putText(_debug_image, str(1), (_polygen[0][0], _polygen[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
                            cv2.putText(_debug_image, str(2), (_polygen[1][0], _polygen[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
                            cv2.putText(_debug_image, str(3), (_polygen[2][0], _polygen[2][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
                            cv2.putText(_debug_image, str(4), (_polygen[3][0], _polygen[3][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)
                            cv2.imshow("[debug] Table field, single.", _debug_image)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()
                        # for idx in range(len(st_binding)-1): === END ===


                        # =============================================
                        #
                        # 繪製整個 binding 的區域
                        _whole_sit_1 = st_binding[idx][0]  # [當前][椅子]
                        _whole_sit_2 = st_binding[idx + 1][0]  # [下一個][椅子]
                        _whole_table_3 = st_binding[idx + 1][0] + _y_unit_v * 2  # [下一個][椅子] + _y_unit_v
                        _whole_table_4 = st_binding[idx][0] + _y_unit_v * 2  # [當前][椅子] + _y_unit_v

                        tmp_dict['whole_sit_region'] = np.array([_whole_sit_1, _whole_sit_2, _whole_table_3, _whole_table_4])
                        # 畫出來
                        if _bind_whole_sit_debug:
                            _polygen = np.array([_whole_sit_1, _whole_sit_2, _whole_table_3, _whole_table_4])
                            _polygen[:, 0] = _polygen[:, 0] * _debug3_w
                            _polygen[:, 1] = _polygen[:, 1] * _debug3_h
                            _polygen = _polygen.astype(np.int32)
                            _debug_image3 = np.zeros((_debug3_h, _debug3_w, 3), np.uint8)
                            cv2.polylines(_debug_image3, [_polygen.astype(np.int32)], True, (0, 0, 255), 1)
                            cv2.putText(_debug_image3, str(1), (_polygen[0][0], _polygen[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 123), 1)
                            cv2.putText(_debug_image3, str(2), (_polygen[1][0], _polygen[1][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 123), 1)
                            cv2.putText(_debug_image3, str(3), (_polygen[2][0], _polygen[2][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 123), 1)
                            cv2.putText(_debug_image3, str(4), (_polygen[3][0], _polygen[3][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 123), 1)
                            cv2.imshow("[debug] Whole sit field, single.", _debug_image3)
                            cv2.waitKey(0)
                            cv2.destroyAllWindows()

                        #
                        res.append(tmp_dict)
                    # if np.isclose(sit_group + _y_unit_v, table_group).all(): # === END ===
                    break
    return res



def draw_binding_list_on_image(image: np.array,
                               bindlist: list,
                               color_whole_sit_table=(255, 255, 255), whole_thickness=8,
                               color_table=(0, 255, 0), table_thickness=2,
                                color_sit=(255, 0, 0), sit_thickness=2, verbose=False):
    """
    把 dict 形式的 binding list 繪製到 image 上。

    1. 繪製整個 座位+椅子  範圍
    2. 椅子 範圍
    3. 桌子 範圍
    """
    # ============================================================
    # # draw each field on  field_output
    field_output = image
    # 所有 binding 座位座位
    for idx, field in enumerate(bindlist):
        #
        # sit_field_region
        # table_field_region
        # whole_sit_region
        #
        _debug_w, _debug_h = field_output.shape[1], field_output.shape[0]
        # 繪製桌子
        table_field = np.array(field['whole_sit_region'])
        # de-normalize
        table_field[:, 0] *= _debug_w
        table_field[:, 1] *= _debug_h

        # 把 backmapping 的 pixel 點投射回去原圖。
        cv2.polylines(field_output, [table_field.astype(np.int32)], isClosed=True, color=color_whole_sit_table,
                      thickness=whole_thickness)
        #
        if verbose:
            print("   位置 {} 的 整個桌子座標 Done!".format(idx + 1))
        # ============================================================
        # ============================================================
        # 繪製椅子
        sits_field = np.array(field['sit_field_region'])
        # de-normalize
        sits_field[:, 0] *= _debug_w
        sits_field[:, 1] *= _debug_h

        # 把 backmapping 的 pixel 點投射回去原圖。
        cv2.polylines(field_output, [sits_field.astype(np.int32)], isClosed=True, color=color_sit,
                      thickness=sit_thickness)
        if verbose:
            print("   位置 {} 的 椅子座標 Done!".format(idx + 1))

        ##############################################################
        # 繪製桌子
        table_field = np.array(field['table_field_region'])
        # de-normalize
        table_field[:, 0] *= _debug_w
        table_field[:, 1] *= _debug_h

        # 把 backmapping 的 pixel 點投射回去原圖。
        cv2.polylines(field_output, [table_field.astype(np.int32)], isClosed=True, color=color_table,
                      thickness=table_thickness)
        if verbose:
            print("   位置 {} 的 桌子座標 Done!".format(idx + 1))
        # ============================================================
            print("座位 {} 的繪製 Done!".format(idx + 1))
    # ============================================================


def calc_tramsform_matrix_result(points=np.ndarray, M=np.ndarray, verbose=False):
    assert isinstance(points, np.ndarray)
    assert isinstance(M, np.ndarray)

    # ============================================================
    assert points.shape[1] == 2  # 座標
    assert M.shape == (3, 3)  # 轉換矩陣

    # ============================================================
    # 進行轉換
    result = np.zeros_like(points)

    for i, xy in enumerate(points):
        x, y = xy
        _x, _y, w = M.dot(np.array([x, y, 1.0], dtype=np.float64))
        _x /= w
        _y /= w
        result[i] = np.array([_x, _y])

    return result


def clac_sit_table_fields_dict(parallelogram_norm=None, real_imgae=None, sits_number=8,
                               debug_mode=False):
    """
    計算傳入的 "正歸化平行四邊形座標" 回傳 切分後的 座位、桌子 的區塊。
    :param parallelogram_norm:  正規化平行四邊形座標
    :param real_imgae:  計算的原圖
    :param sits_number:  要切的座位數量
    :param debug_mode: 印出一堆中間步驟
    :return:
    """
    #
    #
    # 創建一個800x600的灰色底圖像
    if real_imgae is None:
        width, height = 800//4, 600//4
        background = np.ones((height, width, 3), dtype=np.uint8) * 150  # 灰色背景
    else:
        background = real_imgae.copy()
        width, height = background.shape[1::-1]
    clean_background = background.copy()

    if parallelogram_norm is None:
        #  定義平行四邊形的四個點（以比例為單位）
        parallelogram_norm = np.array(
            [[0.25, 0.16666667],
             [0.75, 0.16666667+0.13],  # +0.13  y
             [0.875, 0.6666667+0.13],  # +0.13  y
             [0.375-0.2, 0.6666667],  #   x
             ], dtype=np.float64)  # 32
    # RE-order that
    #parallelogram_norm = sort_poins_is_clockwise_and_leftTop_mostClose_leftTop(parallelogram_norm)
    #
    # side[0]-side[1]_always_most_long
    # 確認 [0]-[1] 邊總是最長，否則調換 [-1] 和 [1]
    _long_side_first = 1
    if _long_side_first:
        parallelogram_norm = long_side_always_first(parallelogram_norm)
    # ==========================================
    parallelogram_pixel = normalize_2_pixel(parallelogram_norm, width, height).astype(np.int32)

    # debug parallelogram_pixel_Point_Order
    parallelogram_pixel_Point_Order = debug_mode
    if parallelogram_pixel_Point_Order and real_imgae is not None:
        _order_parallelogram_pixel = np.zeros((real_imgae.shape[0], real_imgae.shape[1], 3), dtype=np.uint8)
        # _order_parallelogram_pixel.fill(255)  # 白色背景

        # 繪製點
        for i, point in enumerate(parallelogram_pixel):
            cv2.circle(_order_parallelogram_pixel, tuple(point), 5, (0, 0, 255), -1)
            # 計算原點，到此 point 的距離
            _norm_dis = ": {:.4f}".format(np.linalg.norm(np.array(parallelogram_norm[i]) - np.array([0, 0])))
            cv2.putText(_order_parallelogram_pixel, str(i + 1)+_norm_dis, (point[0] + 10, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            # 繪製 0,0 到 point 的直線
            cv2.line(_order_parallelogram_pixel, (0, 0), tuple(point), (0, 255, 0), 2)
        # 顯示圖像
        alpha = 0.5
        beta = 1.0
        gamma = 0
        blended = cv2.addWeighted(background, alpha, _order_parallelogram_pixel, beta, gamma)
        cv2.imshow('[Debug] Points with Order', blended)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    #   parallelogram_pixel_Point_Order ======== END ========

    # 計算長邊與短邊的向量比
    # 長邊
    long_edge = parallelogram_pixel[1] - parallelogram_pixel[0]
    # 短邊
    short_edge = parallelogram_pixel[3] - parallelogram_pixel[0]
    # 長邊與短邊的向量比
    long_edge_to_short_edge_ratio = np.linalg.norm(long_edge) / np.linalg.norm(short_edge)
    assert long_edge_to_short_edge_ratio >= 1.0

    # 繪製平行四邊形
    cv2.polylines(background, [parallelogram_pixel], isClosed=True, color=(0, 0, 255),
                  thickness=2)  # 紅色邊

    num_points = 20
    # 在圖像上隨機生成點的座標並繪製
    for _ in range(num_points):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # 隨機顏色
        if debug_mode:
            cv2.circle(background, (x, y), 2, color, -1)  # 繪製點

    # 保存圖像為文件
    # cv2.imwrite('parallel_quadrilateral.png', background)

    if debug_mode:
        # 顯示圖像（可選）
        cv2.imshow('Original table range', background)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # ==========================================
    # mapping 域大小
    output_size = [800, 600]  # [width, height]
    #
    # 計算透視變換矩陣
    #
    norm_unit_MAX = 0.5
    dst_points_norm = np.array(
        [[0.25, 0.25],
         [0.75, 0.25],
         [0.75, 0.25 + (norm_unit_MAX/ long_edge_to_short_edge_ratio)],
         [0.25, 0.25 + (norm_unit_MAX/ long_edge_to_short_edge_ratio)],
         ],
        dtype=np.float64  # 32
    )
    dst_points = normalize_2_pixel(dst_points_norm.copy(), output_size[0], output_size[1])
    # move right and bottom a little bits
    #
    # !!! Key point !!!
    #
    #dst_points[:, 0] += 200
    #dst_points[:, 1] += 200
    #
    M = None
    inv_M = None

    # 調整投應過後的圖形，讓他整個出現在畫面中的方法，
    # 其實就是找到投過去後的 原圖的四個角是否在畫面內即可。
    while True:
        # 推算額外位移用
        x_exceed = 0.0
        y_exceed = 0.0
        #
        src_points_norm = parallelogram_norm.copy()
        src_points = normalize_2_pixel(parallelogram_norm, width, height)
        # int 版本的
        M = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points.astype(np.float32))
        # 用 正歸化版本的
        M_norm = cv2.getPerspectiveTransform(src_points_norm.astype(np.float32), dst_points_norm.astype(np.float32))

        # 直接 show mapping 後的圖片
        _debug_show_RAW_mapping = debug_mode
        if _debug_show_RAW_mapping:
            debug_output = cv2.warpPerspective(background, M, output_size)
            cv2.imshow('[debug] RAW Perspective result', debug_output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # 原始圖像中的四個邊角點
        _4_corners_in_raw = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1],
            ]
        , dtype=np.float32)
        _4_corners_in_raw_norm = _4_corners_in_raw.copy().astype(np.float64)
        _4_corners_in_raw_norm[:, 0] /= width
        _4_corners_in_raw_norm[:, 1] /= height

        # 計算投射後的點，這是整個圖片的邊，不是桌子的邊
        mapping_corners = np.zeros_like(_4_corners_in_raw, dtype=np.float64)
        mapping_corners_norm = np.zeros_like(_4_corners_in_raw_norm, dtype=np.float64)
        for i, x_y in enumerate(_4_corners_in_raw.astype(np.float64)):
            x, y = x_y
            # mapping_corners[i] = np.matmul(np.array([x, y, 1.0], dtype=np.float64), M)[:2]
            # mapping_corners[i] = np.dot(M, np.array([x, y, 1.0], dtype=np.float64))[:2]  # 錯的
            _x, _y, w = M.dot(np.array([x, y, 1.0], dtype=np.float64))
            mapping_corners[i] = np.array([_x, _y]) / w
            if debug_mode:
                print("int  mapping = ", mapping_corners[i][0], mapping_corners[i][1])
        for i, x_y in enumerate(_4_corners_in_raw_norm):
            x, y = x_y
            #mapping_corners_norm[i] = np.matmul(np.array([x, y, 1.0],  dtype=np.float64), M_norm)[:2]
            # mapping_corners_norm[i] = np.dot(M_norm, np.array([x, y, 1.0], dtype=np.float64))[:2]  # 錯的
            _x, _y, w = M_norm.dot(np.array([x, y, 1.0], dtype=np.float64))
            mapping_corners_norm[i] = np.array([_x, _y]) / w
            if debug_mode:
                print("norm mapping = ", mapping_corners_norm[i][0], mapping_corners_norm[i][1])
        # 檢查有否 會超出去 投影區域的點
        #extra_edge = 10  # 限制投影區域內縮一圈
        # 限制投影區域內縮一圈, 因應等等會延伸座位區(y軸)，故計算桌子一半的高度來當作額外的邊界
        extra_edge = np.linalg.norm(_4_corners_in_raw[0]-_4_corners_in_raw[-1]).astype(np.int32)//2+10
        # 檢查
        # 假設mapping_corners和其他相關變量已經被定義
        x_min_exceed = 0
        x_max_exceed = 0
        y_min_exceed = 0
        y_max_exceed = 0
        #
        if 0:
            for x, y in mapping_corners:
                if x < 0+extra_edge :  # 左邊超出，位移投射最終點
                    x_exceed = abs(x-extra_edge)
                    dst_points[:, 0] += round(x_exceed)
                if x > output_size[0]-extra_edge:  # 右邊超出，加長畫布 output_size[0]
                    x_exceed = abs(x-(output_size[0]-extra_edge))
                    output_size[0] += round(x_exceed)
                if y < 0+extra_edge:  # 上面超出，位移投射最終點
                    y_exceed = abs(y-extra_edge)
                    dst_points[:, 1] += round(y_exceed)
                if y > output_size[1]-extra_edge:  # 下面超出，加長畫布 output_size[1]
                    y_exceed = abs(y-(output_size[1]-extra_edge))
                    output_size[1] += round(y_exceed)

        #
        # 收集超出的值
        for x, y in mapping_corners:
            if x < 0 + extra_edge:
                x_min_exceed = max(x_min_exceed, abs(x - extra_edge))
            if x > output_size[0] - extra_edge:
                x_max_exceed = max(x_max_exceed, abs(x - (output_size[0] - extra_edge)))
            if y < 0 + extra_edge:
                y_min_exceed = max(y_min_exceed, abs(y - extra_edge))
            if y > output_size[1] - extra_edge:
                y_max_exceed = max(y_max_exceed, abs(y - (output_size[1] - extra_edge)))

        # x_exceed, y_exceed = round(x_exceed), round(y_exceed)
        # if x_exceed > 0:
        #     print(f"    需要額外位移目標投影 x_exceed = {x_exceed}")
        # if y_exceed > 0:
        #     print(f"    需要額外位移目標投影 y_exceed = {y_exceed}")
        x_min_exceed, x_max_exceed = round(x_min_exceed), round(x_max_exceed)
        y_min_exceed, y_max_exceed = round(y_min_exceed), round(y_max_exceed)

        MAX_ALLOW_EXCEED = 500
        if x_min_exceed > MAX_ALLOW_EXCEED or \
            x_max_exceed > MAX_ALLOW_EXCEED or \
            y_min_exceed > MAX_ALLOW_EXCEED or \
            y_max_exceed > MAX_ALLOW_EXCEED:
            if debug_mode:
                print("超出最大允許值 {}，放棄位移投影作業!".format(MAX_ALLOW_EXCEED))
                print("x_min_exceed, x_max_exceed = ", x_min_exceed, x_max_exceed)
                print("y_min_exceed, y_max_exceed = ", y_min_exceed, y_max_exceed)
        elif x_exceed > 0 or y_exceed > 0:
            continue

        if debug_mode:
            print("final good output_size = ", output_size)

        # 逆運算
        if debug_mode:
            print("\n逆運算:")
        inv_M = np.linalg.inv(M)
        inv_M_norm = np.linalg.inv(M_norm)
        # 嘗試 用 逆運算的矩陣 回算回原始的點
        for i, x_y in enumerate(mapping_corners):
            x, y = x_y
            # _x, _y = np.dot(inv_M, np.array([x, y, 1], dtype=np.float64))[:2]  # 錯的
            #_x, _y = np.matmul(np.array([x, y, 1.0], dtype=np.float64), inv_M)[:2]
            _x, _y, w = inv_M.dot(np.array([x, y, 1.0], dtype=np.float64))
            _x /= w
            _y /= w
            if debug_mode:
                print("[int] remapping = {:-8.5f}   {:-8.5f}".format(round(_x), round(_y)))
                print("[int]    origin = {:-8.5f}   {:-8.5f}".format(round(_4_corners_in_raw[i][0]), round(_4_corners_in_raw[i][1])))
                print("=====================================================")
        # 驗證 OK! 沒問題

        # norm version
        for i, x_y in enumerate(mapping_corners_norm):
            x, y = x_y
            # _x, _y = np.dot(inv_M_norm, np.array([x, y, 1], dtype=np.float64))[:2]  # 錯的
            #_x, _y = np.matmul(np.array([x, y, 1.0], dtype=np.float64), inv_M_norm)[:2]
            _x, _y, w = inv_M_norm.dot(np.array([x, y, 1.0], dtype=np.float64))
            _x /= w
            _y /= w
            if debug_mode:
                print("[norm] remapping = {:-8.5f}   {:-8.5f}".format(_x, _y))
                print("[norm]    origin = {:-8.5f}   {:-8.5f}".format(_4_corners_in_raw_norm[i][0], _4_corners_in_raw_norm[i][1]))
                print("=====================================================")

        if real_imgae is not None:
            # 拿乾淨的原圖
            output = cv2.warpPerspective(real_imgae.copy(), M, output_size)
        else:
            # 拿假想環境的圖片用
            output = cv2.warpPerspective(background, M, output_size)

        clean_output = output.copy()

        # 繪製 mapping_corners 在仿射的圖圖片上
        for x, y in mapping_corners:
            cv2.circle(output, (int(x), int(y)), 6, (255, 255, 0), -1)
            pass
        break
    # this while-do end

    assert M is not None, "M is None"
    assert inv_M is not None, "inv_M is None"

    # 原圖的點?
    if debug_mode:
        print("原圖的點:", parallelogram_pixel)
    # 繪製在 image1 上
    for xy in parallelogram_pixel:
        x, y = xy
        if debug_mode:
            cv2.circle(background, (x, y), 6, (255, 154, 0), -1)
    # ============================================================
    # 校正過後的點
    if debug_mode:
        print("校正過後的點:", end="")
    mapped_parallelogram_pixel = np.zeros_like(parallelogram_pixel)
    for i, xy in enumerate(parallelogram_pixel):
        x, y = xy
        # _x, _y = np.dot(M, np.array([x, y, 1]))[:2]  # 錯的
        #_x, _y = np.matmul(np.array([x, y, 1.0], dtype=np.float64), M)[:2]  # 錯的2號
        _x, _y, w = M.dot(np.array([x, y, 1.0], dtype=np.float64))
        _x /= w
        _y /= w
        mapped_parallelogram_pixel[i] = np.array([_x, _y], dtype=np.int32)
    # 繪製在 output 上
    for xy in mapped_parallelogram_pixel:
        x, y = xy
        cv2.circle(output, (x, y), 10, (34, 255, 0), -1)
    # 把他們連起來
    for i in range(len(mapped_parallelogram_pixel)):
        p1 = mapped_parallelogram_pixel[i]
        p2 = mapped_parallelogram_pixel[(i+1) % len(mapped_parallelogram_pixel)]
        cv2.line(output, tuple(p1), tuple(p2), (0, 0, 255), 2)

    # display
    if debug_mode:
        cv2.imshow('Perspective result, Point on parallelogram', output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    #
    # 把四邊切分
    # point_set, 切分的線條組 [ [(x, y), (x, y)] ... ]
    point_set = divided_square_and_cals_slice_linesPair(mapped_parallelogram_pixel, sits_number)
    # ============================================================
    for p1, p2 in point_set:
        if debug_mode:
            cv2.line(output, p1, p2, (255, 0, 0), 2)
    # ============================================================

    # 把線段們 inverse mapping
    inv_segment_set = []
    for p1, p2 in point_set:
        #_p1 = np.dot(inv_M, np.array([p1[0], p1[1], 1]))[:2]  # 錯的
        #_p2 = np.dot(inv_M, np.array([p2[0], p2[1], 1]))[:2]  # 錯的
        # _p1 = np.matmul(np.array([p1[0], p1[1], 1.0], dtype=np.float64), inv_M)[:2]
        # _p2 = np.matmul(np.array([p2[0], p2[1], 1.0], dtype=np.float64), inv_M)[:2]
        _x1, _y1, w1 = inv_M.dot(np.array([p1[0], p1[1], 1.0], dtype=np.float64))
        _x2, _y2, w2 = inv_M.dot(np.array([p2[0], p2[1], 1.0], dtype=np.float64))
        _x1, _y1 = _x1 / w1, _y1 / w1
        _x2, _y2 = _x2 / w2, _y2 / w2
        #
        _p1 = np.array([_x1, _y1])
        _p2 = np.array([_x2, _y2])
        inv_segment_set.append([_p1, _p2])
    # 繪製 inverse mapping 後的線段們
    for p1, p2 in inv_segment_set:
        p1 = p1.astype(np.int32).tolist()
        p2 = p2.astype(np.int32).tolist()
        if debug_mode:
            cv2.line(background, p1, p2, (0, 255, 255), 2)
    # ============================================================
    # 推算出所有定位點
    sit_and_table_loc_points = None
    # special test - 計算在 mapping 空間，使用四邊形的標準化座標 根據 output_size(debug 渲染用)
    # 顯示的結果 也會是正確的!!!
    _display_point_debug = debug_mode
    if _display_point_debug:
        __sp_mapping_para_norm = np.zeros(mapped_parallelogram_pixel.shape, dtype=np.float64)
        __sp_mapping_para_norm[:, 0] = mapped_parallelogram_pixel[:, 0] / output_size[0]
        __sp_mapping_para_norm[:, 1] = mapped_parallelogram_pixel[:, 1] / output_size[1]
        _ = calc_parallelogram_location_points(__sp_mapping_para_norm, sits_number)

        # 確認，就算對 calc_parallelogram_location_points 使用標準化座標，他也可以計算正確。
        _sit_loc_points = np.array(_['sit_points']).copy()
        _table_loc_points = np.array(_['table_points']).copy()
        # 轉換回 pixel 座標
        _sit_loc_points[:, 0] *= output_size[0]
        _sit_loc_points[:, 1] *= output_size[1]
        _table_loc_points[:, 0] *= output_size[0]
        _table_loc_points[:, 1] *= output_size[1]
        # 轉換回 pixel 座標印出來 debug
        for xy in _sit_loc_points:
            x, y = xy
            cv2.circle(output, (int(x), int(y)), 3, (123, 255, 255), -1)
        for xy in _table_loc_points:
            x, y = xy
            cv2.circle(output, (int(x), int(y)), 3, (255, 123, 155), -1)
    # DEBUG === _check_norm_mapping_still_good END ===
    # dict = {'sit_points': [], 'table_points': []}
    mapped_parallelogram_norm = np.zeros(mapped_parallelogram_pixel.shape, dtype=np.float64)
    mapped_parallelogram_norm[:, 0] = mapped_parallelogram_pixel[:, 0] / output_size[0]
    mapped_parallelogram_norm[:, 1] = mapped_parallelogram_pixel[:, 1] / output_size[1]
    sit_and_table_loc_points_mapped_norm = calc_parallelogram_location_points(mapped_parallelogram_norm, sits_number,
                                                                              assert_check=True)

    # 把 sit_and_table_loc_points_mapped 上的點逆運算回 background 上
    sit_norm_mapped = sit_and_table_loc_points_mapped_norm["sit_points"]
    table_norm_mapped = sit_and_table_loc_points_mapped_norm["table_points"]

    if debug_mode:
        print("繪製回去的點 座標  (椅子):")
    # 把計算好的座位 印回去原本的圖形上
    for i, xy in enumerate(sit_norm_mapped):
        x, y = xy
        #_origin_x_nrom, _origin_y_nrom = np.dot(inv_M_norm, np.array([x, y, 1]))[:2]  # 錯的
        #_origin_x_nrom, _origin_y_nrom = np.matmul(np.array([x, y, 1.0], dtype=np.float64), inv_M_norm)[:2]
        _origin_x_nrom, _origin_y_nrom, w = inv_M_norm.dot(np.array([x, y, 1.0], dtype=np.float64))
        _origin_x_nrom /= w
        _origin_y_nrom /= w
        #
        re_mapping_pixel_x = _origin_x_nrom * width
        re_mapping_pixel_y = _origin_y_nrom * height
        if debug_mode:
            cv2.circle(background, (int(re_mapping_pixel_x), int(re_mapping_pixel_y)), 5, (123, 111, 12), -1)
            print("   (x={}, y={})".format(int(re_mapping_pixel_x), int(re_mapping_pixel_y)), end=", ")
            if i % 4 == 0 and i != 0:
                print("")
    # 把計算好的桌子 印回去原本的圖形上
    if debug_mode:
        print("\n繪製回去的點 座標  (桌子):")
    for i, xy in enumerate(table_norm_mapped):
        x, y = xy
        # _origin_x_nrom, _origin_y_nrom = np.dot(inv_M_norm, np.array([x, y, 1]))[:2]  # 錯的
        #_origin_x_nrom, _origin_y_nrom = np.matmul(np.array([x, y, 1.0], dtype=np.float64), inv_M_norm)[:2]
        _origin_x_nrom, _origin_y_nrom, w = inv_M_norm.dot(np.array([x, y, 1.0], dtype=np.float64))
        _origin_x_nrom /= w
        _origin_y_nrom /= w
        #
        re_mapping_pixel_x = _origin_x_nrom * width
        re_mapping_pixel_y = _origin_y_nrom * height
        if debug_mode:
            # 繪製座位點
            cv2.circle(background, (int(re_mapping_pixel_x), int(re_mapping_pixel_y)), 3, (255, 111, 255), -1)
            print("   (x={}, y={})".format(int(re_mapping_pixel_x), int(re_mapping_pixel_y)), end=", ")
            if i % 4 == 0 and i != 0:
                print("")
    if debug_mode:
        print("\n==============================")
    # ============================================================
    # 把椅子點 連起來　繪製在 output 上
    _draw_mapped_chair_lines_on_output = debug_mode
    if _draw_mapped_chair_lines_on_output:
        # 前半段 順序，後半段逆序，再拚起來!
        _for_draw_sit_points = np.zeros_like(sit_norm_mapped)
        _for_draw_sit_points[0:len(sit_norm_mapped)//2] = sit_norm_mapped[0:len(sit_norm_mapped)//2].copy()
        _for_draw_sit_points[len(sit_norm_mapped) // 2:] = sit_norm_mapped[-1:len(sit_norm_mapped)//2-1:-1].copy()

        for i in range(len(_for_draw_sit_points)):
            p1 = (_for_draw_sit_points[i] * np.array(output_size)).astype(np.int32)
            p2 = ((_for_draw_sit_points[(i+1) % len(_for_draw_sit_points)]) * np.array(output_size)).astype(np.int32)
            # 椅子連線
            cv2.line(output, tuple(p1), tuple(p2), (123, 111, 12), 2)

    # ============================================================
    # "依序" 分出來，所有的框框 (桌子 椅子 分別)
    # 建構出以下資料結構 sit_table_binding = [ {'sit':[4個點], 'table':[四個點]}, ... ]
    # sit_and_table_loc_points_mapped_norm...
    field_binding = make_sit_table_binding_to_field_binding(sit_and_table_loc_points_mapped_norm,
                                                            field_debug_show=debug_mode)
    assert len(field_binding) == sits_number, "len(field_binding) != sits_number, {} != {}" \
        .format(len(field_binding), sits_number)

    # ============================================================
    # # draw each field on  field_output
    field_output = clean_background.copy()

    # ============================================================
    # field_binding 這個是 在 mapping 空間的結果
    # 複製一個深層副本
    mapping_back_nrom_field_binding = copy.deepcopy(field_binding)

    # re-mapping normlized points to original norm points
    for idx in range(len(mapping_back_nrom_field_binding)):
        # 別名取代
        _SIT = mapping_back_nrom_field_binding[idx]['sit_field_region'].copy()
        _TALBE = mapping_back_nrom_field_binding[idx]['table_field_region'].copy()
        _WHOLE = mapping_back_nrom_field_binding[idx]['whole_sit_region'].copy()
        #
        mapping_back_nrom_field_binding[idx]['sit_field_region'] = \
            calc_tramsform_matrix_result(_SIT, inv_M_norm)
        mapping_back_nrom_field_binding[idx]['table_field_region'] = \
            calc_tramsform_matrix_result(_TALBE, inv_M_norm)
        mapping_back_nrom_field_binding[idx]['whole_sit_region'] = \
            calc_tramsform_matrix_result(_WHOLE, inv_M_norm)
    # ============================================================

    # 把 binding list 的框框畫出來
    if debug_mode:
        draw_binding_list_on_image(field_output, mapping_back_nrom_field_binding)

    if debug_mode:
        cv2.imshow("Image1", background)
        cv2.imshow("output", output)
        cv2.imshow("field_output", field_output)
        cv2.waitKey(300000)
        cv2.destroyAllWindows()

    #return field_binding, M, inv_M, mapping_back_nrom_field_binding
    return mapping_back_nrom_field_binding

