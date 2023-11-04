import numpy as np
from pathlib import Path
from utils import *
import random
import cv2
import argparse
import os
import pickle
from ultralytics import YOLO
import warnings

warnings.filterwarnings("ignore")

# def point_in_polygon(point_xy: tuple[float, float], polygon_list: np.ndarray) -> bool:
#     """
#     判斷點是否在多邊形內，使用 0~1 座標系
#     :param point_xy: 座標 [0, 1]
#     :param polygon_list: 多邊形座標組=.
#     :return: boolean
#     """
#     x, y = point_xy
#     n = len(polygon_list)
#     inside = False
#
#     p1x, p1y = polygon_list[0]
#     for i in range(n + 1):
#         p2x, p2y = polygon_list[i % n]
#         if y > min(p1y, p2y):
#             if y <= max(p1y, p2y):
#                 if x <= max(p1x, p2x):
#                     if p1y != p2y:
#                         xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
#                         if p1x == p2x or x <= xinters:
#                             inside = not inside
#         p1x, p1y = p2x, p2y
#
#     return inside


# def load_table_N_to_data_struct(table_N_path: Path, verbose=False) -> list:
#     """
#     load table_N.pic to data struct
#     :param table_N_path: Path
#     :return: list
#     """
#     # 打開.txt文件以讀取模式
#     with open(f'{table_N_path}', 'rb') as file:
#         loaded_data = pickle.load(file)
#     # 載入的數據
#     if verbose:
#         print("Loaded Data:", np.array(loaded_data))
#     return loaded_data
#
# def load_polygon_to_data_struct(polygon_path: Path, verbose=False) -> list:
#     """
#     load polygon.pic to data struct
#     :param polygon_path: Path
#     :return: list
#     """
#     # 打開.txt文件以讀取模式
#     with open(f'{polygon_path}', 'rb') as file:
#         loaded_data = pickle.load(file)
#     # 載入的數據
#     if verbose:
#         print("Loaded Data:", np.array(loaded_data))
#     return loaded_data
#
#
# def load_pic_dot_chair():
#     """
#     load pickle 的資料，並 denormalization 後，畫在圖片上~
#     :return:
#     """
#     chair_list = []
#     # load pickle
#     with open(f'./datacase/case1/table_N_chair.pic', 'rb') as file:
#         chair_list = pickle.load(file)
#     # 載入的數據
#     print(chair_list)
#
#     # load image
#     image_root = Path(r"./datacase/case1/images")
#     assert image_root.exists(), f"image_root not exists \"{image_root}\""
#     image = list(image_root.glob("*.*"))[0]
#     image = cv2.imread(str(image))
#     image = resize_image_with_max_resolution(image, 800)
#
#     w, h = image.shape[1::-1]
#
#     de_normalize_chair_list = []
#     # de-normalize
#     for chairs in chair_list:
#         de_normalize_chair_list.append([(int(x * w), int(y * h)) for x, y in chairs])
#
#
#     # draw chair point on image
#     for chairs in de_normalize_chair_list:
#         for chair in chairs:
#             cv2.circle(image, chair, 2, (0, 0, 255), -1)
#     # show image
#     cv2.imshow('Image', image)
#     cv2.waitKey(0)


"""
command:
    # param
        <case_root>: 根目錄路徑名稱
        <verbose> : 
        <polygon_pic> : polygon.pic, default
        
    #
    
    python  tmp.py --case_root "C:\\cgit\\zjpj-lib-cam\\datacase\\case1" --verbose
    # 主範圍測試
        python  tmp.py --case_root "./datacase/case1" --mode "polygon" --verbose
    #
        python  tmp.py --case_root "./datacase/case1" --mode "table" --verbose 
"""


def command_main():
    parser = argparse.ArgumentParser(description="My Script")
    # folder path
    parser.add_argument("--case_root", type=str, help="cam folder root path")
    parser.add_argument("--verbose", action="store_true", help="show debug message")
    parser.add_argument("--mode", choices=['table', 'polygon', 'chair'], type=str, help="choose mode")
    parser.add_argument("--polygon_pic", default="polygon.pic", type=str, help="polygon txt path")
    parser.add_argument("--table_pic", default="table_N.pic", type=str, help="polygon txt path")
    parser.add_argument("--image_folder_name", default="images", type=str, help="image folder name")
    # 解析命令行參數
    args = parser.parse_args()

    #
    case_root = Path(args.case_root)  # case 根目錄
    assert case_root.exists(), f"case root not exists \"{case_root}\""
    poly_txt = case_root.joinpath(args.polygon_pic)  # polygon.txt
    assert poly_txt.exists(), f"polygon txt not exists \"{poly_txt}\""

    if args.mode == 'table':
        pic_load_name = case_root.joinpath(args.table_pic)
    elif args.mode == 'polygon':
        pic_load_name = case_root.joinpath(args.polygon_pic)
    elif args.mode == 'chair':
        print("this mode current is hard code mode!!")
        load_pic_dot_chair()  # current is hard code
        return
    # 打開.pic文件以讀取模式
    with open(f'{pic_load_name}', 'rb') as file:
        loaded_data = pickle.load(file)

    # debug
    if args.verbose:
        print("Loaded Data:", np.array(loaded_data))

    #
    w, h = 800, 600
    # random pick 1 set points xy, [0, 1]
    random_norm_xy_list = np.random.rand(1000, 2)

    # create image
    image = np.zeros((h, w, 3), np.uint8)

    # convert normal coordinate to image coordinate,
    loaded_data = np.array(loaded_data)
    # COPY one norm copy
    loaded_data_norm = loaded_data.copy()
    #
    if loaded_data.ndim == 2:
        loaded_data[:, 0] = loaded_data[:, 0] * w
        loaded_data[:, 1] = loaded_data[:, 1] * h
    elif loaded_data.ndim == 3:
        loaded_data[:, :, 0] = loaded_data[:, :, 0] * w
        loaded_data[:, :, 1] = loaded_data[:, :, 1] * h
    # convert to int
    # loaded_data 多邊形本人~
    loaded_data = loaded_data.astype(np.int32)

    polygons = None
    if loaded_data.ndim == 2:
        polygons = [
            np.array([loaded_data], dtype=np.int32),
        ]
    elif loaded_data.ndim == 3:
        polygons = []
        if 1:
            for polygon in loaded_data:
                polygons.append(np.array(polygon, dtype=np.int32))
        else:
            polygons = [
                np.array(loaded_data[0], dtype=np.int32),
            ]

    else:
        raise Exception(f"loaded_data.ndim == {loaded_data.ndim} ?? ")

    # draw polygon
    cv2.polylines(image, polygons, True, (0, 255, 255), 2)
    # draw random points
    green = (0, 255, 0)
    red = (0, 0, 255)

    loaded_data_norms = None
    if loaded_data_norm.ndim == 2:  # loaded_data_norm 本身就是多邊形的點
        loaded_data_norms = np.array([loaded_data_norm], dtype=np.float32)
    elif loaded_data_norm.ndim == 3:  # loaded_data_norm 本身就是多組，多邊形的點
        if 1:
            loaded_data_norms = loaded_data_norm
        else:
            # debug
            loaded_data_norm = loaded_data_norm[1]
            loaded_data_norms = np.array([loaded_data_norm], dtype=np.float32)

    is_green_points = []
    is_red_points = []
    for loaded_data_norm in loaded_data_norms:  # 走訪 polygon
        for xy in random_norm_xy_list:  # 走訪點
            # decided point by norm-coordinate system
            if point_in_polygon(xy, loaded_data_norm):
                is_green_points.append(xy)
            else:
                # cv2.circle(image, (int(xy[0] * w), int(xy[1] * h)), 2, red, -1)
                is_red_points.append(xy)
    # draw red points
    for xy in is_red_points:
        cv2.circle(image, (int(xy[0] * w), int(xy[1] * h)), 2, red, -1)
    # draw green points
    for xy in is_green_points:
        cv2.circle(image, (int(xy[0] * w), int(xy[1] * h)), 2, green, -1)
    cv2.imshow('Image', image)
    cv2.waitKey(0)


def test_load_pickle_and_show_polygon_range():
    tables = load_table_N_to_data_struct(Path("./datacase/case1/table_N.pic"))
    polygon = load_polygon_to_data_struct(Path("./datacase/case1/polygon.pic"))
    print(f"tables (len={len(tables)}): ", tables)
    print(f"polygon (len={len(polygon)}): ", polygon)

    # use above data struct to test point in polygon
    w, h = 800, 600

    # draw polygon on image
    image = np.zeros((h, w, 3), np.uint8)
    # convert normal coordinate to image coordinate,
    converted_polygon = [(int(x * w), int(y * h)) for x, y in polygon]
    #
    converted_tables = []
    for table in tables:
        converted_table = [(int(x * w), int(y * h)) for x, y in table]
        converted_tables.append(converted_table)
    # 框出 table
    for table in converted_tables:
        cv2.polylines(image, [np.array(table, dtype=np.int32)], True, (0, 255, 0), 2)

    # 框出偵測部位多邊形
    cv2.polylines(image, [np.array(converted_polygon, dtype=np.int32)], True, (0, 255, 255), 2)
    cv2.imshow('Image', image)
    cv2.waitKey(0)


def test_yolo():
    model = YOLO('yolov8x.pt')  # pretrained YOLOv8n model
    #
    image_root = Path(r"./datacase/case1/images")
    assert image_root.exists(), f"image_root not exists \"{image_root}\""
    image_list = list(image_root.glob("*.*"))
    # Run batched inference on a list of images
    image_list = [str(image_path) for image_path in image_list]
    results = model(image_list)

    for idx, result in enumerate(results):  # iterate results
        #
        img = result.orig_img
        # 此張圖片的所有偵測 bbox
        boxes = result.boxes.cpu().numpy()  # get boxes on cpu in numpy
        # 這個是字典注意， key 應該是 bbox 的 index
        # labels = result.names  # get labels
        # cla [N, 1], 每個框框的類別
        labels = [int(_) for _ in result.boxes.cls.detach().cpu()]
        # conf [N, 1], 每個框框的 confidence
        confs = result.boxes.conf.detach().cpu()

        for idx, box in enumerate(boxes):  # iterate boxes
            r = box.xyxy[0].astype(int)  # get corner points as int
            print(r)  # print boxes

            # text
            label_idx = labels[idx]
            # 繪製圖示
            ttext = f"{result.names[label_idx]} {confs[idx]:.2f}"
            print(ttext)
            font_scale = 2
            cv2.putText(img, ttext, r[:2], cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)

            #
            cv2.rectangle(img, r[:2], r[2:], (0, 0, 255), 2)  # draw boxes on img
        cv2.imshow(f"aaa {idx}", resize_image_with_max_resolution(img, 800))
    cv2.waitKey(0)
    print("bbbb")
    # return a list of Results objects


def line_avg_segnment_lab():
    # 創建一個帶有白色背景的空白影像
    height, width = 500, 500
    image = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 生成兩個隨機點
    point1 = (random.randint(0, width), random.randint(0, height))
    point2 = (random.randint(0, width), random.randint(0, height))

    divided_points = equally_divided_line_segments(point1, point2, 5, False)

    # 在影像上繪製點
    cv2.circle(image, point1, 5, (0, 0, 255), -1)
    cv2.circle(image, point2, 5, (0, 0, 255), -1)
    # 把端點會繪製上去
    for point in divided_points:
        print("draw divid points = ", point)
        cv2.circle(image, point, 5, (255, 0, 0), -1)

    # 繪製連接兩點的直線
    cv2.line(image, point1, point2, (255, 0, 0), 2)

    # 顯示帶有隨機點和直線的影像
    cv2.imshow('帶有隨機點和直線的影像', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def perspect_transform_test(parallelogram_norm=None, real_imgae=None, sits_number=8):
    # 創建一個800x600的灰色底圖像
    if real_imgae is None:
        width, height = 800//4, 600//4
        background = np.ones((height, width, 3), dtype=np.uint8) * 150  # 灰色背景
    else:
        background = real_imgae.copy()
        width, height = background.shape[1::-1]

    if parallelogram_norm is None:
        #  定義平行四邊形的四個點（以比例為單位）
        parallelogram_norm = np.array(
            [[0.25, 0.16666667],
             [0.75, 0.16666667],
             [0.875, 0.6666667],
             [0.375, 0.6666667]
             ], dtype=np.float64)  # 32
    # RE-order that
    parallelogram_norm = sort_poins_is_clockwise_and_leftTop_mostClose_leftTop(parallelogram_norm)
    parallelogram_pixel = normalize_2_pixel(parallelogram_norm, width, height).astype(np.int32)

    # 計算長邊與短邊的向量比
    # 長邊
    long_edge = parallelogram_pixel[1] - parallelogram_pixel[0]
    # 短邊
    short_edge = parallelogram_pixel[3] - parallelogram_pixel[0]
    # 長邊與短邊的向量比
    long_edge_to_short_edge_ratio = np.linalg.norm(long_edge) / np.linalg.norm(short_edge)
    assert long_edge_to_short_edge_ratio >= 1.0

    # debug parallelogram_pixel_Point_Order
    parallelogram_pixel_Point_Order = 0
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

    # 繪製平行四邊形
    cv2.polylines(background, [parallelogram_pixel], isClosed=True, color=(0, 0, 255),
                  thickness=2)  # 紅色邊

    num_points = 20
    # 在圖像上隨機生成點的座標並繪製
    for _ in range(num_points):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))  # 隨機顏色
        cv2.circle(background, (x, y), 2, color, -1)  # 繪製點

    # 保存圖像為文件
    # cv2.imwrite('parallel_quadrilateral.png', background)

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
        _debug_show_RAW_mapping = 0
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
            mapping_corners[i] = np.dot(M, np.array([x, y, 1.0], dtype=np.float64))[:2]
            print("int  mapping = ", mapping_corners[i][0], mapping_corners[i][1])
        for i, x_y in enumerate(_4_corners_in_raw_norm):
            x, y = x_y
            mapping_corners_norm[i] = np.dot(M_norm, np.array([x, y, 1.0], dtype=np.float64))[:2]
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
            print("超出最大允許值 {}，放棄位移投影作業!".format(MAX_ALLOW_EXCEED))
            print("x_min_exceed, x_max_exceed = ", x_min_exceed, x_max_exceed)
            print("y_min_exceed, y_max_exceed = ", y_min_exceed, y_max_exceed)
            pass
        elif x_exceed > 0 or y_exceed > 0:
            continue

        print("final good output_size = ", output_size)

        # 逆運算
        print("\n逆運算:")
        inv_M = np.linalg.inv(M)
        inv_M_norm = np.linalg.inv(M_norm)
        # 嘗試 用 逆運算的矩陣 回算回原始的點
        for i, x_y in enumerate(mapping_corners):
            x, y = x_y
            _x, _y = np.dot(inv_M, np.array([x, y, 1], dtype=np.float64))[:2]
            print("[int] remapping = {:-8.5f}   {:-8.5f}".format(round(_x), round(_y)))
            print("[int]    origin = {:-8.5f}   {:-8.5f}".format(round(_4_corners_in_raw[i][0]), round(_4_corners_in_raw[i][1])))
            print("=====================================================")
        # 驗證 OK! 沒問題

        # norm version
        for i, x_y in enumerate(mapping_corners_norm):
            x, y = x_y
            _x, _y = np.dot(inv_M_norm, np.array([x, y, 1], dtype=np.float64))[:2]
            print("[norm] remapping = {:-8.5f}   {:-8.5f}".format(_x, _y))
            print("[norm]    origin = {:-8.5f}   {:-8.5f}".format(_4_corners_in_raw_norm[i][0], _4_corners_in_raw_norm[i][1]))
            print("=====================================================")

        if real_imgae is not None:
            # 拿乾淨的原圖
            output = cv2.warpPerspective(real_imgae.copy(), M, output_size)
        else:
            # 拿假想環境的圖片用
            output = cv2.warpPerspective(background, M, output_size)

        # 繪製 mapping_corners 在仿射的圖圖片上
        for x, y in mapping_corners:
            cv2.circle(output, (int(x), int(y)), 6, (255, 255, 0), -1)
        break
    # this while-do end

    assert M is not None, "M is None"
    assert inv_M is not None, "inv_M is None"

    # 原圖的點?
    print("原圖的點:", parallelogram_pixel)
    # 繪製在 image1 上
    for xy in parallelogram_pixel:
        x, y = xy
        cv2.circle(background, (x, y), 6, (255, 154, 0), -1)
    # ============================================================
    # 校正過後的點
    print("校正過後的點:", end="")
    mapped_parallelogram_pixel = np.zeros_like(parallelogram_pixel)
    for i, xy in enumerate(parallelogram_pixel):
        x, y = xy
        _x, _y = np.dot(M, np.array([x, y, 1]))[:2]
        print(round(_x), round(_y))
        mapped_parallelogram_pixel[i] = np.array([_x, _y], dtype=np.int32)
    # 繪製在 output 上
    for xy in mapped_parallelogram_pixel:
        x, y = xy
        cv2.circle(output, (x, y), 10, (34, 255, 0), -1)

    # display
    cv2.imshow('Perspective result, Point on parallelogram', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #
    # 把四邊切分
    # point_set, 切分的線條組 [ [(x, y), (x, y)] ... ]
    point_set = divided_square_and_cals_slice_linesPair(mapped_parallelogram_pixel, sits_number)
    # ============================================================
    for p1, p2 in point_set:
        cv2.line(output, p1, p2, (255, 0, 0), 2)
    # ============================================================

    # 把線段們 inverse mapping
    inv_segment_set = []
    for p1, p2 in point_set:
        _p1 = np.dot(inv_M, np.array([p1[0], p1[1], 1]))[:2]
        _p2 = np.dot(inv_M, np.array([p2[0], p2[1], 1]))[:2]
        inv_segment_set.append([_p1, _p2])
    # 繪製 inverse mapping 後的線段們
    for p1, p2 in inv_segment_set:
        p1 = p1.astype(np.int32).tolist()
        p2 = p2.astype(np.int32).tolist()
        cv2.line(background, p1, p2, (0, 255, 255), 2)
    # ============================================================
    # 推算出所有定位點
    sit_and_table_loc_points = None
    # special test - 計算在 mapping 空間，使用四邊形的標準化座標 根據 output_size(debug 渲染用)
    # 顯示的結果 也會是正確的!!!
    _display_point_debug = 0
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
                                                                              assert_check=False)

    # 把 sit_and_table_loc_points_mapped 上的點逆運算回 background 上
    sit_norm_mapped = sit_and_table_loc_points_mapped_norm["sit_points"]
    table_norm_mapped = sit_and_table_loc_points_mapped_norm["table_points"]

    print("繪製回去的點 座標  (椅子):")
    # 把計算好的座位 印回去原本的圖形上
    for i, xy in enumerate(sit_norm_mapped):
        x, y = xy
        _origin_x_nrom, _origin_y_nrom = np.dot(inv_M_norm, np.array([x, y, 1]))[:2]
        re_mapping_pixel_x = _origin_x_nrom * width
        re_mapping_pixel_y = _origin_y_nrom * height
        cv2.circle(background, (int(re_mapping_pixel_x), int(re_mapping_pixel_y)), 3, (123, 111, 12), -1)
        print("   (x={}, y={})".format(int(re_mapping_pixel_x), int(re_mapping_pixel_y)), end=", ")
        if i % 4 == 0 and i != 0:
            print("")
    # 把計算好的桌子 印回去原本的圖形上
    print("\n繪製回去的點 座標  (桌子):")
    for i, xy in enumerate(table_norm_mapped):
        x, y = xy
        _origin_x_nrom, _origin_y_nrom = np.dot(inv_M_norm, np.array([x, y, 1]))[:2]
        re_mapping_pixel_x = _origin_x_nrom * width
        re_mapping_pixel_y = _origin_y_nrom * height
        cv2.circle(background, (int(re_mapping_pixel_x), int(re_mapping_pixel_y)), 3, (255, 111, 255), -1)
        print("   (x={}, y={})".format(int(re_mapping_pixel_x), int(re_mapping_pixel_y)), end=", ")
        if i % 4 == 0 and i != 0:
            print("")
    print("\n==============================")
    # ============================================================
    # TODO: 根據 {'sit_points': [], 'table_points': []}
    # "依序" 分出來，所有的框框 (桌子 椅子 分別)
    # 建構出以下資料結構 sit_table_binding = [ {'sit':[4個點], 'table':[四個點]}, ... ]

    # ============================================================
    #
    cv2.namedWindow('Image1', cv2.WINDOW_NORMAL)
    cv2.imshow("Image1", background)
    cv2.imshow("output", output)
    cv2.waitKey(300000)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # command_main()
    # test_load_pickle_and_show_polygon_range()
    # load_pic_dot_chair()
    # test_yolo()
    # 給予一的線段，印出他的平分點
    #line_avg_segnment_lab()
    #
    _use_ideal_debug = 1
    if _use_ideal_debug:
        perspect_transform_test()  # ideal test case
    else:
        real_image = cv2.imread(r"C:\cgit\zjpj-lib-cam\datacase\case1\2F_North\2023-10-23-193851.jpg")
        # real_image = resize_image_with_max_resolution(real_image, 100)
        parallelogram_norm = load_table_N_to_data_struct(Path("./datacase/case1/table_N.pic"))[0]
        perspect_transform_test(parallelogram_norm=parallelogram_norm, real_imgae=real_image, sits_number=8)

    pass
