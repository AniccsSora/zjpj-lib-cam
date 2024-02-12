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


if __name__ == "__main__":
    # command_main()
    # test_load_pickle_and_show_polygon_range()
    # load_pic_dot_chair()
    # test_yolo()
    # 給予一的線段，印出他的平分點
    #line_avg_segnment_lab()
    #
    _use_ideal_debug = 0
    if _use_ideal_debug:
        clac_sit_table_fields_dict(debug_mode=True)  # ideal test case
    else:
        real_image = cv2.imread(r"C:\cgit\zjpj-lib-cam\datacase\case1\2F_North\2023-10-23-195003.jpg")
        assert real_image is not None, "real_image is None"
        real_image = resize_image_with_max_resolution(real_image, 800)
        parallelogram_norm = load_table_N_to_data_struct(Path("./datacase/case1/table_N.pic"))[0]
        # 計算座位的 binding list
        mapping_back_binding = clac_sit_table_fields_dict(parallelogram_norm=parallelogram_norm,
                                                          real_imgae=real_image,
                                                          sits_number=8,
                                                          debug_mode=True)
        # 把 binding list 畫在圖片上
        draw_binding_list_on_image(real_image, mapping_back_binding,
                                   color_whole_sit_table=(255, 255, 255), whole_thickness=8,
                                   color_table=(0, 255, 0), table_thickness=2,
                                   color_sit=(255, 0, 0), sit_thickness=2, verbose=False
                                   )
        cv2.imshow("real_image", real_image)
        cv2.waitKey(0)
    pass
