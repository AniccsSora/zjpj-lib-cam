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


def point_in_polygon(point_xy: tuple[float, float], polygon_list: np.ndarray) -> bool:
    """
    判斷點是否在多邊形內，使用 0~1 座標系
    :param point_xy: 座標 [0, 1]
    :param polygon_list: 多邊形座標組=.
    :return: boolean
    """
    x, y = point_xy
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


"""
command:
    # param
        <case_root>: 根目錄路徑名稱
        <verbose> : 
        <polygon_pic> : polygon.pic, default

    #

    python  tmp.py --case_root "C:\\cgit\\zjpj-lib-cam\\datacase\\case1" --verbose
    # 主範圍測試
        python  dot_cmd.py --case_root "./datacase/case1" --mode "polygon" --verbose
    #
        python  dot_cmd.py --case_root "./datacase/case1" --mode "table" --verbose 
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My Script")
    # folder path
    parser.add_argument("--case_root", type=str, help="cam folder root path")
    parser.add_argument("--verbose", action="store_true", help="show debug message")
    parser.add_argument("--mode", choices=['table', 'polygon'], type=str, help="choose mode")
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
    # 打開.txt文件以讀取模式
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