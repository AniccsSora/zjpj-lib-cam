import numpy as np
from pathlib import Path
from utils import *
import random
import cv2
import argparse
import os
import pickle

#
image_copy_G = None
polygon_points_G = None
argas_G = None
draw_polygon_color_G = (0, 0, 255)
#
def draw_polygon(event, x, y, flags, param):
    global polygon_points_G, draw_polygon_color_G
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points_G.append((x, y))
        cv2.circle(image_copy_G, (x, y), 5, draw_polygon_color_G, -1)
        cv2.imshow('Image', image_copy_G)


def dot_polygon_range(image, mask_color_alpha=(255, 0, 0, 50))->list:
    global polygon_points_G, image_copy_G, argas_G
    res = []
    w, h = image.shape[1::-1]
    # global assigned
    image_copy_G = image.copy()
    polygon_points_G = []
    #
    # 創建一個空的遮罩圖像，與原始圖像大小相同，並設置透明度
    mask = np.zeros_like(image, dtype=np.uint8)

    # show images
    cv2.imshow('Image', image_copy_G)
    cv2.setMouseCallback('Image', draw_polygon)
    #
    print("請在圖片上點出要偵測的範圍，最後按 'c' 完成多邊形繪製")
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') or key == ord('C'):  # 按 'c' 完成多邊形
            break

    # 將多邊形點列表轉換為NumPy數組並進行正規化
    polygon_points_G = np.array(polygon_points_G, dtype=np.float32)
    if argas_G.verbose:
        print("polygon_points_G: ", polygon_points_G)
    polygon_points_normalization = [[float(x) / float(w), float(y) / float(h)] for x, y in polygon_points_G]
    if argas_G.verbose:
        print("polygon_points_G(normalization): ", polygon_points_normalization)
    # 使用多邊形點創建遮罩
    cv2.fillPoly(mask, [np.array([polygon_points_G], dtype=np.int32)], mask_color_alpha)

    # 將遮罩應用於原始圖像
    result = cv2.addWeighted(image, 1, mask, 0.7, 0)

    # 顯示結果圖像
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存多邊形點的正規化座標
    if argas_G.verbose:
        print("Normalized Polygon Points:")
        print(polygon_points_normalization)
    # 存檔為 pickle
    with open(case_root.joinpath('polygon.pic'), 'wb') as file:
        pickle.dump(polygon_points_normalization, file)

    # 保存結果圖像
    cv2.imwrite(case_root.joinpath('yolo_detection_range.jpg').__str__(), result)
    # 等待用戶按下任意按鍵關閉窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass


def dot_table_range(image, mask_color_alpha=(0, 255, 0, 50)):
    global polygon_points_G, image_copy_G, argas_G, draw_polygon_color_G
    draw_polygon_color_G = (0, 255, 0)
    polygon_points_normalization = None
    res = []
    w, h = image.shape[1::-1]
    # global assigned
    image_copy_G = image.copy()
    table_range_list = []
    polygon_points_G = []
    #
    # 創建一個空的遮罩圖像，與原始圖像大小相同，並設置透明度
    mask = np.zeros_like(image, dtype=np.uint8)

    print("1. 同一張桌子順著一個方向將其邊緣用點圍住(基本上點4角落即可), 按下 n 存下目前區塊。")
    print("2. 想框住的桌子都框好後框好後按下 'c' 將繪製最後圖形")

    region_cnt = 0
    while True:
        cv2.imshow('Image', image_copy_G)
        #
        cv2.setMouseCallback('Image', draw_polygon)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') or key == ord('C'):  # 按 'c' 完成多邊形
            break

        elif (key == ord('n') or key == ord('N')) and len(polygon_points_G) >= 4:  # 按 'n' 開始新的區塊
            region_cnt += 1
            print(f"區塊 {region_cnt} 完成, \'按下c\' 離開, 或者 \'按下n\'繼續框下一個區塊")

            # 將多邊形點列表轉換為NumPy數組並進行正規化
            polygon_points_G = np.array(polygon_points_G, dtype=np.float32)
            if argas_G.verbose:
                print("polygon_points_G: ", polygon_points_G)
            polygon_points_normalization = [[float(x) / float(w), float(y) / float(h)] for x, y in polygon_points_G]
            if argas_G.verbose:
                print("polygon_points_G(normalization): ", polygon_points_normalization)
            # 使用多邊形點創建遮罩
            cv2.fillPoly(mask, [np.array([polygon_points_G], dtype=np.int32)], mask_color_alpha)
            # 存檔下目前區塊
            table_range_list.append(polygon_points_normalization)
            # 清空多邊形點列表，為下一個區塊準備
            polygon_points_G = []
            polygon_points_normalization = []

    # 顯示結果圖像
    result = cv2.addWeighted(image, 1, mask, 0.7, 0)
    cv2.imshow('Result', result)
    print(f"共框選了 {len(table_range_list)} 個 桌子...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存多邊形點的正規化座標
    if argas_G.verbose:
        print("Normalized table Points:")
        print(table_range_list)
    # 存檔為 pickle
    with open(case_root.joinpath('table_N.pic'), 'wb') as file:
        pickle.dump(table_range_list, file)

    # 保存結果圖像
    cv2.imwrite(case_root.joinpath('table_detection_range.jpg').__str__(), result)

    # 等待用戶按下任意按鍵關閉窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()

"""
command:
    # param
        <case_root>
        <verbose> 
    #
    # 主偵測範圍設定
    python  step_1_case_layout_paint.py --case_root "C:\\cgit\\zjpj-lib-cam\\datacase\\case1" --mode='polygon'
    python  step_1_case_layout_paint.py --case_root "./datacase/case1" --mode="polygon" --verbose
    
    # table 範圍偵測設定
    python  step_1_case_layout_paint.py --case_root "./datacase/case1" --mode="table" --verbose
    
    
# 執行結果
    會在 <--case_root> 下長出 polygon.txt 和 valid_detection_range.jpg
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="My Script")
    # folder path
    parser.add_argument("--case_root", type=str, help="cam folder root path")
    parser.add_argument("--verbose", action="store_true", help="show debug message")
    parser.add_argument("--mode", choices=['table', 'polygon'], type=str, help="choose mode")
    parser.add_argument("--image_folder_name",  default="images", type=str, help="image folder name")
    # 解析命令行參數
    args = parser.parse_args()
    #
    argas_G = args
    _case_root = Path(args.case_root)
    process_mode = args.mode
    #
    if _case_root.is_absolute() is False:
        _case_root = Path(os.getcwd()).joinpath(_case_root)
    #
    case_root = _case_root
    assert case_root.exists(), f"case root not exists \"{case_root}\""

    # get image path, random pick one
    raw_camImage_folder = args.image_folder_name
    assert case_root.joinpath(raw_camImage_folder).exists(), "raw images folder does not exists"
    img_path = random.choice(list(case_root.joinpath(raw_camImage_folder).glob("*.jpg")))

    # read image
    image = cv2.imread(str(img_path))
    image = resize_image_with_max_resolution(image, 800)


    if process_mode == 'polygon':
        # 產出 polygon.pic and review-圖片
        print("*************************************")
        print("* 多邊形模式, 即選出 yolo 需辨識的區塊 *")
        print("*************************************")
        dot_polygon_range(image)
    elif process_mode == 'table':
        # 產出 table_X.pic and review-圖片_X
        print("***************")
        print("* 桌子框選模式 *")
        print("***************")
        dot_table_range(image)