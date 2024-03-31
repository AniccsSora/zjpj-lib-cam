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
    print("3. 按下 'q' 強制結束")

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
        else:
            if key == ord('q') or key == ord('Q'):
                print("強制結束!!")
                exit(0)
    # 顯示結果圖像
    result = cv2.addWeighted(image, 1, mask, 0.7, 0)
    cv2.imshow('Result', result)
    print(f"共框選了 {len(table_range_list)} 個 桌子...")
    print("按下空白存檔...")
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
    print("顯示預覽範圍 按下空白按鍵關閉視窗...")
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
    print("預覽已經保存按 下空白按鍵關閉視窗...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pass

chair_tmp_G = []
chairs_list_G = []
dot_chair_callback_img_G = None
dot_chair_callback_img_name_G = None
def dot_chair_callback(event, x, y, flags, param):
    global chair_tmp_G, dot_chair_callback_img_G
    if event == cv2.EVENT_LBUTTONDOWN:
        chair_tmp_G.append([x, y])
        cv2.circle(dot_chair_callback_img_G, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow(dot_chair_callback_img_name_G, dot_chair_callback_img_G)

def dot_chair_range(image, pic_file_name, mask_color_alpha=(0, 255, 0, 50)):
    global argas_G, chair_tmp_G, chairs_list_G, dot_chair_callback_img_G, dot_chair_callback_img_name_G
    #
    image = resize_image_with_max_resolution(image, 800)
    image_clean = image.copy()

    # 打開.pic文件以讀取模式
    with open(f'{pic_file_name}', 'rb') as file:
        loaded_data = pickle.load(file)
    print("table nums =", len(loaded_data))

    # de-normalization
    w, h = image.shape[1::-1]
    loaded_data = np.array(loaded_data)
    if loaded_data.ndim == 3:
        # 把每個點都 de-normalization
        loaded_data[:, :, 0] = loaded_data[:, :, 0] * w
        loaded_data[:, :, 1] = loaded_data[:, :, 1] * h
        # convert to int
        loaded_data = loaded_data.astype(np.int32)
    else:
        assert 0

    for idx, table_dots in enumerate(loaded_data):
        print("目前處理第", idx+1, "張桌子，左鍵點選可能是座位的 '點', 按下C 換下一張桌子")
        #
        table_window_name = f"Table {idx+1}"

        # draw a table polygon points
        # 這個是正規化的點
        table_dots = np.array(table_dots, dtype=np.int32)
        polygons = [
            table_dots
        ]
        mask = np.zeros_like(image, dtype=np.uint8)
        # sufu
        cv2.fillPoly(mask, [np.array(polygons, dtype=np.int32)], mask_color_alpha)
        result = cv2.addWeighted(image, 1, mask, 0.7, 0)

        # add callback
        dot_chair_callback_img_G = result
        dot_chair_callback_img_name_G = table_window_name
        if argas_G.verbose:
            print("table_dots: ", polygons)
        # show image
        cv2.imshow(table_window_name, result)
        cv2.setMouseCallback(table_window_name, dot_chair_callback)  # sufu
        print(f"請對著 highlight 的桌子周圍把 '椅子' 點出來, 按下 'c' 當前桌子(桌子{idx+1})...")
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') or key == ord('C'):  # 按 'c' 完成多邊形
                chairs_list_G.append(chair_tmp_G)
                chair_tmp_G = []
                break

        # destory all windows
        cv2.destroyAllWindows()

    for_save_img = image_clean.copy()
    for table_chairs in chairs_list_G:
        for chair in table_chairs:
            cv2.circle(for_save_img, tuple(chair), 5, (255, 0, 0), -1)
    cv2.imwrite(case_root.joinpath('chair_detection_range.jpg').__str__(), for_save_img)

    return chairs_list_G
"""
command:
    # param
        <case_root>
        <verbose> 
    #
    # 主偵測範圍設定 [deprecation command] vvvvvvvvvvvvvvvvvvv
    #python  step_1_case_layout_paint.py --case_root "C:\\cgit\\zjpj-lib-cam\\datacase\\case1" --mode='polygon'
    python  step_1_case_layout_paint.py --case_root "./datacase/case1" --mode="polygon" --verbose
    python  step_1_case_layout_paint.py --case_root "./datacase/case2" --mode="polygon" --image_folder_name="B1F_south" --verbose
    #### [deprecation command]  ^^^^^^^^^^^^^^^^^
    
    # table 範圍偵測設定
    python  step_1_case_layout_paint.py --case_root "./datacase/case1" --mode="table" --verbose
    python  step_1_case_layout_paint.py --case_root "./datacase/case1" --mode="table" --image_folder_name="2F_North" --verbose
    #
    python  step_1_case_layout_paint.py --case_root "./datacase/case2" --mode="table" --image_folder_name="B1F_south" --verbose
    
    # chair 
    python  step_1_case_layout_paint.py --case_root "./datacase/case1" --mode="chair" --verbose
    python  step_1_case_layout_paint.py --case_root "./datacase/case2" --mode="chair" --image_folder_name="B1F_south"
# 執行結果
    會在 <--case_root> 下長出 polygon.txt 和 valid_detection_range.jpg
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="My Script")
    # folder path
    parser.add_argument("--case_root", type=str, help="cam folder root path")
    parser.add_argument("--verbose", action="store_true", help="show debug message")
    parser.add_argument("--mode", choices=['table', 'polygon', 'chair'], type=str, help="choose mode")
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
    w, h = image.shape[1::-1]

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
    elif process_mode == 'chair':
        # 產出 table_X.pic and review-圖片_X
        print("***************")
        print("* 椅子點選模式 *")
        print("***************")
        table_pickle = 'table_N.pic'  # mode='table' 產出的 table_N.pic
        assert case_root.joinpath(table_pickle).exists(), "please do mode='table' first, and ensure table_N.pic exists"
        res = dot_chair_range(image, pic_file_name=case_root.joinpath(table_pickle))

        normalized_chair_list = []
        # normalization
        for _table in res:
            _ = np.array(_table, dtype=np.float32) / np.array([w, h], dtype=np.float32)
            _ = _.tolist()
            normalized_chair_list.append(_)
        with open(case_root.joinpath('table_N_chair.pic'), 'wb') as file:
            pickle.dump(normalized_chair_list, file)
        print("存檔完畢!!")
