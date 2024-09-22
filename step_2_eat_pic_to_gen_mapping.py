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
import math
import glob
from typing import List, Tuple
import time
from web_fetch_DONT_UPLOAD_GITHUB import lib_camera_generator
from web_fetch_DONT_UPLOAD_GITHUB import mkv_frame_generator, mkv_frame_generator_sec
from utilsv2.state_machine.StateMachine import TB_StateMachine
from utilsv2.state_machine.StateMachine import State as TB_State
from utilsv2.my_queue.qqueue import Queue as My_Queue
import flask_web_app as myweb_app

def develope_mode():
    return False
    #
    # print("\n == 開發 Debug == ")
    # return True

args_G = None

class Secne_Table_chair:
    def __init__(self, case_root, debug_mode=False):
        self.args = args_G
        self.debug_mode = debug_mode
        self.polygen_points = None  # polygen_points
        self.table_points = None  # table_points
        self.chair_points = None  # chair_points
        self.state_machine_init = False
        self.state_history_init = False
        self._path_init(case_root)
        self.yolo_debug_bboxes_class_p = []
        assert self.polygen_points is not None
        assert self.table_points is not None
        assert self.chair_points is not None
        #
        # 在判斷人的 bbox 時候，我們針對他的 bbox 寬做一個擴張，讓他的 bbox 寬變大一點，盡可能去佔住椅子
        self.person_extra_occupy = 0.3
        #
        self.case_root = case_root
        #
        print(f"Use {self.args.yolo_w_name} weight!")
        self.model = YOLO(self.args.yolo_w_name)  # pretrained YOLOv8n model , yolov8l, yolov8x
        # v8 labels definition
        # person, handbag, cup, laptop, mouse, cell phone, book
        # TODO: 根據使用的 model 記得修改
        #self.used_v8_labels = [0, 26, 41, 63, 64, 67, 73] # normal
        # self.used_v8_labels = [0,1,2,3,4,5,6,7,8,9]  # self training 拿一堆  class but 太難標記
        self.used_v8_labels = [0, 1]  # 只有 Object:0, Person:1
        """
        self.v8_labels =
        {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
         6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
         11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
         16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
         21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
         26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
         31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
         36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
         41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
         46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
         51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
         56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
         61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
         66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
         71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
         76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
        """
        # init talbe save fields
        # 有幾個人在 table N 內
        self.person_in_table_N = {}
        for table_id in range(self.table_numbers()):
            self.person_in_table_N[table_id] = []
        #
        # init chairs saveing fields
        self.person_in_chair_N = {}
        for table_id in range(self.table_numbers()):
            self.person_in_chair_N[table_id] = [0 for _ in range(self.chairs_numbers_in_table_N(table_id))]
        #
        #

    def is_point_in_polygon(self, point):
        if len(point) == 4:
            # xyxy, cls center point
            point = [((point[0] + point[2])/2), (point[1]+point[3])/2]
        elif len(point) == 2:
            point = (point[0], point[1])
        else:
            raise ValueError(f"point length = {len(point)}")
            return False
        #
        if point[0] > 1 or point[1] > 1:
            point[0] /= self.yolo_detect_w
            point[1] /= self.yolo_detect_h
        #
        return point_in_polygon(point, self.polygen_points)

    def is_point_in_table_N(self, table_id, point):
        table_id = self._table_idx_check(table_id)
        #
        if len(point) == 4:
            # xyxy, cls center point
            point = [((point[0] + point[2])/2), (point[1]+point[3])/2]
        elif len(point) == 2:
            point = (point[0], point[1])
        else:
            raise ValueError(f"point length = {len(point)}")
            return False

        # 我希望他們都是正規化的座標並存下，這個是for 未來的淺藏錯誤
        assert point[0] != 1 or point[1] != 1, f"future debug point = {point}"
        #
        if point[0] > 1 or point[1] > 1:
            point[0] /= self.yolo_detect_w
            point[1] /= self.yolo_detect_h
        #
        return point_in_polygon(point, self.table_points[table_id])

    def yolo_detect(self, image, verbose=False):
        image_list = [image]
        assert len(image_list) == 1
        results = self.model(image_list)
        # 偵測時使用的大小
        self.yolo_detect_w, self.yolo_detect_h = image.shape[1::-1]
        #
        self.current_yolo_results = []

        for idx, result in enumerate(results):  # iterate results
            #
            img = result.orig_img
            # 此張圖片的所有偵測 bbox
            boxes = result.boxes.cpu().numpy()  # get boxes on cpu in numpy

            # 這個是字典注意， key 應該是 bbox 的 index
            # labels = result.names  # get labels
            # if 0 in result.boxes.cls.detach().cpu():
            #     print("person in this frame")

            # cla [N, 1], 每個框框的類別
            labels = [int(_) for _ in result.boxes.cls.detach().cpu()]
            # conf [N, 1], 每個框框的 confidence
            confs = result.boxes.conf.detach().cpu()
            # mask 用來過濾不要的與要的類別
            used_mask = np.isin(labels, self.used_v8_labels)

            for idx, box in enumerate(boxes):  # iterate boxes
                # 這是 r 一個 box(xyxy)，不是只一個數字!!!
                # 而且這是實際數字，不是正歸化結果
                r = box.xyxyn[0].astype(int)  # get corner points as int
                #  TODO: 應該改用 xyxyn[0]  來存下正規化的椅子。
                #print(r)  # print boxes

                # text
                label_idx = labels[idx]
                # 繪製圖示
                ttext = f"{result.names[label_idx]} {confs[idx]:.2f}"
                #print(ttext)
                font_scale = 2
                if used_mask[idx]:
                    # Don't draw it.
                    # cv2.putText(img, ttext, r[:2], cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
                    # Don't draw it.
                    # cv2.rectangle(img, r[:2], r[2:], (0, 0, 255), 2)  # draw boxes on img
                    #
                    # record this result
                    self.current_yolo_results.append((box.xyxyn[0], box.xywhn[0],
                                                      result.names[label_idx],
                                                      confs[idx])
                                                     )
            #
            if verbose:
                cv2.imshow(f"aaa {idx}", resize_image_with_max_resolution(img, 800))
            #
            #
        if verbose:
            # 要一起show 多張，所以放在 for-loop 外邊
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def _path_init(self, case_root):
        """
        解析 case_root 下的 3支 pic 檔案，並將其轉換資料點。
        """
        # 偵測 case_root 底下的所有 pic 檔案
        pic_list = list(case_root.glob("*.pic"))
        # 需要 3 個 pic 檔案，
        # 1.偵測範圍
        # 2. table 範圍
        # 3. chair 位置
        # 使用 python step_1_case_layout_paint.py 來生成
        assert len(pic_list) == 3, f"只偵測到 {len(pic_list)} 支 pic 檔案, 需要 3 支，請使用 step1 程式來生成他們"

        #
        _chk_f_list = [_.name for _ in pic_list]
        assert args.polygon_pic in _chk_f_list, f"{args.polygon_pic} not exists"
        assert args.table_pic in _chk_f_list, f"{args.table_pic} not exists"
        assert args.chair_pic in _chk_f_list, f"{args.chair_pic} not exists"

        # load 總偵測範圍
        _ = case_root.joinpath(args.polygon_pic)
        assert case_root.joinpath(args.polygon_pic).exists()
        self.polygen_points = load_polygon_to_data_struct(_)

        # load 桌子位置
        _ = case_root.joinpath(args.table_pic)
        assert case_root.joinpath(args.table_pic).exists()
        self.table_points = load_table_N_to_data_struct(_)
        self.table_sit_binding_norm = [None for _ in range(len(self.table_points))]
        for i in range(len(self.table_points)):
            self.table_sit_binding_norm[i] = clac_sit_table_fields_dict(np.array(self.table_points[i]),
                                                                        debug_mode=self.debug_mode,  # !!! debug mode
                                                                        args=self.args)  # seat_over_table_rate
        # load 椅子位置
        _ = case_root.joinpath(args.chair_pic)
        assert case_root.joinpath(args.chair_pic).exists()
        if develope_mode():
            print("load_pic_dot_chair pickle content = ")
            print(load_pic_dot_chair(_))
        self.chair_points = load_pic_dot_chair(_)

    def table_numbers(self):
        return len(self.table_points)

    def chairs_numbers_in_table_N(self, table_id):
        """
        回傳 table_id 中的椅子數量
        """
        table_id = self._table_idx_check(table_id)
        return len(self.chair_points[table_id])

    def _table_idx_check(self, table_id):
        ret = -1
        if table_id >= self.table_numbers():
            raise ValueError(f"給定 table_id = {table_id}, 可指定的桌子範圍 {0} ~ {self.table_numbers() - 1}")
        else:
            ret = table_id
        return ret

    def display_all_region_chair_table(self):
        """
        顯示所有的 table 和 chair 繪製在畫面上
        :return:
        """
        global args_G

        # 隨便 load 一張圖片
        image_root = Path(self.case_root).joinpath(args_G.image_folder_name)
        assert image_root.exists(), f"image_root not exists \"{image_root}\""
        image = list(image_root.glob("*.*"))[0]
        image = cv2.imread(str(image))
        w, h = 800, 600
        image = resize_image_with_max_resolution(image, w)

        # 偵測範圍 多邊形
        converted_polygon = [(int(x * w), int(y * h)) for x, y in self.polygen_points]

        # 各桌子 四邊形
        tables = []
        for points_set in self.table_points:
            # convert normal coordinate to image coordinate,
            converted_table = [(int(x * w), int(y * h)) for x, y in points_set]
            tables.append(converted_table)

        # 各 椅子點
        chairs = []
        for points_set in self.chair_points:  # copy???
            # convert normal coordinate to image coordinate,
            converted_chair = [(int(x * w), int(y * h)) for x, y in points_set]
            chairs.append(converted_chair)

        poly_colors = (0, 255, 255)
        red_colors = (0, 0, 255)
        blue_colors = (255, 0, 0)
        green_colors = (0, 255, 0)

        # 繪製偵測範為多邊形
        cv2.polylines(image, [np.array(converted_polygon)], True, poly_colors, 2)
        #
        # 繪製各桌子
        for table in tables:
            cv2.polylines(image, [np.array(table)], True, red_colors, 2)
        #
        # 繪製各椅子
        for chair in chairs:
            for chair_point in chair:
                cv2.circle(image, chair_point, 5, green_colors, -1)  # 實心

        cv2.imshow("scene visualization", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def draw_rectangle_xyxy(self, image, top_left, bottom_right, color=(0, 0, 255), thickness=2):
        """
        cv2.retangle warp function
        :return:
        """
        top_left = top_left.copy()
        bottom_right = bottom_right.copy()
        assert len(top_left) == 2
        assert len(bottom_right) == 2

        w, h = image.shape[1::-1]

        # de-normalization
        if (0 <= top_left[0] <= 1) and (0 <= top_left[1] <= 1) and \
           (0 <= bottom_right[0] <= 1) and (0 <= bottom_right[1] <= 1):
            # 正規化座標，轉換成圖片座標
            top_left[0] = int(top_left[0] * w)
            top_left[1] = int(top_left[1] * h)
            bottom_right[0] = int(bottom_right[0] * w)
            bottom_right[1] = int(bottom_right[1] * h)

        if isinstance(top_left, np.ndarray) and \
           isinstance(bottom_right, np.ndarray):
                top_left = tuple(top_left.astype(int))
                bottom_right = tuple(bottom_right.astype(int))
        if develope_mode():
            print("期望他們是 xyxy 格式 的左上右下的座標，因為要繪製 cv2 所以要視實際座標: ")
            print("\ttop_left    : ", top_left)
            print("\tbottom_right: ", bottom_right)
        cv2.rectangle(image, top_left, bottom_right, color, thickness)

    def draw_putText_xyxy(self, frame, text, xyxyn, font_style, fontScale, color, thickness):
        w, h = frame.shape[1::-1]


        left_top_x = int(xyxyn[0] * w)
        left_top_y = int(xyxyn[1] * h)
        bbox_h = int(h * abs(xyxyn[2] - xyxyn[0]))
        # 字體向上移動一咪咪 (修改 '乘號' 後的成數 來更改距離)
        delta_y = int(bbox_h * 0.1)
        #
        _ = (left_top_x, left_top_y-delta_y)
        if develope_mode():
            print("font_put_position 需要是 tuple 格式，因為要繪製 cv2 所以要視實際座標: ")
            print("\tfont_put_position: ", _)

        cv2.putText(frame, text, _, font_style, fontScale, color, thickness)

    def _most_close_table_idx(self, xy: Tuple[float, float], tables: List) -> int:
        def point_to_segment_distance(px, py, x1, y1, x2, y2):
            # 計算線段長度
            segment_length = math.hypot(x2 - x1, y2 - y1)

            # 如果線段長度為0，則點和線段的距離為點到端點的距離
            if segment_length == 0:
                return math.hypot(px - x1, py - y1)

            # 計算點到線段的投影點
            t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (segment_length * segment_length)
            t = max(0, min(1, t))
            projection_x = x1 + t * (x2 - x1)
            projection_y = y1 + t * (y2 - y1)

            # 計算點到投影點的距離
            distance = math.hypot(px - projection_x, py - projection_y)
            return distance

        def point_to_quad_distance(px, py, quad):
            assert len(quad) == 4   # 四個點
            x1, y1 = quad[0]
            x2, y2 = quad[1]
            x3, y3 = quad[2]
            x4, y4 = quad[3]
            distances = [
                point_to_segment_distance(px, py, x1, y1, x2, y2),
                point_to_segment_distance(px, py, x2, y2, x3, y3),
                point_to_segment_distance(px, py, x3, y3, x4, y4),
                point_to_segment_distance(px, py, x4, y4, x1, y1)
            ]
            return min(distances)

        closest_idx = -1
        min_distance = float('inf')

        for i, table in enumerate(tables):
            distance = point_to_quad_distance(xy[0], xy[1], table)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i

        return closest_idx

    def check_person_in_which_table(self, xywhn):
        x, y, _, _ = xywhn
        self.table_points

        return self._most_close_table_idx((x, y), self.table_points)

    def check_person_covers_chair(self, xyxyn, table_id, width_extra):
        w, h = xyxyn[2] - xyxyn[0], xyxyn[3] - xyxyn[1]
        shift_w = w * (width_extra/2)
        for chair_idx in range(len(self.person_in_chair_N[table_id])):
            chair_xn, chair_yn = self.chair_points[table_id][chair_idx]  # copy???
            _polygon_list = [
                [xyxyn[0]-shift_w, xyxyn[1]],
                [xyxyn[2]+shift_w, xyxyn[1]],
                [xyxyn[2]+shift_w, xyxyn[3]],
                [xyxyn[0]-shift_w, xyxyn[3]],
            ]
            #  self.person_in_chair_N  : 椅子 MASK
            if point_in_polygon( (chair_xn, chair_yn), _polygon_list):
                self.person_in_chair_N[table_id][chair_idx] = 1
            # 只給 1，不給 0

    def draw_all_chair_dot(self, frame):
        w, h = frame.shape[1::-1]
        for talbe_idx in range(self.table_numbers()):
            for chair_xn, chair_yn in self.chair_points[talbe_idx].copy():
                chair_x = int(chair_xn * w)
                chair_y = int(chair_yn * h)
                cv2.circle(frame, (chair_x, chair_y), 5, (0, 255, 0), -1)

    def draw_table_idx_on_frame(self, frame):
        w, h = frame.shape[1::-1]
        for table_idx in range(self.table_numbers()):
            self.table_points[table_idx]
            center_t_x, center_t_y = calculate_polygon_center(self.table_points[table_idx])
            if 0:
                # shift left font start position
                center_t_x = center_t_x - w*0.05
            cv2.putText(frame, f"table: {table_idx+1}", (int(center_t_x*w), int(center_t_y*h)), cv2.FONT_ITALIC,
                        2.0, (255, 0, 0), 2, cv2.LINE_AA)


    def yolo_watch(self, frame, imshow_window_name="sufu", debug_show=False, debug_resize=0):
        """
        用檢測某個畫面，並填充此 class 的資料
        :param frame:
        :param imshow_window_name: debug 用的 window name
        :param debug_show: 會顯示 yolo watch 的偵測狀況
        :param debug_resize: 0: 不縮放原始大小,  >0: 高度縮放到指定大小
        :return:
        """
        self.draw_yolo_box = True
        self.yolo_detect(frame, verbose=self.debug_mode)

        # 更新 binding list 訊息，經由 yolo_detect() 結果填充
        #
        self.update_yolo_result_into_binding_list()

        # ====================================================================
        # 繪製總偵測區塊在 frame 上
        #  de-normalize, 反正歸化內部的點
        _w, _h = frame.shape[1::-1]

        polygen_points = [(int(x * _w), int(y * _h)) for x, y in self.polygen_points]
        # draw polygen
        if debug_show:
            cv2.polylines(frame, [np.array(polygen_points, dtype=np.int32)],
                          True, (0, 255, 255), 2)
        #
        #  self.current_yolo_results = [ (xyxyn, label_name, p), ...]
        #
        for _sufu_tuple in self.current_yolo_results:
            xyxyn, xywhn, label_name, p = _sufu_tuple
            # 如果 當前座標不再偵測範圍內，就跳過
            if not self.is_point_in_polygon(xyxyn):
                continue

            if debug_show:
                # draw boxes on img
                self.draw_rectangle_xyxy(frame, xyxyn[:2], xyxyn[2:], (0, 0, 255), 2)
            # elif self.draw_yolo_box:
            #     # 繪製 yolo bbox!!!
            #     self.draw_rectangle_xyxy(frame, xyxyn[:2], xyxyn[2:], (255, 0, 255), 4)



            if debug_show:
                fontScale = 1.5  # 字體大小
                self.draw_putText_xyxy(frame, f"{label_name} {p * 100:.0f}%", xyxyn, cv2.FONT_HERSHEY_COMPLEX,
                                       fontScale, (0, 0, 255), 2)
            # elif self.draw_yolo_box:
            #     fontScale = 1.5
            #     self.draw_putText_xyxy(frame, f"{label_name} {p * 100:.0f}%", xyxyn, cv2.FONT_HERSHEY_COMPLEX,
            #                            fontScale, (255, 0, 255), 4)

            # 取得繪製的資訊以利後續 可以繪製在最上層 不會被 桌面的方格遮住。

            self.yolo_debug_bboxes_class_p.append((xyxyn, label_name, p))

            #
            #  此 function 真正要做的是填充數據到 class
            #  以下即是執行此行為
            #
            this_person_in_table_idx = -1
            if label_name == 'person':
                # 這個人在哪個桌子上
                #
                this_person_in_table_idx = self.check_person_in_which_table(xyxyn)

                #  把這個人的眶(座標) 與 桌子綁定
                #
                # # self.person_in_table_N  : 紀錄 實際框框
                # # Out[3]: {0: [], 1: [array([341, 120, 406, 222])]}  <-- 應該會記錄 normal coordinate
                #
                self.person_in_table_N[this_person_in_table_idx].append(xyxyn)

                #  檢查這個人的框框是否包住了哪個椅子點位
                #
                # self.person_in_chair_N : 椅子是否被占用
                # Out[4]: {0: [0, 0, 0, 0, 0, 0, 0, 0], 1: [0, 0, 0, 0, 0, 0, 0]}
                #
                # 他會修改屬於他的 table id 的 椅子 mask, 為 1。
                # debug show
                # for__debug = frame.copy()
                # self.draw_all_chair_dot(for__debug)
                # self.draw_rectangle_xyxy(for__debug, xyxyn[:2], xyxyn[2:], (0, 0, 255), 2)  # draw boxes on img
                # cv2.imshow("current people and chair show", resize_image_with_max_resolution(for__debug, 800))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                self.check_person_covers_chair(xyxyn, this_person_in_table_idx, self.person_extra_occupy)

                #
            if debug_show:
                print("debug bbox info:")
                print("\t", "bbox:", xyxyn)
                print("\t", "label_name:", label_name)
                print("\t", "p:", p)

        ##
        if debug_show:
            _ = frame.copy()  # for show copy
            if debug_resize:
                _ = resize_image_with_max_resolution(_, debug_resize)
            cv2.imshow(imshow_window_name, _)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # ====================================================================
        # 桌子資料處理完畢，以下處理人在哪個位置上~~
        # ====================================================================
        #
        # self.person_in_table_N
        # Out[3]: {0: [], 1: [array([341, 120, 406, 222])]}
        # 以上代表桌子 0 沒有，桌子 1 有一個人
        #
        self.chair_person_binding_list = []
        #
        # for key, val in self.person_in_table_N.items():
        #     # print(key, val)
        #     # key: table_id
        #     # val: [某人的xyxy, ... ] 可能有很多個
        #     if len(val) == 0:
        #         # 沒有人，跳過
        #         continue
        #     else:
        #         tmp = self.person_in_chair_N[key]
        #         # 取得此張桌子的所有椅子點位置
        #         _current_table_chairs = self.chair_points[key].copy()  # 注意這種用法要 copy ...
        #         # de-normalize, 反正歸化內部的點
        #         _w, _h = frame.shape[1::-1]
        #         _current_table_chairs = [(int(x * _w), int(y * _h)) for x, y in _current_table_chairs]
        #         print(val)
        #         _current_table_persons_xyxy = val
        #         _current_table_persons_pos_center = [ [int((_[0]+_[2])/2), int((_[1]+_[3])/2)] for _ in _current_table_persons_xyxy]
        #         # cals to point distance
        #         _best_idx = None
        #         _best_distance = 99999999
        #         for _person_idx, person_pos in enumerate(_current_table_persons_pos_center):  # 有人再走訪就好
        #             for __idx, __chair_pos in enumerate(_current_table_chairs):
        #                 # if tmp[__idx] == 1:  # 如果此椅子綁了人 就跳過 不評估
        #                 #     continue
        #                 __dis = math.hypot(person_pos[0]-__chair_pos[0], person_pos[1]-__chair_pos[1])
        #                 if __dis < _best_distance:
        #                     _best_distance = __dis
        #                     _best_idx = __idx   # 替換為這個椅子
        #             # ??
        #             ## 已經幫一位 person 評估完一個椅子，將此人，綁訂在這個椅子上等等不要走訪此椅子。
        #             tmp[_best_idx] = 1
        #             # binging 椅子跟人
        #             binding_data = (_current_table_persons_xyxy[_person_idx], _current_table_chairs[_best_idx])
        #             self.chair_person_binding_list.append(binding_data)
        #         # 覆蓋此桌子的 chair mask~
        #         self.person_in_chair_N[key] = tmp


    def __check_xyxy_in_current_yolo_results(self, xyxyn):
        """
        檢查哪個 self.current_yolo_results 出現再 xyxyn 中
        :param xyxyn:
        :return:  要更新的資訊 dict
        """
        res = {'label_name': [], 'xyxyn': [], 'p': [] }

        for _sufu_tuple in self.current_yolo_results:
            _xyxyn_yolo, _xywh_yolo, _label_name, _p = _sufu_tuple
            if point_in_polygon(_xywh_yolo[:2], xyxyn):
                # 這個 yolo 框框在 xyxyn 中
                res['label_name'].append(_label_name)
                res['xyxyn'].append(_xyxyn_yolo)
                res['p'].append(_p)
        return res

    def get_secne_state_machine(self, table_id, chair_id):
        if self.state_machine_init == False:
            self.state_machine_init = True
            # new state machine handler
            self.scene_state_machine = [ list() for _ in range(self.table_numbers())]
            for __table_id in range(self.table_numbers()):
                # 初始化所有 state machine
                self.scene_state_machine[__table_id] = \
                [TB_StateMachine() for _ in range(self.chairs_numbers_in_table_N(__table_id))]
        #
        return self.scene_state_machine[table_id][chair_id]


    # ===================================================================
    def get_scene_state_history(self, table_id, chair_id):
        if self.state_history_init == False:
            self.state_history_init = True
            # new queue history handler
            self.scene_state_history = [ list() for _ in range(self.table_numbers())]
            for __table_id in range(self.table_numbers()):
                def _new_queue():
                    _queue_len = 5  # 狀態機 queue長度 (state machine len)
                    _queue = My_Queue(_queue_len)
                    [_queue.enqueue(TB_State.UNDEFINE) for _ in range(_queue_len)]
                    return _queue
                # 初始化所有 state queue history
                self.scene_state_history[__table_id] = \
                [_new_queue() for _ in range(self.chairs_numbers_in_table_N(__table_id))]
        #
        return self.scene_state_history[table_id][chair_id]

    #===================================================================
    def update_yolo_result_into_binding_list(self):
        # 桌子結果數量相等必須
        assert len(self.table_sit_binding_norm) == len(self.person_in_table_N)
        """
        self.table_sit_binding_norm[桌子編號][座位號]
        """
        def _New_field_dict_init(table_id, chair_id):
            """ 要附加的額外資訊 dict init Data Unit """
            # 沒有用 單一初始化池的方式
            # _queue_len = 5
            # _queue = My_Queue(_queue_len)
            # [_queue.enqueue(TB_State.UNDEFINE) for _ in range(_queue_len)]
            #

            return {"in_sit_field_region_Objects": [],   # 存入名稱用
                    "in_sit_field_region_Objects_xyxyn": [],  # bbox 位置
                    "in_sit_field_region_Objects_p": [],  # 精準度用
                    # ---
                    "in_table_field_region_Objects": [],
                    "in_table_field_region_Objects_xyxyn": [],
                    "in_table_field_region_Objects_p": [],
                    # ---
                    "in_chair_field_region_Objects": [],
                    "in_chair_field_region_Objects_xyxyn": [],
                    "in_chair_field_region_Objects_p": [],
                    # --- 附上 init 的 state machine
                    #"state_machine": TB_StateMachine(),
                    "state_machine": self.get_secne_state_machine(table_id, chair_id),
                    #"history_state_queue": _queue,
                    "history_state_queue": self.get_scene_state_history(table_id, chair_id),
                    }
        # ============================
        for t_idx in range(len(self.table_sit_binding_norm)):
            for sit_idx in range(len(self.table_sit_binding_norm[t_idx])):
                # dict
                self.table_sit_binding_norm[t_idx][sit_idx]
                _current_binding_dict = self.table_sit_binding_norm[t_idx][sit_idx]
                #
                _wait_to_updateIN_dict = _New_field_dict_init(t_idx, sit_idx)

                # ============================
                # 取得此座位(桌子+椅子 的範圍)的座標
                _whole_sit_F = _current_binding_dict["whole_sit_region"]
                #  檢查這個 sit field 有包住哪些 current_yolo_result 的中心點
                _in_F_res = self.__check_xyxy_in_current_yolo_results(_whole_sit_F)
                _wait_to_updateIN_dict["in_sit_field_region_Objects"] = _in_F_res['label_name']
                _wait_to_updateIN_dict["in_sit_field_region_Objects_xyxyn"] = _in_F_res['xyxyn']
                _wait_to_updateIN_dict["in_sit_field_region_Objects_p"] = _in_F_res['p']
                # ============================
                # 取得此桌子的座標
                _table_F = _current_binding_dict["table_field_region"]
                #  檢查這個 table field 有包住哪些 current_yolo_result 的中心點
                _in_F_res = self.__check_xyxy_in_current_yolo_results(_table_F)
                _wait_to_updateIN_dict["in_table_field_region_Objects"] = _in_F_res['label_name']
                _wait_to_updateIN_dict["in_table_field_region_Objects_xyxyn"] = _in_F_res['xyxyn']
                _wait_to_updateIN_dict["in_table_field_region_Objects_p"] = _in_F_res['p']
                # ============================
                # 取得此椅子的座標
                _chair_F = _current_binding_dict["sit_field_region"]
                #  檢查這個 chair field 有包住哪些 current_yolo_result 的中心點
                _in_F_res = self.__check_xyxy_in_current_yolo_results(_chair_F)
                _wait_to_updateIN_dict["in_chair_field_region_Objects"] = _in_F_res['label_name']
                _wait_to_updateIN_dict["in_chair_field_region_Objects_xyxyn"] = _in_F_res['xyxyn']
                _wait_to_updateIN_dict["in_chair_field_region_Objects_p"] = _in_F_res['p']
                # ============================
                # 更新
                self.table_sit_binding_norm[t_idx][sit_idx].update(_wait_to_updateIN_dict)

"""
    讀取 3 個生成出的 pic 檔案，並將其轉換成我自己的類別。
    # 這是使用 test image
    python  step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --verbose
    
    # 
    # 這是使用 爬下來的 data, case 1
    python  step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1"  --image_folder_name "2F_North"
    # 1.3 倍版
    python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1"  --seat_over_table_rate=1.3 --image_folder_name "2F_North" --output_save_path="./datacase/case1/output" --debug_mode
    1716343016298_65998
    # 
    # web: 開另外一個 terminal 執行。
    # python ./flask_web_app.py
    
    # 這是使用 爬下來的 data, case 2
    python  step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case2"  --image_folder_name "B1F_south"
    
    
    # 建議使用這個 command 先確認，預先繪製的區塊是正確的顯示在圖片上
    # 顯示預先定義的資料: 使用測試測資
    python  step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --check_preAnchor
    python  step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case2"  --image_folder_name "B1F_south" --check_preAnchor
    
    # 在 {random_pick_test_image} 隨機找 data detection。
    python  step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --random_pick_test_image --image_folder_name "2F_North"
    
    TODO: this command
    # 偵測整包資料夾的圖片
    python  step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --image_folder_name "2F_North" --detect_all_image 
    
    # 用別的 weight
    python  step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --image_folder_name "2F_North" --detect_all_image --yolo_w_name="yolov8n.pt"
    
    
    !!! 即時監測 command !!!
    #
    #  注意 需要手動檢查 --case_root 內是否已經有區域設定的 pic 檔案。
    #
    case1 的 layout 的配置是 '2F閱覽區(北側)' = --camara  --camara_name="2F閱覽區(北側)"
    
    command:
        python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --camara --camara_name="2F閱覽區(北側)"
    
    
    
    ################
    pip install
    ---
    pip install opencv-python
    pip install ultralytics
    pip install flask
    ################
    ~~~ 使用 local file ~~~
    #
    ## 路徑在這: `.\\datacase\\experiment4paper`
    #
    command:
        #1716166626497_25833, 這是 north 北側，所以用 case 1 內的 去預設定檔案
    
        # A_2_B case 1:
            python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716166626497_25833.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716166626497_25833"
            python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716176433698_67499.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716176433698_67499"
            python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716178963497_66001.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716178963497_66001"
            --
            python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716176433698_67499.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716176433698_67499"
            python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716176433698_67499.mkv" --output_save_path="./datacase/experiment4paper/A_2_B/1716176433698_67499_debug" --debug_mode
              
            
        # A_2_C case:
            python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_C/1716166243998_45733.mkv" --output_save_path="./datacase/experiment4paper/A_2_C/1716166243998_45733"
            python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_C/1716189403797_67499.mkv" --output_save_path="./datacase/experiment4paper/A_2_C/1716189403797_67499"
        
        # B_2_A case:
            python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_A/1716168757297_45600.mkv" --output_save_path="./datacase/experiment4paper/B_2_A/1716168757297_45600"
            python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_A/1716180647496_72001.mkv" --output_save_path="./datacase/experiment4paper/B_2_A/1716180647496_72001"
            
        # B_2_C case:
            python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_C/1716180957997_66433.mkv" --output_save_path="./datacase/experiment4paper/B_2_C/1716180957997_66433"
            python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_C/1716186104299_29833.mkv" --output_save_path="./datacase/experiment4paper/B_2_C/1716186104299_29833"
            python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/B_2_C/1716180957997_66433.mkv" --output_save_path="./datacase/experiment4paper_debug/B_2_C/1716180957997_66433" --debug_mode
        
        # C_2_A case:
            python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/C_2_A/1716187755797_74999.mkv" --output_save_path="./datacase/experiment4paper/C_2_A/1716187755797_74999"
            python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/C_2_A/1716193517098_41199.mkv" --output_save_path="./datacase/experiment4paper/C_2_A/1716193517098_41199"
        
        # C_2_B case:            
            python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/C_2_B/1716182290998_55032.mkv" --output_save_path="./datacase/experiment4paper/C_2_B/1716182290998_55032"
            python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/C_2_B/1716186242297_64900.mkv" --output_save_path="./datacase/experiment4paper/C_2_B/1716186242297_64900"
        
        # 1.3 倍 測試
        python step_2_eat_pic_to_gen_mapping.py --case_root "./datacase/case1" --local_video --local_video_path="./datacase/experiment4paper/A_2_B/1716166626497_25833.mkv" --seat_over_table_rate=1.3 --output_save_path="J:\temps/zjpj_output/datacase/experiment4paper_1_3/A_2_B/1716166626497_25833"
        
        

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My Script")
    # folder path
    parser.add_argument("--case_root", type=str, help="cam folder root path")
    parser.add_argument("--image_folder_name", default="images", type=str, help="image folder name")
    parser.add_argument("--polygon_pic", default="polygon.pic", type=str, help="polygon txt path")
    parser.add_argument("--table_pic", default="table_N.pic", type=str, help="polygon txt path")
    parser.add_argument("--chair_pic", default="table_N_chair.pic", type=str, help="polygon txt path")
    parser.add_argument("--seat_over_table_rate", default=1.4, type=float, help="seat's side:table' side (side is shorter side).")
    parser.add_argument("--check_preAnchor", action="store_true", help="display predefine data.")
    parser.add_argument("--verbose", action="store_true", help="show debug message")
    parser.add_argument("--random_pick_test_image", action="store_true", help="random pick test image")
    parser.add_argument("--detect_all_image", action="store_true", help="detect all image in {image_folder_name}")
    #parser.add_argument("--yolo_w_name", default="yolov8x.pt", type=str, help="Used yolo weight fileaname.",
    parser.add_argument("--yolo_w_name", default="./customer_training_pt/best_xpt_01.pt", type=str, help="Used yolo weight fileaname.",
                        choices=[
                            'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
                        ]
                        )
    # use real-time camera
    parser.add_argument("--camara", action="store_true", help="Use real carama to fetch image.")
    parser.add_argument("--camara_name", default="", type=str, help="camara",
                        choices=['2F閱覽區(北側)','2F閱覽區 (南側)',
                                 '2F閱覽區(西南側)','B1F閱覽區(北側)',
                                 'B1F閱覽區(南側)','B1F閱覽區(西側)']
                        )
    #
    parser.add_argument("--local_video", action="store_true", help="Use local video.")
    parser.add_argument("--local_video_path", default="experiment4paper", type=str, help="local video path.")
    #
    parser.add_argument("--output_save_path", type=str, help="if have save path, than will save output to target path.")
    #
    parser.add_argument("--debug_mode", action="store_true", help="debug mode")
    # parser.add_argument("--mode", choices=['table', 'polygon', 'chair'], type=str, help="choose mode")
    # 解析命令行參數
    args = parser.parse_args()
    args_G = args
    #
    case_root = Path(args.case_root)  # case 根目錄

    #
    scene_1 = Secne_Table_chair(case_root=case_root, debug_mode=args.debug_mode)

    if args.check_preAnchor:
        scene_1.display_all_region_chair_table()
    # 1
    # iii = r"C:\cgit\zjpj-lib-cam\datacase\case1\images\2022-09-12-17-18.jpg"
    # 0
    # iii = r"C:\cgit\zjpj-lib-cam\datacase\case1\images\2022-09-11-22-40.jpg"
    # 3
    iii = r"C:\cgit\zjpj-lib-cam\datacase\case1\images\sp.jpg"

    image_root = Path(args.case_root).joinpath(args.image_folder_name)

    # random pick
    if args.random_pick_test_image:
        assert image_root.exists(), f"image_root not exists \"{image_root}\""
        iii = str(random.choice([_ for _ in image_root.glob("*.*")]))
    #test_frame = resize_image_with_max_resolution(cv2.imread(iii), 1000)
    #for _frame in image_root.glob("*.*"):
    #for _frame in [iii, iii, iii]:

    # 使用圖片路徑
    # [_.__str__() for _ in image_root.glob("*.*")][1:200]

    # use cv2.ndarray
    # A_A = [cv2.imread(_) for _ in[iii, iii, iii]]

    _fecth_method = None
    if args.camara:
        # use generator
        _fecth_method = lib_camera_generator(args.camara_name, sleep_t=0.2)
    elif args.local_video:
        #_fecth_method = mkv_frame_generator(args.local_video_path)
        _fecth_method = mkv_frame_generator_sec(args.local_video_path, interval_sec=1)
    else:
        # just usr list of str
        _fecth_method = [_.__str__() for _ in image_root.glob("*.*")]
    assert _fecth_method is not None


    if args_G.output_save_path:
        # save path
        detected_output_save_path = Path(args_G.output_save_path)
        # create folder
        detected_output_save_path.mkdir(parents=True, exist_ok=True)
        #
        _for_saveing_counter = 1

    for _frame in _fecth_method:
        test_frame = None
        ret_code = None
        # check class
        if isinstance(_frame, str):
            test_frame = cv2.imread(_frame)
        elif isinstance(_frame, np.ndarray):
            test_frame = _frame
        elif isinstance(_frame, tuple):
            assert len(_frame) == 2  # assert format = (image, status_code)
            test_frame, ret_code = _frame
        else:
            raise ValueError(f"unknow type??: {type(_frame)}")
        #
        #
        if ret_code is not None:
            if ret_code != 0:  # 0 mean OK!
                print("camera read error! wait 5 sec to continue...")
                time.sleep(5)
                continue

        clean_frame = test_frame.copy()
        # 檢測此 frame 並填充此 class 的資料
        scene_1.yolo_watch(test_frame, debug_show=args_G.debug_mode, debug_resize=800)  # dddddddddddddddd

        #
        # print("debug chair_person_binding_list:")
        # print("\t", scene_1.chair_person_binding_list)
        # #
        # print("debug person_in_chair_N:")
        # print("\t", scene_1.person_in_chair_N)
        # {0: [0, 0, 0, 0, 0, 0, 0, 0], 1: [0, 0, 0, 0, 0, 0, 0]}  <-- 初始範例

        #

        use_old_method_rander = False
        if use_old_method_rander:
            #
            for table_idx, chair_mask in scene_1.person_in_chair_N.items():
                if sum(chair_mask) == 0:
                    print("桌子", table_idx+1, "沒有人")
                    continue
                else:
                    print("桌子 {} : 此桌目前占用 {} 人 / 剩餘座位: {} / 桌子座位總數: {}".
                          format(table_idx+1, sum(chair_mask), len(chair_mask) - sum(chair_mask), len(chair_mask)))
                    # random color
                    _rand_color = (random.randint(127, 255), random.randint(127, 255), random.randint(127, 255))
                    for idx, mask_TF in enumerate(chair_mask):
                        if mask_TF:  # mask true false...
                            # 印出座標
                            _norm_pos_xy = scene_1.chair_points[table_idx][idx].copy()
                            #print("assert debug: ", _norm_pos_xy)
                            if (0 <= _norm_pos_xy[0] <= 1 and 0 <= _norm_pos_xy[1] <= 1):
                                _norm_pos_xy[0] = int(_norm_pos_xy[0] * scene_1.yolo_detect_w)
                                _norm_pos_xy[1] = int(_norm_pos_xy[1] * scene_1.yolo_detect_h)
                            #print("  : 椅子編號", idx, "的位置有人", _norm_pos_xy)
                            # 繪製 dot
                            __dot_eval_x, __dot_eval_y = clean_frame.shape[1::-1]
                            __min_dot_size = int (min(__dot_eval_x, __dot_eval_y) * 0.01)
                            cv2.circle(clean_frame, _norm_pos_xy, __min_dot_size, _rand_color, -1)  # 實心
            #
            # 繪製桌子編號
            scene_1.draw_table_idx_on_frame(clean_frame)
            #
            cv2.imshow("Person dot detection Debug, same table person have same dot Color",
                      resize_image_with_max_resolution(clean_frame, 800))
            #cv2.waitKey(1000)
        # ===================================================================
        #
        use_field_binding_list_rander = True
        #
        DRAW_TABLE_FIELD = False
        DRAW_SIT_ONLY_FIELD = False

        # 繪製座位上的文字
        DRAW_STATE_MACHINE_TEXT_ON_SIT = False
        #
        #
        if use_field_binding_list_rander:
            # copy output frame
            field_output_frame2 = test_frame.copy()
            _w, _h = field_output_frame2.shape[1::-1]
            # 整個桌子占用的顏色
            whole_sit_bbox_color_occupied, whole_thickness = (0, 0, 255), 10
            whole_sit_bbox_color_idle = (0, 255, 0)
            whole_sit_bbox_color_available = (255, 0, 0)
            #
            # 桌子占用的顏色
            table_bbox_color_occupied, table_thickness = (0, 0, 255), 8
            table_bbox_color_idle = (5, 255, 2)
            #
            # 椅子占用的顏色
            sit_bbox_color_occupied, sit_thickness = (0, 0, 255), 8
            sit_bbox_color_idle = (10, 255, 10)
            #
            for table_idx, cur_table_sit_list in enumerate(scene_1.table_sit_binding_norm):
                # parameters ~~ preparing ~~
                display_table_idx = table_idx + 1
                for sit_idx, a_sit in enumerate(cur_table_sit_list):
                    display_sit_idx = sit_idx + 1
                    # 檢查這個位置的 注入的 Objects 訊息，決定是否需要印出
                    _whole_sit_F = a_sit["whole_sit_region"]
                    _table_F = a_sit["table_field_region"]
                    _sit_F = a_sit["sit_field_region"]
                    _a_sit_state_machine = a_sit["state_machine"]
                    _a_stt_state_history = a_sit["history_state_queue"]
                    #
                    # 檢查 whole field 範圍是否有 label
                    _w_bboxn = _whole_sit_F
                    #
                    # 印個基底白
                    draw_norm_polygon_on_image(field_output_frame2, _w_bboxn, (255, 255, 255),
                                               thickness=whole_thickness, inward=False)
                    # 座位區有物物件進入(有 label)
                    if len(a_sit["in_sit_field_region_Objects"]) != 0\
                            or len(a_sit["in_table_field_region_Objects"]) != 0\
                            or len(a_sit["in_chair_field_region_Objects"]) != 0:
                        #*******************
                        # # !!! 舊的繪製地方。新版狀態機繪製，會往下方後判斷後繪製
                        # 有 label 則化占用框
                        # for w_label in a_sit["in_sit_field_region_Objects"]:
                        #     draw_norm_polygon_on_image(field_output_frame2, _w_bboxn, whole_sit_bbox_color_occupied,
                        #                                whole_thickness)
                        #     # 除非未來有多個 label 要印出咚咚的需求，否則有印 occupied 即可跳走
                        #     break
                        #*******************
                        # ================== 處理 state machine ==================
                        # 有人
                        #print("debug: ", a_sit["in_sit_field_region_Objects"])
                        #if 'person' in a_sit["in_sit_field_region_Objects"]:
                        if 'person' in [str(_).lower() for _ in a_sit["in_sit_field_region_Objects"]]:
                            # move state to Occupied(有佔用) 狀態

                            _a_sit_state_machine.to_occupied()
                            _a_stt_state_history.enqueue(TB_State.OCCUPIED)
                        else:  # 非人物品占用
                            # move state machine to available(可使用，但有物品) 狀態
                            _a_sit_state_machine.to_available()
                            _a_stt_state_history.enqueue(TB_State.AVAILABLE)
                    else:
                        # 空閒狀態
                        # # !!! 舊的繪製地方。新版狀態機繪製，會往下方後判斷後繪製
                        # draw_norm_polygon_on_image(field_output_frame2, _w_bboxn, whole_sit_bbox_color_idle,
                        #                            whole_thickness)
                        # move state machine to Vacant(無佔用) 狀態
                        _a_sit_state_machine.to_vacant()
                        _a_stt_state_history.enqueue(TB_State.VACANT)  # 紀錄塞入狀態

                    #
                    # ===================================================================
                    #3
                    the_truth_state = _a_sit_state_machine.get_truth_state(_a_stt_state_history)
                    # New ---- 根據狀態機的真實狀態，決定要印出的文字
                    if the_truth_state == "OCCUPIED":
                        draw_norm_polygon_on_image(field_output_frame2, _w_bboxn, whole_sit_bbox_color_occupied,
                                                       whole_thickness)
                    elif the_truth_state == "AVAILABLE":
                        draw_norm_polygon_on_image(field_output_frame2, _w_bboxn, whole_sit_bbox_color_available,
                                                   whole_thickness)
                    elif the_truth_state == "VACANT":
                        draw_norm_polygon_on_image(field_output_frame2, _w_bboxn, whole_sit_bbox_color_idle,
                                                   whole_thickness)
                    #
                    if DRAW_STATE_MACHINE_TEXT_ON_SIT:
                        # 印出 state_list
                        # print(_a_stt_state_history._get_queue_lst())
                        ####
                        # 透過狀態機，決定要印出的文字
                        #the_truth_state = _a_sit_state_machine.get_truth_state(_a_stt_state_history)
                        state_display_str = _a_sit_state_machine.get_state_str()
                        if (the_truth_state != -1):  # 具有真實狀態 可採用
                            state_display_str = the_truth_state
                        # =======
                        # 算出中心點
                        _poly_center = calculate_polygon_center(_w_bboxn)
                        # 繪製 state text on image.
                        cv2.putText(img=field_output_frame2,
                                    text=state_display_str,
                                    org=(int(_poly_center[0] * field_output_frame2.shape[1]),  # x, de-normalize
                                         int(_poly_center[1] * field_output_frame2.shape[0])),  # y, de-normalize
                                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                                    fontScale=1.5, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA
                                    )

                    #
                    # 檢查 table field 範圍是否有 label:
                    _w_bboxn = _table_F
                    if DRAW_TABLE_FIELD:
                        if len(a_sit["in_table_field_region_Objects"]) != 0:
                            # 有 label 則化占用框
                            for w_label in a_sit["in_table_field_region_Objects"]:
                                draw_norm_polygon_on_image(field_output_frame2, _w_bboxn, table_bbox_color_occupied,
                                                           table_thickness)
                                # 除非未來有多個 label 要印出咚咚的需求，否則有印 occupied 即可跳走
                                break
                        else:
                            # 空閒狀態
                            draw_norm_polygon_on_image(field_output_frame2, _w_bboxn, table_bbox_color_idle,
                                                       table_thickness)
                    #
                    # ===================================================================
                    # 檢查 sit field 範圍是否有 label:
                    _w_bboxn = _sit_F
                    if DRAW_SIT_ONLY_FIELD:
                        if len(a_sit["in_chair_field_region_Objects"]) != 0:
                            # 有 label 則化占用框
                            for w_label in a_sit["in_chair_field_region_Objects"]:
                                draw_norm_polygon_on_image(field_output_frame2, _w_bboxn, sit_bbox_color_occupied,
                                                           sit_thickness)
                                # 除非未來有多個 label 要印出咚咚的需求，否則有印 occupied 即可跳走
                                break
                        else:
                            # 空閒狀態
                            draw_norm_polygon_on_image(field_output_frame2, _w_bboxn, sit_bbox_color_idle,
                                                       sit_thickness)
                    #  A sit iterate complete... go to next sit ~
                #  A table iterate complete... go to next sit ~
            # ===============================================================
            # 繪製 self.yolo_debug_bboxes_class_p 在準備輸出的 圖片上
            for _xyxyn, _label_name, _p  in scene_1.yolo_debug_bboxes_class_p:

                if _label_name == 'Person':
                    # 計算 這個 bbox 的中間點  _xyxyn = [x1, y1, x2, y2]
                    # 他們是正規化座標
                    _w, _h = field_output_frame2.shape[1::-1]
                    _center = calculate_bbox_center(_xyxyn)
                    #計算好的絕對座標
                    abs_point_x_y = (np.array([_w, _h]) * np.array(calculate_bbox_center(_xyxyn))).astype(int)
                    # 繪製一個紅點在  abs_point_x_y 上面。
                    cv2.circle(field_output_frame2, tuple(abs_point_x_y), 5, (0, 0, 255), -1)


                # draw bbox
                scene_1.draw_rectangle_xyxy(field_output_frame2, _xyxyn[:2], _xyxyn[2:], (0,0,225), 2)
                _fontScale = 1.5  # 文字大小
                # 繪製說明文字
                scene_1.draw_putText_xyxy(field_output_frame2, f"{_label_name} {_p * 100:.0f}%", _xyxyn,
                                          cv2.FONT_HERSHEY_COMPLEX,
                                            _fontScale, (0, 0, 255), 2)
            scene_1.yolo_debug_bboxes_class_p = []  # init~~
            # ===============================================================


            # all table done show it
            cv2.imshow("field_output_frame2", resize_image_with_max_resolution(field_output_frame2, 800))

            #
            if args_G.output_save_path:
                # save path
                cv2.imwrite(str(detected_output_save_path.joinpath(f"{_for_saveing_counter:0>5}.jpg")), field_output_frame2)
                _for_saveing_counter += 1
        # ===================================================================
        # use_field_binding_list_rander === END ===
        cv2.waitKey(800)
        # ===================================================================

        # gen tags for web
        new_tags_list = []
        # 取一個空的 dict for save result
        new_tags = myweb_app.WEB_get_empty_D_template_for_seat()
        # hard-code
        new_tags["tag_name"] = "scene_1"

        # set other value
        new_tags["number_of_tables"] = scene_1.table_numbers()

        # - set "table_layout": (-1, -1),
        assert  scene_1.chairs_numbers_in_table_N(0)  >  0   # 預期 第 0 張桌子有 > 0 的座位
        new_tags["table_layout"] = (2, scene_1.chairs_numbers_in_table_N(0)//2)

        #  "seats_color": [], 根據 桌子狀態，填入顏色
        # --- 固定參數 取得~
        _seat_blue_ = myweb_app.WEB_SEAT_COLOR["blue_"]
        _seat_green = myweb_app.WEB_SEAT_COLOR["green"]
        _seat_red__ = myweb_app.WEB_SEAT_COLOR["red__"]
        _seat_undefine = (255,255,255)
        # ---

        # ---
        hehe_dict = scene_1.person_in_chair_N

        # -- set "seats_color": [],
        for idx, val in hehe_dict.items():
            _a_seat_color_empty_ready = []
            #print("桌子id", idx, ", list", val)

            _a_table_queue_history = scene_1.scene_state_history[idx]
            _a_table_state_machines = scene_1.scene_state_machine[idx]

            # 走訪 val
            for _in_idx in range(len(val)):  ##  根據每個位置的狀態機 來決定顏色
                a_seat_single_queue = _a_table_queue_history[_in_idx]
                a_seat_single_st = _a_table_state_machines[_in_idx]
                a_seat_truth_label = a_seat_single_st.get_truth_state(a_seat_single_queue)
                if a_seat_truth_label == "OCCUPIED":
                    _a_seat_color_empty_ready.append(_seat_red__)
                elif a_seat_truth_label == "AVAILABLE":
                    _a_seat_color_empty_ready.append(_seat_blue_)
                elif a_seat_truth_label == "VACANT":
                    _a_seat_color_empty_ready.append(_seat_green)
                else:
                    _a_seat_color_empty_ready.append(_seat_undefine)

            new_tags["seats_color"].append(_a_seat_color_empty_ready)
        print(" ")

        # 這邊超級 hard code 座位順序
        if len(new_tags['seats_color']) == 2:
            seat_0 = new_tags['seats_color'][0].copy()
            seat_1 = new_tags['seats_color'][1].copy()
            # swap
            new_tags['seats_color'][0] = None
            # [0~8] = 0 1 2 3     4 5 6 7
            # [3::-1] + [7:3:-1] > 3 2 1 0    7 6 5 4
            new_tags['seats_color'][0] = seat_1[8//2:] + seat_1[0:8//2]
            new_tags['seats_color'][1] = None
            new_tags['seats_color'][1] = seat_0[8//2:] + seat_0[0:8//2]
        elif len(new_tags['seats_color']) == 1:
            new_tags['seats_color'][0] = new_tags['seats_color'][0][::-1]
        # hard-code end

        new_tags_list.append(new_tags)
        myweb_app.write_tags_to_yaml(new_tags_list)
        # ================= for web tags end =================

        # save scene_1 image
        # save image

        #to_web_image = test_frame.copy()  # no debug
        to_web_image = field_output_frame2.copy()  # debug
        cv2.imwrite("./web_datas/scene_real_time/scene_1.jpg", to_web_image)
        # ============================ for web alter data process end =======================================

    #cv2.waitKey(0)
    cv2.destroyAllWindows()

