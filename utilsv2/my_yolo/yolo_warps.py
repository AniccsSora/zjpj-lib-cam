from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
from utilsv2.gen_utils import *
import numpy as np
from typing import Tuple, List

"""
{0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light',
 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench',
 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket',
 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave',
 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',
 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier',
 79: 'toothbrush'}
 {0: '人', 1: '腳踏車', 2: '汽車', 3: '摩托車', 4: '飛機',
  5: '巴士', 6: '火車', 7: '卡車', 8: '船', 9: '交通燈',
  10: '消防栓', 11: '停車標誌', 12: '停車計時器', 13: '長凳',
  14: '鳥', 15: '貓', 16: '狗', 17: '馬', 18: '羊', 19: '牛',
  20: '大象', 21: '熊', 22: '斑馬', 23: '長頸鹿', 24: '背包',
  25: '雨傘', 26: '手提包', 27: '領帶', 28: '手提箱', 29: '飛盤',
  30: '滑雪板', 31: '滑雪板', 32: '運動球', 33: '風箏', 34: '棒球棒',
  35: '棒球手套', 36: '滑板', 37: '衝浪板', 38: '網球拍',
  39: '瓶子', 40: '酒杯', 41: '杯子', 42: '叉子', 43: '刀',
  44: '湯匙', 45: '碗', 46: '香蕉', 47: '蘋果', 48: '三明治',
  49: '柳橙', 50: '青花菜', 51: '胡蘿蔔', 52: '熱狗', 53: '披薩',
  54：“甜甜圈”，55：“蛋糕”，56：“椅子”，57：“沙發”，58：“盆栽植物”，
  59: '床', 60: '餐桌', 61: '廁所', 62: '電視', 63: '筆記型電腦',
  64: '滑鼠', 65: '遠端', 66: '鍵盤', 67: '手機', 68: '微波爐',
  69：'烤箱'，70：'烤麵包機'，71：'水槽'，72：'冰箱'，73：'書'，
  74：'時鐘'，75：'花瓶'，76：'剪刀'，77：'泰迪熊'，78：'吹風機'，
  79：'牙刷'}
"""


def get_yolo_result_by_lable_name(name:str, images:List)-> Tuple[List[np.ndarray], List[int]]:
    """
    :param name: 想要的 label name
    :param images:  圖片 List
    :return: return Tuple 兩個 List 永遠與 `images(圖片 List)` 等長 ，
             return 為 None 表示 此圖片不存在目標 name 的 label，
             反之他會存放著 對應其偵測到數量的 ndarray, 與其 confidence.
    """
    model = YOLO('yolov8x.pt')
    results = model(images)

    res = [ None for _ in images]
    res_conf = [ None for _ in images]
    # Process results list
    for idx, result in enumerate(results):
        boxes = result.boxes  # Boxes object for bbox outputs
        labels = result.boxes.cls.detach().cpu()  # id
        labels_name = [model.names.get(_.item()) for _ in labels]  # 實際名稱

        if name not in labels_name:
            continue  # skip this image

        mask = [n == name for n in labels_name]

        sub_images = []
        sub_confs = []
        # draw bbox
        for bb_idx, box in enumerate(boxes):
            if mask[bb_idx] != True:  # 略過非關注的 bbox
                continue
            x1,y1,x2,y2 = [ int(_.item()) for _ in box.xyxy[0]]
            w = x2 - x1
            h = y2 - y1
            conf = int(box.conf.item()*100)
            _sub_image = images[idx][y1:y1+h, x1:x1+w]
            sub_images.append(_sub_image)
            sub_confs.append(conf)
        res[idx] = sub_images
        res_conf[idx] = sub_confs

    return (res, res_conf)


def main1():
    # Load a model
    model = YOLO('yolov8x.pt')  # pretrained YOLOv8n model
    # 秀出 model 的 label name.
    # model.names
    # {0: 'person',
    #  1: 'bicycle',
    #  2: 'car',
    #  ...
    #  78: 'hair drier',
    #  79: 'toothbrush'}

    # Run batched inference on a list of images
    image = cv2.imread(r"./images/nobody_lib.jpg")
    image2 = cv2.imread(r"./images/bus.jpg")
    images = [image, image2]
    results = model(images)  # return a list of Results objects

    # Process results list
    for idx, result in enumerate(results):
        annotator = Annotator(images[idx])
        boxes = result.boxes  # Boxes object for bbox outputs
        labels = result.boxes.cls.detach().cpu()  # id
        labels_name = [model.names.get(_.item()) for _ in labels]  #  實際名稱
        conf = boxes.conf.detach().cpu()
        boxes.shape  # 原圖 size
        boxes.xywh
        boxes.xywhn
        boxes.xyxy
        boxes.xyxyn
        print(labels_name)

        # draw bbox
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            conf = box.conf
            annotator.box_label(b, f"{model.names[int(c)]}_{int(conf.item()*100)}")
        img = annotator.result()
        cv2.imshow(f'YOLO V8 Detection_{idx}', resize_max_res(img))
    cv2.waitKey(0)


def main2(debug=False):
    image = cv2.imread(r"./utilsv2/my_yolo/images/nobody_lib.jpg")
    image2 = cv2.imread(r"./utilsv2/my_yolo/images/bus.jpg")
    images = [image, image2]

    objects_list, object_confs = get_yolo_result_by_lable_name('dining table', images)
    """
    image : 有兩個目標物件
    image2: 沒有物件
    -------------------------
    objects_list = [[ [ndarray1...], [ndarray_2] ]      , None]
    object_confs = [[  70          , 40          ]      , None]
    """
    i = 0
    for bboxes, confs in zip(objects_list, object_confs):
        if bboxes is None:  # this image no target label
            continue
        for bbox, conf in zip(bboxes, confs):
            i += 1
            if debug:
                cv2.imshow(f"{i}_conf_{conf}", resize_max_res(bbox))
    if debug:
        cv2.waitKey(0)


if __name__ == "__main__":
    main2(True)