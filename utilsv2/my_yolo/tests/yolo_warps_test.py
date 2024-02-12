import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from ultralytics import YOLO
from utilsv2.my_yolo import yolo_warps

def test_yolo_warps():
    model = YOLO('yolov8n.pt')  # pretrained YOLOv8n model
    # 秀出 model 的 label name.
    # model.names
    # {0: 'person',
    #  1: 'bicycle',
    #  2: 'car',
    #  ...
    #  78: 'hair drier',
    #  79: 'toothbrush'}

    # Run batched inference on a list of images
    results = model([r"./utilsv2/my_yolo/images/nobody_lib.jpg"])  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        labels = result.boxes.cls.detach().cpu()  # id
        labels_name = [model.names.get(_.item()) for _ in labels]  # 實際名稱
        conf = result.boxes.conf.detach().cpu()
        boxes.shape  # 原圖 size
        boxes.xywh
        boxes.xywhn
        boxes.xyxy
        boxes.xyxyn
        print(labels_name)

def test_yolo_warps2():
    # 測試實際拿取 dining table 用例
    yolo_warps.main2()