import cv2
import numpy as np
from pathlib import Path
from utils import *

# 定義全局變數來保存線的位置
line_color = (0, 0, 255)  # 紅色 (BGR顏色)
line_thickness = 2
horizontal_line_y = None
vertical_line_x = None
good_points = []

# 回調函數，用於處理鼠標事件
def draw_lines(event, x, y, flags, param):
    global horizontal_line_y, vertical_line_x

    if event == cv2.EVENT_LBUTTONDOWN:
        # 用戶點擊鼠標左鍵，繪制水平線和垂直線
        horizontal_line_y = y
        vertical_line_x = x
        good_points.append((x, y))
        redraw_lines()

# 用於繪制水平線和垂直線的函數
def redraw_lines(window_name='image', points=None):
    global  horizontal_line_y, vertical_line_x, good_points
    if points is None:
        if horizontal_line_y is not None:
            cv2.line(img, (0, horizontal_line_y), (img.shape[1], horizontal_line_y), line_color, line_thickness)
        if vertical_line_x is not None:
            cv2.line(img, (vertical_line_x, 0), (vertical_line_x, img.shape[0]), line_color, line_thickness)
    else:
        for vertical_line_x, horizontal_line_y in good_points:
            cv2.line(img, (0, horizontal_line_y), (img.shape[1], horizontal_line_y), line_color, line_thickness)
            cv2.line(img, (vertical_line_x, 0), (vertical_line_x, img.shape[0]), line_color, line_thickness)
    # 在視窗中顯示圖像
    cv2.imshow(window_name, img)


if __name__ == "__main__":
    img_path = Path('./images/zj_lab/4.jpg')  # 輸入圖片的路徑
    img = resize_image_with_max_resolution(cv2.imread(str(img_path)), 800)
    clean_img = img.copy()
    cv2.imshow('image', img)
    # 創建視窗並將回調函數與視窗關聯
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_lines)

    while True:
        key = cv2.waitKey(10)
        if key == ord('x') or key == ord('X'):  # 按下x鍵退出循環
            break
        if key == ord('z') or key == ord('Z'):  # 按下x鍵退出循環
            good_points.pop()
            img = clean_img.copy()
            redraw_lines(points=good_points)

    cv2.destroyAllWindows()
