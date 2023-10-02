import cv2
import numpy as np
from pathlib import Path  # 匯入 Path 類
from utils import *
# 初始化全局變數
points = []
trans_matrix_G = None
user_lines = []
drawing_mode_G = 'vertical'
user_lines_H_G = []
user_lines_V_G = []

# 保存線的位置
line_color = (0, 0, 255)  # 紅色 (BGR顏色)
line_thickness = 2
horizontal_line_y = None
vertical_line_x = None
good_points = []
trans_img = None


# callback to check user draw lines
def draw_lines(event, x, y, flags, param):
    global horizontal_line_y, vertical_line_x

    if trans_img is not None:
        img_PPP = 0.05  # 圖片的最短邊幾趴是容忍度??
        # trans_img is ndarray
        # get image w and h
        terlrate = int(min(trans_img.shape[1::-1]) * img_PPP)
    else:
        print("[warn]: use default terlrate...")
        terlrate = 20


    for gx, gy in good_points:
        if abs(y - gy) < terlrate:
            y = gy
        if abs(x - gx) < terlrate:
            x = gx

    if event == cv2.EVENT_LBUTTONDOWN:
        horizontal_line_y = y
        vertical_line_x = x
        good_points.append((x, y))
        redraw_lines()

# 用於繪制水平線和垂直線的函數
def redraw_lines(window_name='transform_image_A', line_xy=None):
    global horizontal_line_y, vertical_line_x, trans_img
    if line_xy is None:
        if horizontal_line_y is not None:
            cv2.line(trans_img, (0, horizontal_line_y), (trans_img.shape[1], horizontal_line_y), line_color, line_thickness)
        if vertical_line_x is not None:
            cv2.line(trans_img, (vertical_line_x, 0), (vertical_line_x, trans_img.shape[0]), line_color, line_thickness)
    else:
        for vertical_line_x, horizontal_line_y in line_xy:
            cv2.line(trans_img, (0, horizontal_line_y), (trans_img.shape[1], horizontal_line_y), line_color, line_thickness)
            cv2.line(trans_img, (vertical_line_x, 0), (vertical_line_x, trans_img.shape[0]), line_color, line_thickness)
    # 在視窗中顯示圖像
    cv2.imshow(window_name, trans_img)

# 點擊事件的回調函數
def click_event(event, x, y, flags, param):
    global points, trans_matrix_G

    if event == cv2.EVENT_LBUTTONDOWN:
        # 在圖片上畫一個小圓圈來標記點的位置

        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        cv2.imshow('image', img)

        # 將點的座標添加到列表中
        points.append((x, y))

        # 當收集到四個點時，進行透視變換並保存結果
        if len(points) == 4:
            cv2.destroyAllWindows()
            trans_matrix_G = get_transform_matrix(img, points)

# 執行透視變換
def transform_image(img, trans_matrix, exceed=1.5):
    global user_lines

    # 複製原圖像以避免修改原始圖像
    transformed_img = img.copy()

    # 設置目標圖像的寬度和高度與原圖相同
    width, height = img.shape[1], img.shape[0]
    new_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    result = cv2.warpPerspective(img, trans_matrix, (int(width*exceed), int(height*exceed)), borderValue=(255, 255, 255))
    # 縮放
    result = resize_image_with_max_resolution(result, 800)
    new_w, new_h = result.shape[1], result.shape[0]
    # 顯示變換後的圖像
    cv2.imshow('Transformed Image', result)

    # 使用 Path 來設定保存圖像的路徑
    output_path = Path('./output')
    output_path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path.joinpath('transformed_image.jpg')), result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result, (new_w, new_h)

def get_transform_matrix(img, points):
    # 複製原圖像以避免修改原始圖像
    transformed_img = img.copy()

    # 設置目標圖像的寬度和高度與原圖相同
    width, height = img.shape[1], img.shape[0]
    new_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    # 執行透視變換
    matrix = cv2.getPerspectiveTransform(np.array(points, dtype=np.float32), new_points)

    return matrix

if __name__ == "__main__":
    global img, clean_img
    # 載入圖片
    #img_path = Path('./images/test1.png')  # 輸入圖片的路徑
    img_path = Path('./images/zj_lab/4.jpg')  # 輸入圖片的路徑
    img = resize_image_with_max_resolution(cv2.imread(str(img_path)), 800)
    w, h = img.shape[1], img.shape[0]
    clean_img = img.copy()

    # 顯示圖片並設置點擊事件的回調函數
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    print("順時針 或者 逆時針 精準按下桌子的四個角落。")
    # 等待用戶按下任意鍵來退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    transfer_result = None
    if trans_matrix_G is not None:
        print(trans_matrix_G)
        # 定義要向左上移動的偏移量
        dx = int(w*0.2)  # 向左移動 50 像素
        dy = int(h*0.2)  # 向上移動 50 像素

        # 修改矩陣中的平移部分以實現左上移動
        trans_matrix_G[0, 2] += dx
        trans_matrix_G[1, 2] += dy
        print("\n\n\n\n\n=======================")
        print("按下空白 並下一步。")
        transfer_result, new_w_h = transform_image(clean_img, trans_matrix_G)

    else:
        print("trans_matrix_G is None, exit program.")

    # 在變形的圖案上繪製水平與垂直線
    trans_img = transfer_result.copy()
    cv2.imshow('transform_image_A', trans_img)
    cv2.setMouseCallback('transform_image_A', draw_lines)
    print("\n\n\n\n\n=======================")
    print("1. 按下滑鼠左鍵，繪製十字線")
    print("2. 按下 X 退出，並儲存此反校正後的圖片。")
    print("3. 按下 Z undo 上一條繪製的線。")
    while True:
        key = cv2.waitKey(10)

        if key == ord('x') or key == ord('X'):  # 按下x鍵退出
            break
        if key == ord('z') or key == ord('Z'):  # 按下z鍵取消上一條線條
            if len(good_points) != 0:
                good_points.pop()
                trans_img = transfer_result.copy() # reset trans_img one!
                redraw_lines(line_xy=good_points)

    cv2.destroyAllWindows()

    # 執行逆變換
    inverse_matrix = np.linalg.inv(trans_matrix_G)
    if transfer_result is not None:
        repainted_image = cv2.warpPerspective(trans_img, inverse_matrix, new_w_h, borderValue=(255, 255, 255))
        # 使用 Path 來設定保存圖像的路徑
        output_path = Path('./output')
        output_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path.joinpath('repainted_image.jpg')), repainted_image)
        cv2.imshow('Repainted Image', repainted_image)
        cv2.waitKey(0)
