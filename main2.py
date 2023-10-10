import cv2
import numpy as np
from pathlib import Path  # 匯入 Path 類
from utils import *
import time
# 初始化全局變數
points = []
user_check_4point_OK = False
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
    global points, trans_matrix_G, user_check_point_OK, clean_img

    if event == cv2.EVENT_LBUTTONDOWN:
        local_img = clean_img.copy()
        if len(points) >= 4:
            points = points[0:3]  # remove last one
        points.append((x, y))

        # 在圖片上畫一個小圓圈來標記點的位置
        for point in points:
            cv2.circle(local_img, point, 2, (0, 0, 255), -1)
        cv2.imshow('image', local_img)


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
    # =============================================
    develop_pre_input = True
    if develop_pre_input:
        points = [(65, 139), (592, 315), (764, 463), (117, 371)]
        user_check_4point_OK = True
    # =============================================


    # 載入圖片
    #img_path = Path('./images/test1.png')  # 輸入圖片的路徑
    img_path = Path('./images/zj_lab/4.jpg')  # 輸入圖片的路徑
    img = resize_image_with_max_resolution(cv2.imread(str(img_path)), 800)
    w, h = img.shape[1], img.shape[0]
    clean_img = img.copy()

    # 顯示圖片並設置點擊事件的回調函數
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    print("順時針 或者 逆時針 按下桌子的四個角落。(w,a,s,d 可以微調當前點的位置。)")
    print("當畫面上有四個點都確定OK沒問題，按下空白 並下一步。")
    # =============================================
    # 以下的 while 就是讓使用者點桌子的四個角落而已
    # =============================================
    while True:
        if len(points) == 4 and user_check_4point_OK:
            break
        key = cv2.waitKey(0)
        move_direction = None
        if key == ord('w') or key == ord('W'):
            move_direction = 'up'
        elif key == ord('s') or key == ord('S'):
            move_direction = 'down'
        elif key == ord('a') or key == ord('A'):
            move_direction = 'left'
        elif key == ord('d') or key == ord('D'):
            move_direction = 'right'
        # space
        elif key == ord(' ') and len(points) == 4:
            user_check_4point_OK = True

        if move_direction and len(points) > 0:
            x, y = points[-1]
            if move_direction == 'up':
                points[-1] = (x, y-1)
            elif move_direction == 'down':
                points[-1] = (x, y+1)
            elif move_direction == 'left':
                points[-1] = (x-1, y)
            elif move_direction == 'right':
                points[-1] = (x+1, y)

            cv2.destroyAllWindows()
            # repaint points
            img = clean_img.copy()
            for point in points:
                cv2.circle(img, point, 2, (0, 0, 255), -1)
            cv2.imshow('image', img)
            cv2.setMouseCallback('image', click_event)

    cv2.destroyAllWindows()
    for point in points:
        cv2.circle(img, point, 2, (0, 0, 255), -1)
    # 繪製封閉多邊形用
    pts = np.array(points, dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    print("points: ", points)
    cv2.imshow('image 4 points are good!', img)
    cv2.waitKey(0)
    #  使用者已經調整完畢桌子的四個角落
    #  =============================================
    #
    # 當收集到四個點時，進行透視變換並保存結果
    if len(points) == 4:
        cv2.destroyAllWindows()
        trans_matrix_G = get_transform_matrix(img, points)
        print("trans_matrix_G is: ", trans_matrix_G)
    else:
        raise Exception("len(points) != 4, it is {}".format(len(points)))
    # =============================================
    # 接下來使用 由使用者繪製 桌面方框
    # =============================================
    transfer_result = None
    if trans_matrix_G is not None:
        print(trans_matrix_G)
        # 定義要向左上移動的偏移量
        dx = int(w*0)  # 向左移動 50 像素
        dy = int(h*0)  # 向上移動 50 像素

        # 修改矩陣中的平移部分以實現左上移動
        trans_matrix_G[0, 2] += dx
        trans_matrix_G[1, 2] += dy
        print("\n\n\n\n\n=======================")
        transfer_result, new_w_h = transform_image(clean_img, trans_matrix_G)
        print("transfer_result (w, h): ", transfer_result.shape[1::-1])
        print("new_w_h (w, h): ", new_w_h)
        print("按下空白 並下一步。")
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
        key = cv2.waitKey(0)

        if key == ord('x') or key == ord('X'):  # 按下x鍵退出
            break
        if key == ord('z') or key == ord('Z'):  # 按下z鍵取消上一條線條
            if len(good_points) != 0:
                good_points.pop()
                trans_img = transfer_result.copy() # reset trans_img one!
                redraw_lines(line_xy=good_points)
    cv2.destroyAllWindows()
    #
    ## 定義待轉換的點, good_points 是校正桌面上的十字點中心
    src_points = np.array([good_points], dtype=np.float32)
    # int
    new_src_points = []
    for xy in src_points[0]:
        _x, _y = xy
        new_src_points.append([int(_x), int(_y)])
    print("src_points (x,y):", new_src_points)

    # debug: print src_points on clean_img
    debug_img = clean_img.copy()
    for xy in new_src_points:
        cv2.circle(debug_img, tuple(xy), 2, (0, 0, 255), -1)
    cv2.imshow('debug_img', debug_img)
    cv2.waitKey(0)
    #

    # 執行逆變換
    inverse_matrix = np.linalg.inv(trans_matrix_G)

    ## 使用逆透視變換
    dst_points = cv2.perspectiveTransform(src_points, inverse_matrix)
    print("dst_points (x,y):", dst_points)
    #

    if transfer_result is not None:
        repainted_image = cv2.warpPerspective(trans_img, inverse_matrix, new_w_h, borderValue=(255, 255, 255))
        # 使用 Path 來設定保存圖像的路徑
        output_path = Path('./output')
        output_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path.joinpath('repainted_image.jpg')), repainted_image)
        cv2.imshow('Repainted Image', repainted_image)
        cv2.waitKey(0)
