import cv2
import numpy as np
from pathlib import Path  # 匯入 Path 類
from utils import *
# 初始化全局變數
points = []
trans_matrix_G = None

# 點擊事件的回調函數
def click_event(event, x, y, flags, param):
    global points, trans_matrix_G

    if event == cv2.EVENT_LBUTTONDOWN:
        # 在圖片上畫一個小圓圈來標記點的位置
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow('image', img)

        # 將點的座標添加到列表中
        points.append((x, y))

        # 當收集到四個點時，進行透視變換並保存結果
        if len(points) == 4:
            cv2.destroyAllWindows()
            trans_matrix_G = get_transform_matrix(img, points)

# 執行透視變換
def transform_image(img, trans_matrix, exceed=1.2):
    # 複製原圖像以避免修改原始圖像
    transformed_img = img.copy()

    # 設置目標圖像的寬度和高度與原圖相同
    width, height = img.shape[1], img.shape[0]
    new_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    result = cv2.warpPerspective(img, trans_matrix, (int(width*exceed), int(height*exceed)), borderValue=(255, 255, 255))

    # 顯示變換後的圖像
    cv2.imshow('Transformed Image', resize_image_with_max_resolution(result, 800))

    # 使用 Path 來設定保存圖像的路徑
    output_path = Path('./output')
    output_path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path.joinpath('transformed_image.jpg')), result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    # 載入圖片
    #img_path = Path('./images/test1.png')  # 輸入圖片的路徑
    img_path = Path('./images/zj_lab/1.jpg')  # 輸入圖片的路徑
    img = resize_image_with_max_resolution(cv2.imread(str(img_path)), 800)
    w, h = img.shape[1], img.shape[0]
    clean_img = img.copy()

    # 顯示圖片並設置點擊事件的回調函數
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)

    # 等待用戶按下任意鍵來退出
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if trans_matrix_G is not None:
        print(trans_matrix_G)
        # 定義要向左上移動的偏移量
        dx = int(w*0.2)  # 向左移動 50 像素
        dy = int(h*0.2)  # 向上移動 50 像素

        # 修改矩陣中的平移部分以實現左上移動
        trans_matrix_G[0, 2] += dx
        trans_matrix_G[1, 2] += dy
        transform_image(clean_img, trans_matrix_G)
    else:
        print("trans_matrix_G is None, exit program.")
