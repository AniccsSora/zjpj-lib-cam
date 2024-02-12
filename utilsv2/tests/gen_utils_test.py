import numpy as np
import cv2
from utilsv2.gen_utils import *
import matplotlib
matplotlib.use('Agg')  # 使用非GUI後端

def test_resize_max_res():
    # make fake iamge
    image3c = np.random.randint(0, 255, (1000, 2000, 3), dtype=np.uint8)
    image1c = np.random.randint(0, 255, (801, 801), dtype=np.uint8)

    # test for 3 channel image
    resized_image = resize_max_res(image3c, 800)
    assert resized_image.shape[0] <= 800
    assert resized_image.shape[1] <= 800
    assert resized_image.shape[2] == 3
    # test for 1 channel image
    resized_image = resize_max_res(image1c, 800)
    assert resized_image.shape[0] <= 800
    assert resized_image.shape[1] <= 800
    assert resized_image.ndim == 2


def test_plot_images():
    # make some fake images.
    image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    edges = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    lines = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    hahaha = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    plot_images(
        [image, gray_image, edges, lines],
        ['Original Image', 'Gray Image', 'Edges', 'Lines Detected']
    )
    plot_images(
        [image, gray_image, edges, lines, hahaha],
        ['Original Image', 'Gray Image', 'Edges', 'Lines Detected', 'hahaha']
    )
    assert True