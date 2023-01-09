import cv2
import PIL
from PIL import ImageDraw, Image, ImageFont, ImageOps
import numpy as np
from glob import glob
from tqdm import tqdm
import os


# CV2からPILへ変換する関数

def cvToPil(image):
    #opencv > PIL型
    new_image = image.copy()
    if new_image.ndim == 2:     # モノクロ
        pass
    elif new_image.shape[2] == 3:   # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:   # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)

    new_image = Image.fromarray(new_image)
    return new_image

# PILからCV2へ変換する関数


def pilToCv(image):
    # PIL > opencv
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


def process(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    # 画像を灰色にする
    img_gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    ##
    output = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # ノイズ除去
    blurs = []
    for i in range(10):
        if i == 0:
            blurs.append(cv2.GaussianBlur(output, (5, 5), 0))
        else:
            blurs.append(cv2.GaussianBlur(blurs[-1], (5, 5), 0))

    return blurs[-1]
