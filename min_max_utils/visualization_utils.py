import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image


def draw_rect_with_text(img: np.array, rect_coords: tuple, text: str, color, text_color, **rect_kwargs):
    x1, y1, x2, y2 = list(map(int, rect_coords))
    image = cv2.rectangle(img, (x1, y1), (x2, y2), color, **rect_kwargs)
    if text:
        image = draw_text(image, text, (x1 + 15, y1 + 30), text_color, x2 - x1)
    return image

def get_scaled_font(text, area_size, img_fraction=0.5):
    font = ImageFont.truetype("Inter-Bold.ttf", 1)
    fontsize = 1
    while font.getsize(text)[0] < img_fraction * area_size:
        fontsize += 1
        font = ImageFont.truetype("Inter-Bold.ttf", fontsize)
    return font

def draw_text(img: np.array, text: str, coords: tuple[int], text_color: tuple[int], area_size: int):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.text(coords, text, font=get_scaled_font(text, area_size), fill=text_color)
    img = np.asarray(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def draw_line(img: np.array, line: np.array, area_coord: np.array, color=(0, 200, 0), **kwargs):
    x1_area, _, x2_area, __ = area_coord
    x1_line, y1, x2_line, y2 = line
    x_start = x1_line if x1_area < x1_line else x1_area
    x_end = x2_line if x2_line < x2_area else x2_area
    cv2.line(img, (x_start, y1), (x_end, y2), color, **kwargs)
