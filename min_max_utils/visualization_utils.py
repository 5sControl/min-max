import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image


def draw_rect(img, rect_coords, color, **rect_kwargs):
    x1, y1, x2, y2 = list(map(int, rect_coords))
    image = cv2.rectangle(img, (x1, y1), (x2, y2), color, **rect_kwargs)
    return image


def draw_text(img, coords: list, text: str, area_size:int, text_color: tuple, min_font_size: int = 10, img_fraction: int = 0.3):
    image = draw_text_util(img, text, coords=coords, text_color=text_color, area_size=area_size,
                           min_font_size=min_font_size, fraction=img_fraction)
    return image


def get_scaled_font(text: str, area_size: int, img_fraction: float = 0.5, min_font_size: int = 14) -> ImageFont:
    font = ImageFont.truetype("fonts/Inter-Bold.ttf", min_font_size)
    fontsize = min_font_size
    while font.getlength(text) < img_fraction * area_size:
        fontsize += 1
        font = ImageFont.truetype("fonts/Inter-Bold.ttf", fontsize)
    return font


def draw_text_util(img: np.array, text: str, coords: tuple[int, int], text_color: tuple[int, int, int], area_size: int,
                   min_font_size: int, fraction: float) -> np.array:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    draw.text(
        coords, 
        text, 
        font=get_scaled_font(text, area_size, fraction) if fraction is not None else ImageFont.truetype("fonts/Inter-Bold.ttf", min_font_size), 
        fill=text_color,
        min_font_size=min_font_size
    )
    # noinspection PyTypeChecker
    img = np.asarray(pil_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


def draw_line(img: np.array, line: np.array, area_coord: np.array, color: tuple = (0, 200, 0), **kwargs) -> np.array:
    x1_area, _, x2_area, __ = area_coord
    x1_line, y1, x2_line, y2 = line
    x_start = x1_line if x1_area < x1_line else x1_area
    x_end = x2_line if x2_line < x2_area else x2_area
    cv2.line(img, (x_start, y1), (x_end, y2), color, **kwargs)
    return img
