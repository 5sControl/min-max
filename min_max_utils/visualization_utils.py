import cv2
import numpy as np


def draw_rect_with_text(img: np.array, rect_coords: tuple, text: str, color, text_color, **rect_kwargs):
    x1, y1, x2, y2 = list(map(int, rect_coords))
    image = cv2.rectangle(img, (x1, y1), (x2, y2), color, **rect_kwargs)
    cv2.putText(
        image,
        text,
        (x1 + 15, y1 + 30),
        cv2.FONT_HERSHEY_TRIPLEX,
        0.6,
        text_color,
        1)
    return image


def draw_line(img: np.array, line: np.array, area_coord: np.array, color=(0, 0, 255), **kwargs):
    x1_area, _, x2_area, __ = area_coord
    x1_line, y1, x2_line, y2 = line
    x_start = x1_line if x1_area < x1_line else x1_area
    x_end = x2_line if x2_line < x2_area else x2_area
    cv2.line(img, (x_start, y1), (x_end, y2), color, **kwargs)
