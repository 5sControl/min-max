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
