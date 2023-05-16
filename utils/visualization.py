import cv2
import numpy as np


def draw_rect_with_text(img: np.array, rect_coords: tuple, text: str, color, **rect_kwargs):
    x1, y1, x2, y2 = rect_coords
    image = cv2.rectangle(img, (x1, y1), (x2, y2), color, **rect_kwargs)
    cv2.putText(
        image,
        text,
        (x1 + 30, y1 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2)
    return image
