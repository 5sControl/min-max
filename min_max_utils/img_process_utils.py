import cv2
import numpy as np
from PIL import Image
from numba import njit, float64, int64


def save_image(img: np.array, name: str):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    pil_img.save(name, 'PNG', transparency=(10, 10, 10))

@njit(float64[:](float64[:], int64[:]))
def transfer_coords(prev_coords: np.array, main_item_coords: tuple) -> np.array:
    result_coords = prev_coords.copy()
    result_coords[:2] += main_item_coords[:2]
    result_coords[2:-1] += main_item_coords[:2]
    return result_coords
