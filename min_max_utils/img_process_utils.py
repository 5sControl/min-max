import cv2
import numpy as np
import torch
from PIL import ImageFont, ImageDraw, Image

def save_image(img: np.array, name: str):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    pil_img.save(name, 'PNG', transparency=(10, 10, 10))

def transfer_coords(prev_coords: torch.Tensor, area_coords: tuple) -> list:
    x1, y1, x2, y2 = area_coords
    prev_coords = prev_coords.numpy()
    local_boxes = prev_coords.reshape(-1)
    x1n, y1n, x2n, y2n = local_boxes
    coords = (x1n + x1, y1n + y1, x2n + x1, y2n + y1)
    return coords
