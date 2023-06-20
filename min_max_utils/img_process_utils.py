import cv2
import numpy as np
import torch
from PIL import ImageFont, ImageDraw, Image


def save_image(img: np.array, name: str):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    pil_img.save(name, 'PNG', transparency=(10, 10, 10))


def transfer_coords(prev_coords: torch.Tensor, main_item_coords: tuple) -> list:
    x1_item, y1_item, x2_item, y2_item = main_item_coords
    prev_coords = prev_coords.cpu().numpy()
    local_boxes = prev_coords.reshape(-1)
    x1n, y1n, x2n, y2n, proba = local_boxes
    coords = [x1n + x1_item, y1n + y1_item, x2n + x1_item, y2n + y1_item, proba]
    return coords
