import cv2
import numpy as np
import torch
from PIL import ImageFont, ImageDraw, Image

def save_image(img: np.array, name: str):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    pil_img.save(name, 'PNG', transparency=(10, 10, 10))
