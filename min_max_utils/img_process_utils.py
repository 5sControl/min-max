import math
import cv2
import numpy as np
import torch


def shape2size(shape): return np.array([shape[1], shape[0]])


def fill_by_content(canvas, image):
    target_size, object_size = shape2size(
        canvas.shape), shape2size(image.shape)

    object_size = tuple(
        (object_size * (target_size / object_size).min()).astype(int))
    image = cv2.resize(image, object_size)

    shift = (target_size - object_size) // 2
    l, t, r, b = np.clip(np.stack((shift, object_size + shift), axis=0),
                         a_min=np.array([0, 0]),
                         a_max=target_size).reshape(4).astype(int)
    canvas[t:b, l:r] = image[:b - t, :r - l]
    return canvas


def letterbox(image, size=(640, 640), color=(114, 114, 114)):
    place_holder = np.ones(shape=(*size, 3), dtype=np.uint8) * \
        np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    x = fill_by_content(place_holder, image)
    return x


def get_inverse_letterbox_map(region, target_size):
    object_size, target_size = np.array(
        region[1] - region[0]), np.array(target_size)
    size = (object_size * (target_size / object_size).min()).astype(int)
    shift = (target_size - size) // 2
    lt_rb = np.clip(np.stack((shift, size + shift), axis=0),
                    a_min=np.array([0, 0]),
                    a_max=target_size).reshape(2, 2).astype(int)
    shift = lt_rb[0].reshape(1, 1, 2)
    return lambda x: (x - shift) / np.array(size).reshape(1, 1, 2) * np.array(object_size).reshape(1, 1, 2) + \
        + region[0].reshape(1, 1, 2)


def convert_image(img_, device='cpu'):
    try:
        img = letterbox(img_)
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img
    except Exception as exc:
        logger.warning("Empty image")
