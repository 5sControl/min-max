import json
from collections import Counter
import logging
import colorlog
import cv2
import numpy as np
from min_max_utils.img_process_utils import transfer_coords
from typing import Sequence


def drop_area(areas: Sequence[dict], item_idx: int, item: dict, subarea_idx: int):
    logger = logging.getLogger('min_max_logger')
    if len(item['coords']) == 1:
        logger.info("Item was dropped - {}".format(areas.pop(item_idx)))
    else:
        logger.info("Subarea was dropped - {}".format(item.get('coords').pop(subarea_idx)))


def create_logger():
    logger = logging.getLogger('min_max_logger')
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'CRITICAL': 'bold_red,bg_white',
        }))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


def find_red_line(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0, 140, 50])
    upper_red = np.array([8, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([175, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 + mask1
    output_img = img.copy()
    output_img[np.where(mask == 0)] = 0
    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask == 0)] = 0
    src = cv2.cvtColor(output_img, cv2.COLOR_HSV2RGB)
    src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    dst = cv2.medianBlur(src, 3)
    lines_p = cv2.HoughLinesP(dst, rho=1, theta=np.pi / 180,
                              threshold=50, lines=None, minLineLength=20, maxLineGap=15)
    lines = []
    if lines_p is not None:
        for i in range(0, len(lines_p)):
            line = lines_p[i][0]
            if abs(line[1] - line[3]) < 25:
                lines.append(line)
    return lines


def is_line_in_area(area, line):
    x1, y1, x2, y2 = line
    min_x, min_y, max_x, max_y = area

    if (x1 <= min_x and x2 <= min_x) or (y1 <= min_y and y2 <= min_y) or (x1 >= max_x and x2 >= max_x) or (
            y1 >= max_y and y2 >= max_y):
        return False

    m = (y2 - y1) / (x2 - x1 + 1e-3)

    y = m * (min_x - x1) + y1
    if min_y < y < max_y:
        return True

    y = m * (max_x - x1) + y1
    if min_y < y < max_y:
        return True

    x = (min_y - y1) / m + x1
    if min_x < x < max_x:
        return True

    x = (max_y - y1) / m + x1
    if min_x < x < max_x:
        return True

    return False


def most_common(lst):
    data = Counter(lst)
    return max(lst, key=data.get)


def check_box_in_area(box_coord: list, area_coord: list):
    box_corners = (
        (box_coord[0], box_coord[1]),
        (box_coord[0], box_coord[3]),
        (box_coord[2], box_coord[1]),
        (box_coord[2], box_coord[3])
    )
    for box_corner in box_corners:
        if area_coord[0] < box_corner[0] <= area_coord[2] and area_coord[1] < box_corner[1] <= area_coord[3]:
            return True
    return False


def filter_boxes(main_item_coords, boxes_coords, area_coords=None):
    result = []
    for box_coord in boxes_coords:
        box_coord = transfer_coords(box_coord, np.array(main_item_coords).astype(np.int64))
        if check_box_in_area(box_coord[:4], area_coords):
            result.append(box_coord)
    return [len(result), result]


def convert_coords_from_dict_to_list(coords: dict) -> list:
    values = list(map(int, list(coords.values())))
    assert len(values) == 4
    return [values[0], values[2], values[1], values[3]]
