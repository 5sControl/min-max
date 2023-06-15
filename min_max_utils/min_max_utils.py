import json
from collections import Counter
import logging
import uuid
import datetime
import colorlog
import cv2
import numpy as np
import os
import requests
from min_max_utils.visualization_utils import draw_rect, draw_text, draw_line
from min_max_utils.img_process_utils import save_image, transfer_coords


def drop_area(areas: list[dict], item_idx: int, item: dict, subarea_idx: int):
    logger = logging.getLogger('min_max_logger')
    if len(item['coords']) == 1:
        logger.info("Item was dropped - {}".format(areas.pop(item_idx)))
    else:
        logger.info(
            "Subarea was dropped - {}".format(item.get('coords').pop(subarea_idx)))


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
    lower_red = np.array([0, 135, 50])
    upper_red = np.array([7, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170, 70, 50])
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

    dst = cv2.Canny(src, 150, 250, None, 3)
    lines_p = cv2.HoughLinesP(dst, rho=1, theta=np.pi / 180,
                              threshold=50, lines=None, minLineLength=15, maxLineGap=10)
    lines = []
    if lines_p is not None:
        for i in range(0, len(lines_p)):
            line = lines_p[i][0]
            if abs(line[1] - line[3]) < 100:
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


def check_box_in_area(box_coord, area_coord):
    box_center = ((box_coord[0] + box_coord[2]) / 2, (box_coord[1] + box_coord[3]) / 2)
    if area_coord[0] < box_center[0] <= area_coord[2] and area_coord[1] < box_center[1] <= area_coord[3]:
        return True
    return False


def filter_boxes(main_item_coords, _, boxes_coords, area_coords=None, check=True):
    result = []
    for box_coord in boxes_coords:
        box_coord = transfer_coords(box_coord, main_item_coords)
        if not check:
            result.append(box_coord)
        elif check and check_box_in_area(box_coord[:4], area_coords):
            result.append(box_coord)
    return [len(result), result]


def convert_coords_from_dict_to_list(coords: dict) -> list:
    values = list(map(int, list(coords.values())))
    assert len(values) == 4
    return [values[0], values[2], values[1], values[3]]


def send_report(n_boxes_history, img, areas, folder, logger, server_url, boxes_coords, zones):
    red_lines = find_red_line(img)
    report = []
    if not zones:
        for item in areas:
            item['zoneId'] = None
        zones.append(
            {
                "zoneId": None,
                "zoneName": None,
                "coords":
                    [{
                        "x1": 0,
                        "x2": 1920,
                        "y1": 0,
                        "y2": 1080
                }],
                "items": []
            }
        )
    for zone in zones:
        zone_dict = {'zoneId': zone['zoneId'], 'zoneName': zone['zoneName'], 'items': []}
        report.append(zone_dict)
    for item_index, item in enumerate(areas):
        itemid = item['itemId']

        item_name = item['itemName']

        image_name_url = folder + '/' + str(uuid.uuid4()) + '.png'
        img_rect = img.copy()
        rectangle_color = (0, 102, 204)
        for zone in zones:
            img_rect = draw_rect(img_rect, convert_coords_from_dict_to_list(zone.get("coords")[0]), rectangle_color)

        is_red_line_in_item = False

        for subarr_idx, coord in enumerate(item['coords']):
            area_coords = convert_coords_from_dict_to_list(coord)

            is_red_line_in_subarea = False

            for line in red_lines:
                if is_line_in_area(area_coords, line):
                    img_rect = draw_line(img_rect, line, area_coords, thickness=4)
                    is_red_line_in_subarea = is_red_line_in_item = True

            text_item = f"{item_name}: {n_boxes_history[item_index][subarr_idx] if not is_red_line_in_subarea else 'low stock level'}"

            img_rect = draw_rect(img_rect, area_coords, rectangle_color, thickness=2)

            for idx, bbox_coords in enumerate(boxes_coords[item_index][subarr_idx]):
                text = str(round(float(bbox_coords[4]), 2))

                img_rect = draw_rect(img_rect, bbox_coords[:4], (255, 51, 255), thickness=2)
                img_rect = draw_text(img_rect, bbox_coords[:4], text, (255, 255, 255), proba=True)

            img_rect = draw_text(img_rect, area_coords, text_item, (255, 255, 255), proba=False)

        for idx, zone in enumerate(zones):
            if zone.get("zoneId") == item.get("zoneId"):
                report[idx]['items'].append(
                    {
                        "itemId": itemid,
                        "count": sum(n_boxes_history[item_index]),
                        "image": image_name_url,
                        "low_stock_level": is_red_line_in_item,
                    }
                )
        if not os.path.exists(folder):
            os.makedirs(folder)
        save_image(img_rect, image_name_url)
    photo_start = {
        'date': datetime.datetime.now()
    }
    report_for_send = {
        'camera': os.path.basename(folder),
        'algorithm': "min_max_control",
        'start_tracking': str(photo_start['date']),
        'stop_tracking': str(photo_start['date']),
        'photos': [{'date': str(photo_start['date'])}],
        'violation_found': False,
        'extra': report
    }

    logger.info(
        '\n'.join(['<<<<<<<<<<<<<<<<<SEND REPORT!!!!!!!>>>>>>>>>>>>>>',
                   str(report_for_send),
                   f'{server_url}:8000/api/reports/report-with-photos/'])
    )

    try:
        requests.post(
            url=f'{server_url}:80/api/reports/report-with-photos/', json=report_for_send)
    except Exception as exc:
        logger.error("Error while sending report occurred: {}".format(exc))
