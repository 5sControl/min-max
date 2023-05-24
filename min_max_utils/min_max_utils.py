import logging
import uuid
import datetime
import colorlog
import cv2
import numpy as np
import os
import requests
from min_max_utils.visualization_utils import draw_rect_with_text


def drop_area(areas: list[dict], item_idx: int, item: dict[list], subarea_idx: int):
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
    lower_red = np.array([0, 100, 50])
    upper_red = np.array([8, 255, 255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([173, 50, 50])
    upper_red = np.array([179, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    # join my masks
    mask = mask0 + mask1
    output_img = img.copy()
    output_img[np.where(mask == 0)] = 0
    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask == 0)] = 0
    src = cv2.cvtColor(output_img, cv2.COLOR_HSV2RGB)
    src = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    dst = cv2.Canny(src, 50, 200, None, 3)
    linesP = cv2.HoughLinesP(dst, rho=1, theta=np.pi / 180,
                             threshold=61, lines=None, minLineLength=25, maxLineGap=10)
    lines = []
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            if abs(l[1] - l[3]) < 25:
                lines.append(l)
    return lines


def is_line_in_area(area, line):
    x1, y1, x2, y2 = line
    minX, minY, maxX, maxY = area

    if (x1 <= minX and x2 <= minX) or (y1 <= minY and y2 <= minY) or (x1 >= maxX and x2 >= maxX) or (
            y1 >= maxY and y2 >= maxY):
        return False

    m = (y2 - y1) / (x2 - x1 + 1e-3)

    y = m * (minX - x1) + y1
    if minY < y < maxY:
        return True

    y = m * (maxX - x1) + y1
    if minY < y < maxY:
        return True

    x = (minY - y1) / m + x1
    if minX < x < maxX:
        return True

    x = (maxY - y1) / m + x1
    if minX < x < maxX:
        return True

    return False


def send_report(n_boxes_history, img, areas, folder, logger, server_url, boxes_coords, img_shapes):
    red_lines = find_red_line(img)
    report = []
    n_boxes_history = np.array(n_boxes_history).mean(
        axis=0).round().astype(int)
    for area_index, item in enumerate(areas):
        itemid = item['itemId']

        try:
            item_name = item['itemName']
        except Exception:
            item_name = False

        image_name_url = folder + '/' + str(uuid.uuid4()) + '.jpg'
        img_copy = img.copy()
        img_rect = img.copy()

        rectangle_color = (41, 255, 26)
        text = f"Id: {itemid}"
        if item_name:
            text = f"Id: {itemid}, Name: {item_name}"
        is_red_line = False
        for coord in item['coords']:
            x1, x2, y1, y2 = tuple(map(round, coord.values()))
            img_rect = draw_rect_with_text(
                img_rect,
                (x1, y1, x2, y2),
                text,
                rectangle_color,
                thickness=2
            )
            for bbox_coords in boxes_coords[area_index]:
                pass

            crop_im = img[
                round(coord['y1']):round(coord['y2']),
                round(coord['x1']):round(coord['x2'])
            ]
            for line in red_lines:
                if is_line_in_area((coord['x1'], coord['y1'], coord['x2'], coord['y2']), line):
                    is_red_line = True
                    break
        mean_val = n_boxes_history[area_index]
        report.append(
            {
                "itemId": itemid,
                "count": int(mean_val),
                "image_item": image_name_url,
                "low_stock_level": is_red_line
            }
        )
        if not os.path.exists(folder):
            os.makedirs(folder)
        cv2.imwrite(image_name_url, img_rect)
    save_photo_url = folder + '/' + str(uuid.uuid4()) + '.jpg'
    cv2.imwrite(save_photo_url, img)
    photo_start = {
        'url': save_photo_url,
        'date': datetime.datetime.now()
    }
    report_for_send = {
        'camera': os.path.basename(folder),
        'algorithm': "min_max_control",
        'start_tracking': str(photo_start['date']),
        'stop_tracking': str(photo_start['date']),
        'photos': [{'image': str(photo_start['url']), 'date': str(photo_start['date'])}],
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
