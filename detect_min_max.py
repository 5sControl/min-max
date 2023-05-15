from pathlib import Path
import uuid
import cv2
import torch
import numpy as np
import json
import requests
import datetime
from models.experimental import attempt_load
import asyncio
import logging
import colorlog
from utils.visualization import draw_rect_with_text
from collections import deque


from utils.datasets import LoadImages, letterbox
from utils.general import check_img_size, non_max_suppression, set_logging, \
    increment_path
from utils.torch_utils import select_device, TracedModel



def create_logger():
    logger = logging.getLogger('my_logger')
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
    linesP = cv2.HoughLinesP(dst, rho=1, theta=np.pi / 180, threshold=61, lines=None, minLineLength=25, maxLineGap=10)
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
    
    if (x1 <= minX and x2 <= minX) or (y1 <= minY and y2 <= minY) or (x1 >= maxX and x2 >= maxX) or (y1 >= maxY and y2 >= maxY):
        return False

    m = (y2 - y1) / (x2 - x1 + 1e-3)

    y = m * (minX - x1) + y1
    if y > minY and y < maxY:
        return True

    y = m * (maxX - x1) + y1
    if y > minY and y < maxY:
        return True

    x = (minY - y1) / m + x1
    if x > minX and x < maxX:
        return True

    x = (maxY - y1) / m + x1
    if x > minX and x < maxX:
        return True
    
    return False
    


def conver_image(img_, img_size, stride, device):
    img = letterbox(img_, img_size, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img



async def run_min_max(areas, username, password, algorithm, ip_address, server_url, source, box_model_weights, view_img,
                      save_txt, img_size, human_model_weights, n_steps_without_human):
        print(areas, username, password, algorithm, ip_address, server_url, source, box_model_weights, view_img,
              save_txt, img_size, human_model_weights, n_steps_without_human)
        logger = create_logger()
        areas = json.loads(areas)
        history_length = 15
        n_boxes_history = deque(maxlen=history_length)

        opt = {"exist_ok": False, "name": "exp", "project": "runs/detect", "update": False,
               "augment": False, "agnostic_nms": False, "classes": [0], "nosave": True, "save_conf": False,
               "device": "cpu", "iou_thres": 0.45, "conf_thres": 0.4}

        save_dir = Path(increment_path(Path(opt['project']) / opt['name'], exist_ok=opt['exist_ok']))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        set_logging()
        device = select_device(opt['device'])
        box_model = attempt_load(box_model_weights, map_location=device)  
        human_model = attempt_load(human_model_weights, map_location=device)
        stride = int(box_model.stride.max()) 
        img_size = check_img_size(img_size, s=stride) 

        box_model = TracedModel(box_model, device)
        human_model = TracedModel(human_model, device)

        dataset = LoadImages(source, img_size=img_size, stride=stride, username=username, password=password)

        if device.type != 'cpu':
            box_model(
                torch.zeros(1, 3, img_size, img_size).to(device).type_as(next(box_model.parameters())))  # run once
        img_size = (640, 640)

        is_human_was_detected = True
        n_iters = 0
        for path, img, im0s, _ in dataset:
            if not path:
                await asyncio.sleep(1)
                continue
            await asyncio.sleep(1)
            n_iters += 1
            if n_iters % 20 == 0:
                logger.debug("20 detect iterations passed")
            im0 = im0s
            img_for_human = im0.copy()

            full_img = conver_image(img_for_human, img_size, stride, device)
            with torch.no_grad():  # Calculating gradients would cause a GPU memory leak

                pred_humans = human_model(full_img, augment=opt['augment'])[0]
                pred_humans = non_max_suppression(pred_humans, opt['conf_thres'], opt['iou_thres'],
                                                          classes=opt['classes'],
                                                          agnostic=opt['agnostic_nms'])
                is_human_in_area_now = bool(torch.numel(pred_humans[0]))

            if is_human_in_area_now:
                        logger.debug("Human was detected")

            num_boxes_per_area = []

            for area_index, item in enumerate(areas):  # for each area
                counter = 0
                itemid = item['itemId']
                try:
                    item_name = item['itemName']
                except Exception:
                    item_name = False
                item_image_name = str(uuid.uuid4())
                image_name_url = f'images/{ip_address[0]}/' + item_image_name + '.jpg'
                img_copy = im0.copy()

                rectangle_color = (41, 123, 255)
                text = f"Id: {itemid}"
                if item_name:
                    text = f"Id: {itemid}, Name: {item_name}"

                for coord in item['coords']:


                    crop_im = im0[
                              round(coord['y1']):round(coord['y2']),
                              round(coord['x1']):round(coord['x2'])
                    ]
                    
                    img = conver_image(crop_im, img_size, stride, device)
                    if is_human_was_detected and is_human_in_area_now:  # wait for human dis
                        n_boxes_history.clear()
                        num_boxes_per_area.clear()

                    elif not is_human_was_detected and is_human_in_area_now:  # will start in next iter
                        n_boxes_history.clear()
                        num_boxes_per_area.clear()

                    elif is_human_was_detected and not is_human_in_area_now:  # start counting
                        logger.debug("Boxes counting was started")
                        with torch.no_grad():
                            pred_boxes = box_model(img, augment=opt['augment'])[0]
                            pred_boxes = non_max_suppression(pred_boxes, opt['conf_thres'],
                                                            opt['iou_thres'],
                                                            classes=opt['classes'],
                                                            agnostic=opt['agnostic_nms']
                                                            )
                        for det in pred_boxes:
                            counter += len(det)
                        num_boxes_per_area.append(counter)

                    elif not is_human_was_detected and not is_human_in_area_now and\
                        len(n_boxes_history):
                        logger.debug("Boxes counting...")
                        with torch.no_grad():
                            pred_boxes = box_model(img, augment=opt['augment'])[0]
                            pred_boxes = non_max_suppression(pred_boxes, opt['conf_thres'], opt['iou_thres'],
                                                            classes=opt['classes'],
                                                            agnostic=opt['agnostic_nms'])
                        for det in pred_boxes:
                            counter += len(det)
                        num_boxes_per_area.append(counter)

            is_human_was_detected = is_human_in_area_now

            if len(num_boxes_per_area) == len(areas):
                n_boxes_history.append(num_boxes_per_area)

            if len(n_boxes_history) >= n_steps_without_human:  
                red_lines = find_red_line(im0)
                report = []
                n_boxes_history = np.array(n_boxes_history).mean(axis=0).round().astype(int)
                for area_index, item in enumerate(areas):
                    itemid = item['itemId']

                    try:
                        item_name = item['itemName']
                    except Exception:
                        item_name = False

                    image_name_url = f'images/{ip_address[0]}/' + str(uuid.uuid4()) + '.jpg'
                    img_copy = im0.copy()
                    img_rect = im0.copy()

                    rectangle_color = (41, 255, 26)
                    text = f"Id: {itemid}"
                    if item_name:
                        text = f"Id: {itemid}, Name: {item_name}"

                    for coord in item['coords']:
                        x1, x2, y1, y2 = tuple(map(round, coord.values()))
                        img_rect = draw_rect_with_text(
                            img_rect,
                            (x1, y1, x2, y2), 
                            text,
                            rectangle_color,
                            thickness=2
                        )

                        crop_im = im0[
                                round(coord['y1']):round(coord['y2']),
                                round(coord['x1']):round(coord['x2'])
                        ]
                        is_red_line = False
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
                    cv2.imwrite(image_name_url, img_rect)
                    
                save_photo_url = f'images/{ip_address[0]}/' + str(uuid.uuid4()) + '.jpg'
                cv2.imwrite(save_photo_url, im0)
                photo_start = {
                    'url': save_photo_url,
                    'date': datetime.datetime.now()
                }
                report_for_send = {
                    'camera': ip_address[0],
                    'algorithm': algorithm[0],
                    'start_tracking': str(photo_start['date']),
                    'stop_tracking': str(photo_start['date']),
                    'photos': [{'image': str(photo_start['url']), 'date': str(photo_start['date'])}],
                    'violation_found': False,
                    'extra': report
                }

                logger.info(
                    '\n'.join(['<<<<<<<<<<<<<<<<<SEND REPORT!!!!!!!>>>>>>>>>>>>>>',
                               str(report_for_send),
                               f'http://{server_url[0]}:8000/api/reports/report-with-photos/'])
                )
                try:
                    requests.post(url=f'http://{server_url[0]}:80/api/reports/report-with-photos/', json=report_for_send)
                except Exception as exc:
                    print(exc, 'req exc')
                    pass
                n_boxes_history = deque(maxlen=history_length)  # clear history for next iteration

