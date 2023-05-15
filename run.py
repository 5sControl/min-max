from utils.visualization import draw_rect_with_text
from models.experimental import attempt_load
from utils.min_max_utils import *
from utils.torch_utils import select_device, TracedModel
from utils.general import check_img_size, non_max_suppression, set_logging
import datetime
import uuid
import warnings
from collections import deque
from pathlib import Path
import requests
import os
import ast
import time

warnings.filterwarnings("ignore")


areas = os.environ.get("AREAS")
username = os.environ.get("USERNAME")
password = os.environ.get("PASSWORD")
ip_address = os.environ.get("IP_ADDRESS")
server_url = os.environ.get("SERVER_URL")
box_model_weights = os.environ.get("BOX_MODEL")
human_model_weights = os.environ.get("HUMAN_MODEL")
img_size = os.environ.get("IMG_SIZE")
n_steps = os.environ.get("N_STEPS")
source = os.environ.get("SOURCE")


logger = create_logger()
areas = ast.literal_eval(areas)
history_length = 15
n_boxes_history = deque(maxlen=history_length)

opt = {"exist_ok": False, "name": "exp", "project": "runs/detect", "update": False,
       "augment": False, "agnostic_nms": False, "classes": [0], "nosave": True, "save_conf": False,
       "device": "cpu", "iou_thres": 0.45, "conf_thres": 0.4}


set_logging()
device = select_device(opt['device'])
box_model = attempt_load(box_model_weights, map_location=device)
human_model = attempt_load(human_model_weights, map_location=device)
stride = int(box_model.stride.max())
img_size = check_img_size(img_size, s=stride)

box_model = TracedModel(box_model, device)
human_model = TracedModel(human_model, device)

dataset = LoadImages(source, img_size=img_size, stride=stride,
                     username=username, password=password)

if device.type != 'cpu':
    box_model(
        torch.zeros(1, 3, img_size, img_size).to(
            device).type_as(next(box_model.parameters()))
    )  # run once

is_human_was_detected = True
n_iters = 0
for path, img, im0s, _ in dataset:
    if not path:
        time.sleep(1)
        continue
    time.sleep(1)
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
                print(img)
                for det in pred_boxes:
                    counter += len(det)
                num_boxes_per_area.append(counter)

            elif not is_human_was_detected and not is_human_in_area_now and \
                    len(n_boxes_history):
                logger.debug("Boxes counting...")
                with torch.no_grad():
                    pred_boxes = box_model(img, augment=opt['augment'])[0]
                    pred_boxes = non_max_suppression(pred_boxes, opt['conf_thres'], opt['iou_thres'],
                                                     classes=opt['classes'],
                                                     agnostic=opt['agnostic_nms'])
                print(img)
                for det in pred_boxes:
                    counter += len(det)
                    print(len(det))
                    print("____")
                num_boxes_per_area.append(counter)

    is_human_was_detected = is_human_in_area_now

    if len(num_boxes_per_area) == len(areas):
        n_boxes_history.append(num_boxes_per_area)

    if len(n_boxes_history) >= n_steps:
        red_lines = find_red_line(im0)
        report = []
        n_boxes_history = np.array(n_boxes_history).mean(
            axis=0).round().astype(int)
        for area_index, item in enumerate(areas):
            itemid = item['itemId']

            try:
                item_name = item['itemName']
            except Exception:
                item_name = False

            image_name_url = f'images/{ip_address[0]}/' + \
                str(uuid.uuid4()) + '.jpg'
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

        save_photo_url = f'images/{ip_address[0]}/' + \
            str(uuid.uuid4()) + '.jpg'
        cv2.imwrite(save_photo_url, im0)
        photo_start = {
            'url': save_photo_url,
            'date': datetime.datetime.now()
        }
        report_for_send = {
            'camera': ip_address[0],
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
                       f'http://{server_url[0]}:8000/api/reports/report-with-photos/'])
        )
        try:
            requests.post(
                url=f'http://{server_url[0]}:80/api/reports/report-with-photos/', json=report_for_send)
        except Exception as exc:
            print(exc, 'req exc')
            pass
        # clear history for next iteration
        n_boxes_history = deque(maxlen=history_length)
