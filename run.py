from min_max_utils.visualization import draw_rect_with_text
from min_max_utils.HTTPLIB2Capture import HTTPLIB2Capture
from min_max_utils.min_max_utils import *
from ObjectDetectionModel import ObjDetectModel
import datetime
import uuid
import warnings
from collections import deque
import requests
import os
import json
import ast
import dotenv


dotenv.load_dotenv("confs/settings.env")
warnings.filterwarnings("ignore")


areas = os.environ.get("areas")
username = os.environ.get("username")
password = os.environ.get("password")
server_url = os.environ.get("server_url")
box_model_weights = "min_max_v0.2.6.pt"
human_model_weights = "yolov7.pt"
img_size = 640
n_steps = 5
source = os.environ.get("camera_url")
folder = os.environ.get("folder")
print("areas - ", areas)

logger = create_logger()
areas = ast.literal_eval(areas)
history_length = 15
n_boxes_history = deque(maxlen=history_length)

with open("confs/configs.json", "r") as conf:
    opt = json.load(conf)

device = select_device(opt['device'])


box_model = ObjDetectModel(
    box_model_weights,
    device,
    opt['conf_thres'],
    opt['iou_thres'],
    opt['classes']
)
human_model = ObjDetectModel(
    human_model_weights,
    device,
    opt['conf_thres'],
    opt['iou_thres'],
    opt['classes']
)

stride = box_model.stride
img_size = check_img_size(img_size, s=stride)

dataset = HTTPLIB2Capture(source, img_size=img_size, stride=stride,
                          username=username, password=password)

is_human_was_detected = True
n_iters = 0
while (True):
    path, im0s = dataset.get_snapshot()
    if not path:
        logger.warning("Img path is none")
        continue
    n_iters += 1
    if n_iters % 20 == 0:
        logger.debug("20 detect iterations passed")
    im0 = im0s
    img_for_human = im0.copy()

    full_img = convert_image(img_for_human, img_size, stride, device)
    is_human_in_area_now = human_model(full_img) != 0

    if is_human_in_area_now:
        logger.debug("Human was detected")

    num_boxes_per_area = []
    n_items = 0

    for area_index, item in enumerate(areas):  # for each area
        counter = 0
        itemid = item['itemId']
        try:
            item_name = item['itemName']
        except Exception:
            item_name = False
        item_image_name = str(uuid.uuid4())
        image_name_url = folder + '/' + item_image_name + '.jpg'
        img_copy = im0.copy()

        rectangle_color = (41, 123, 255)
        text = f"Id: {itemid}"
        if item_name:
            text = f"Id: {itemid}, Name: {item_name}"
        n_items += len(item['coords'])
        for coord in item['coords']:

            crop_im = im0[
                round(coord['y1']):round(coord['y2']),
                round(coord['x1']):round(coord['x2'])
            ]

            img = convert_image(crop_im, img_size, stride, device)
            if is_human_was_detected and is_human_in_area_now:  # wait for human dis
                n_boxes_history.clear()
                num_boxes_per_area.clear()

            elif not is_human_was_detected and is_human_in_area_now:  # will start in next iter
                n_boxes_history.clear()
                num_boxes_per_area.clear()

            elif is_human_was_detected and not is_human_in_area_now:  # start counting
                logger.debug("Boxes counting was started")
                num_boxes_per_area.append(box_model(img))

            elif not is_human_was_detected and not is_human_in_area_now and \
                    len(n_boxes_history):
                logger.debug("Boxes counting...")
                num_boxes_per_area.append(box_model(img))

    is_human_was_detected = is_human_in_area_now

    if len(num_boxes_per_area) >= n_items:
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

            image_name_url = folder + '/' + \
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
            if not os.path.exists(folder):
                os.makedirs(folder)
            cv2.imwrite(image_name_url, img_rect)
        save_photo_url = folder + '/' + \
            str(uuid.uuid4()) + '.jpg'
        cv2.imwrite(save_photo_url, im0)
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
            print(exc, 'req exc')
            pass
        # clear history for next iteration
        n_boxes_history = deque(maxlen=history_length)
