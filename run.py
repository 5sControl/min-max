from min_max_utils.HTTPLIB2Capture import HTTPLIB2Capture
from min_max_utils.min_max_utils import *
from min_max_models.ObjectDetectionModel import ObjDetectModel
from min_max_utils.img_process_utils import convert_image
import warnings
from collections import deque
import os
from dotenv import load_dotenv
from confs.load_configs import *
import ast


load_dotenv('confs/settings.env')
warnings.filterwarnings("ignore")


areas = os.environ.get("areas")
username = os.environ.get("username")
password = os.environ.get("password")
server_url = os.environ.get("server_url")
source = os.environ.get("camera_url")
folder = os.environ.get("folder")
print("areas - ", areas)

logger = create_logger()
areas = ast.literal_eval(areas)
history_length = 15
n_boxes_history = deque(maxlen=history_length)


box_model = ObjDetectModel(
    BOX_MODEL_PATH,
    DEVICE,
    CONF_THRES,
    IOU_THRES,
    CLASSES
)
human_model = ObjDetectModel(
    HUMAN_MODEL_PATH,
    DEVICE,
    CONF_THRES,
    IOU_THRES,
    CLASSES
)

stride = box_model.stride

dataset = HTTPLIB2Capture(source, stride=stride,
                          username=username, password=password)

is_human_was_detected = True
n_iters = 0
while True:
    img = dataset.get_snapshot()
    if not img:
        logger.warning("Empty image")
        continue
    n_iters += 1
    if n_iters % 20 == 0:
        logger.debug("20 detect iterations passed")
    img_for_human = img.copy()

    img_for_human = convert_image(img_for_human)
    is_human_in_area_now = human_model(img_for_human)[0] != 0

    if is_human_in_area_now:
        logger.debug("Human was detected")

    areas_stat = []
    n_items = 0

    for area_index, item in enumerate(areas.copy()):  # for each area
        counter = 0

        n_items += len(item['coords'])
        for subarr_idx, coord in enumerate(item['coords'].copy()):
            x1, y1, x2, y2 = list(
                map(round, (coord['x1'], coord['y1'], coord['x2'], coord['y2'])))
            if x1 == x2 or y1 == y2:
                logger.warning("Empty area")
                n_items -= 1
                drop_area(areas, area_index, item, subarr_idx)
                continue
            cropped_img = convert_image(
                img[
                    round(coord['y1']):round(coord['y2']),
                    round(coord['x1']):round(coord['x2'])
                ]
            )
            if is_human_was_detected and is_human_in_area_now:  # wait for human disappearing
                n_boxes_history.clear()
                areas_stat.clear()

            elif not is_human_was_detected and is_human_in_area_now:  # will start in next iter
                n_boxes_history.clear()
                areas_stat.clear()

            elif is_human_was_detected and not is_human_in_area_now:  # start counting
                logger.debug("Boxes counting was started")
                areas_stat.append(box_model(cropped_img))

            elif not is_human_was_detected and not is_human_in_area_now and \
                    len(n_boxes_history):
                logger.debug("Boxes counting...")
                areas_stat.append(box_model(cropped_img))

    is_human_was_detected = is_human_in_area_now

    if len(areas_stat) >= n_items:
        n_boxes_history.append([el[0] for el in areas_stat])

    if len(n_boxes_history) >= N_STEPS:
        coords = [el[1] for el in areas_stat]
        send_report(n_boxes_history, img, areas,
                    folder, logger, server_url, coords)
        n_boxes_history = deque(maxlen=history_length)
