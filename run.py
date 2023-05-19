from min_max_utils.HTTPLIB2Capture import HTTPLIB2Capture
from min_max_utils.min_max_utils import *
from min_max_models.ObjectDetectionModel import ObjDetectModel
from min_max_utils.img_process_utils import check_img_size, convert_image
import uuid
import warnings
from collections import deque
import os
import dotenv
from confs.load_configs import *
import ast


dotenv.load_dotenv("confs/settings.env")
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
img_size = check_img_size(IMG_SIZE, s=stride)

dataset = HTTPLIB2Capture(source, img_size=img_size, stride=stride,
                          username=username, password=password)

is_human_was_detected = True
n_iters = 0
while True:
    path, im0s = dataset.get_snapshot()
    if not path:
        logger.warning("Img path is none")
        continue
    n_iters += 1
    if n_iters % 20 == 0:
        logger.debug("20 detect iterations passed")
    im0 = im0s
    img_for_human = im0.copy()

    full_img = convert_image(img_for_human, img_size, stride, DEVICE)
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

            img = convert_image(crop_im, img_size, stride, DEVICE)
            if is_human_was_detected and is_human_in_area_now:  # wait for human disappearing
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

    if len(n_boxes_history) >= N_STEPS:
        send_report(n_boxes_history, im0, areas, folder, logger, server_url)
        n_boxes_history = deque(maxlen=history_length)
