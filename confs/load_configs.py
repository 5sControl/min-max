import json
from min_max_utils.torch_utils import select_device


with open("confs/configs.json", "r") as conf:
    configs = json.load(conf)
    CONF_THRES = configs.get("conf_thres")
    IOU_THRES = configs.get("iou_thres")
    BOX_MODEL_PATH = configs.get("box_detect_model")
    HUMAN_MODEL_PATH = configs.get("human_detect_model")
    CLASSES = configs.get("classes")
    DEVICE = select_device(configs.get("device"))
    IMG_SIZE = configs.get("img_size")
    N_STEPS = configs.get("n_steps")
