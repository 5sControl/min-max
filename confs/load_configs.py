import json


with open("confs/configs.json", "r") as conf:
    configs = json.load(conf)
    CONF_THRES = configs.get("conf_thres")
    IOU_THRES = configs.get("iou_thres")
    BOX_MODEL_PATH = configs.get("box_detect_model")
    HUMAN_MODEL_PATH = configs.get("human_detect_model")
    CLASSES = configs.get("classes")
    IMG_SIZE = configs.get("img_size")
    N_STEPS = configs.get("n_steps")
