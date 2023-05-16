import json


opt = {"exist_ok": False, "name": "exp", "project": "runs/detect", "update": False,
       "augment": False, "agnostic_nms": False, "classes": [0], "nosave": True, "save_conf": False,
       "device": "cpu", "iou_thres": 0.45, "conf_thres": 0.4}

with open("confs/configs.json", "w") as conf:
    json.dump(opt, conf)

with open("confs/configs.json", "r") as conf:
    print(json.load(conf))
