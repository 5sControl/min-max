import json


with open("confs/configs.json", "r") as conf:
    configs: dict = json.load(conf)

with open("model_image/flask_configs/flask_confs.json", "r") as flask_conf:
    configs.update(json.load(flask_conf))
