import json


with open("flask_configs/flask_confs.json", "r") as conf:
    configs = json.load(conf)
    box_model_configs = configs["box_model"]
    human_model_configs = configs["human_model"]
    bottle_model_configs = configs["bottle_model"]
