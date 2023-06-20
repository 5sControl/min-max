import json


with open("confs/configs.json", "r") as conf:
    configs = json.load(conf)
    N_STEPS = configs.get("n_steps")
    DEBUG_FOLDER = configs.get("debug_folder")
