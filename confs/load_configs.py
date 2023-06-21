import json


with open("confs/configs.json", "r") as conf:
    configs = json.load(conf)
    N_STEPS = configs.get("n_steps")
    DEBUG_FOLDER = configs.get("debug_folder")

#with open("model_image/flask_configs/flask_confs.json", "r") as flask_conf:
    #configs = json.load(conf)
    #PORT = configs.get("port")
