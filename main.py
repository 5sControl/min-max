from min_max_utils.HTTPLIB2Capture import HTTPLIB2Capture
from min_max_utils.min_max_utils import create_logger, convert_coords_from_dict_to_list
from min_max_models.ObjectDetectionModel import ObjDetectionModel
import warnings
import os
from run import run_min_max
from dotenv import load_dotenv
from confs.load_configs import *
import ast

load_dotenv("confs/settings.env")
warnings.filterwarnings("ignore")

areas = os.environ.get("areas")
username = os.environ.get("username")
password = os.environ.get("password")
server_url = os.environ.get("server_url")
source = os.environ.get("camera_url")
folder = os.environ.get("folder")
zones = os.environ.get("zones")

logger = create_logger()
areas = ast.literal_eval(areas)
zones = ast.literal_eval(zones)


box_model = ObjDetectionModel(
    BOX_MODEL_PATH,
    CONF_THRES,
    IOU_THRES,
    CLASSES
)
human_model = ObjDetectionModel(
    HUMAN_MODEL_PATH,
    CONF_THRES,
    IOU_THRES,
    CLASSES
)

dataset = HTTPLIB2Capture(source, username=username, password=password, logger=logger)

run_min_max(dataset, logger, human_model, box_model, areas, folder, server_url, zones)
