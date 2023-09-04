from connection import ImageCapture
from min_max_utils.min_max_utils import create_logger
import warnings
import os
from dotenv import load_dotenv
from MinMaxAlgorithm import MinMaxAlgorithm
import json


warnings.filterwarnings("ignore")

if os.environ.get("extra") is None:
    load_dotenv("confs/settings.env")
extra: str = os.environ.get("extra")
extra = json.loads(extra)[0]
areas = extra.get("areas")
zones = extra.get("zones")

username = os.environ.get("username")
password = os.environ.get("password")
server_url = os.environ.get("server_url")
camera_ip = os.environ.get("camera_ip")
folder = os.environ.get("folder")

logger = create_logger()

dataset = ImageCapture(
    camera_ip,
    username=username, 
    password=password, 
    logger=logger,
    server_url=server_url
)

algo = MinMaxAlgorithm(dataset, logger, areas, folder, server_url, zones)
algo.start()
