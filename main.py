from connection import HTTPLIB2Capture
from min_max_utils.min_max_utils import create_logger
import warnings
import os
from dotenv import load_dotenv
from run import MinMaxAlgorithm
from confs.load_configs import *


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
source = os.environ.get("camera_url")
folder = os.environ.get("folder")

logger = create_logger()

dataset = HTTPLIB2Capture(source, username=username, password=password, logger=logger)

algo = MinMaxAlgorithm(dataset, logger, areas, folder, server_url, zones)
algo.start()