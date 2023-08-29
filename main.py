from connection import ImageCapture, run_sio
from min_max_utils.min_max_utils import create_logger
import warnings
import os
from dotenv import load_dotenv
from MinMaxAlgorithm import MinMaxAlgorithm
import json
import asyncio


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
source = os.environ.get("camera_ip")
folder = os.environ.get("folder")

logger = create_logger()

dataset = ImageCapture(
    source,
    username=username, 
    password=password, 
    logger=logger
)

algo = MinMaxAlgorithm(dataset, logger, areas, folder, server_url, zones)
algo.start()


async def main():
    await asyncio.gather(run_sio(server_url + ':3456'), algo.start())


loop = asyncio.get_event_loop()
try:
    loop.run_until_complete(main())
except Exception as exc:
    print(exc)
finally:
    loop.close()
