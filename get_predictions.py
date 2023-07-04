import requests
import numpy as np
from logging import Logger

PORT = 5000

def predict_human(img: np.array, server_url: str, logger: Logger):
    response = requests.post(
            f"{server_url}:{PORT}/predict_human",
            json={
                "image": img.tolist()
            }
        )
    status_code = response.status_code
    if status_code == 200:
        n_boxes = response.json().get('n_boxes')
        coordinates = np.array(response.json().get("coordinates"))
    else:
         logger.warning(
              "Response code = {}.\n response = {}".format(status_code, response)
         )
         n_boxes = None
         coordinates = None
    return [n_boxes, coordinates]
    
def predict_bottles(img: np.array, server_url: str, logger: Logger):
    response = requests.post(
            f"{server_url}:{PORT}/predict_bottles",
            json={
                "image": img.tolist()
            }
        )
    status_code = response.status_code
    if status_code == 200:
        n_bottles = response.json().get('n_items')
        coordinates = np.array(response.json().get("coordinates"))
    else:
         logger.warning(
              "Response code = {}.\n JSON = {}".format(status_code, response.json())
         )
         n_bottles = None
         coordinates = None
    return [n_bottles, coordinates]

def predict_boxes(img: np.array, server_url: str, logger: Logger):
        response = requests.post(
            f"{server_url}:{PORT}/predict_boxes",
            json={
                "image": img.tolist()
            }
        )
        status_code = response.status_code
        if status_code == 200:
            n_boxes = response.json().get('n_items')
            coordinates = np.array(response.json().get("coordinates"))
        else:
            logger.warning(
                "Response code = {}.\n response = {}".format(status_code, response)
            )
            n_boxes = None
            coordinates = None
        return [n_boxes, coordinates]


