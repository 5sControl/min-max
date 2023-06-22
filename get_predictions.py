import requests
import numpy as np
import cv2


PORT = 5000

def predict_human(img: np.array, server_url: str):
    response = requests.post(
            f"{server_url}:{PORT}/predict_human",
            json={
                "image": img.tolist()
            }
        )
    n_boxes = response.json().get('n_boxes')
    coordinates = np.array(response.json().get("coordinates"))
    return [n_boxes, coordinates]
    

def predict_boxes(img: np.array, server_url: str):
        response = requests.post(
            f"{server_url}:{PORT}/predict_boxes",
            json={
                "image": img.tolist()
            }
        )
        n_boxes = response.json().get('n_boxes')
        coordinates = np.array(response.json().get("coordinates"))
        return [n_boxes, coordinates]


