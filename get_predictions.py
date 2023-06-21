import requests
import numpy as np
import cv2


PORT = 8888

def predict_human(img: np.array, server_url: str):
    cv2.imwrite("img.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    response = requests.post(
            f"{server_url}:{PORT}/predict_human",
            files={
                "image": open("img.jpg", 'rb')
            }
        )
    n_boxes = response.json().get('n_boxes')
    coordinates = np.array(response.json().get("coordinates"))
    return [n_boxes, coordinates]
    

def predict_boxes(img: np.array, server_url: str):
    cv2.imwrite("images/img.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    with open("img.jpg", 'rb') as img_file:
        response = requests.post(
            f"{server_url}:{PORT}/predict_boxes",
            files={
                "image": img_file
            }
        )
        n_boxes = response.json().get('n_boxes')
        coordinates = np.array(response.json().get("coordinates"))
        return [n_boxes, coordinates]


