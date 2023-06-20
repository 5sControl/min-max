import requests
import numpy as np
import cv2


PORT = 3000

def predict_human(img: np.array):
    cv2.imwrite("img.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    with open("img.jpg", 'rb') as img_file:
        response = requests.post(
            f"http://localhost:{PORT}/predict_human",
            files={
                "image": img_file
            }
        )
        n_boxes = response.json().get('n_boxes')
        coordinates = response.json().get("coordinates")
        return [n_boxes, coordinates]
    

def predict_boxes(img: np.array):
    cv2.imwrite("img.jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    with open("img.jpg", 'rb') as img_file:
        response = requests.post(
            f"http://localhost:{PORT}/predict_boxes",
            files={
                "image": img_file
            }
        )
        n_boxes = response.json().get('n_boxes')
        coordinates = response.json().get("coordinates")
        return [n_boxes, coordinates]


