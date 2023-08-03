import requests
import numpy as np
from logging import Logger
from PIL import Image
import io


PORT = 5000


class ModelPredictionsReceiver:
    def __init__(self, server_url: str, logger: Logger) -> None:
        self._server_url = server_url
        self._logger = logger

    @staticmethod
    def _convert_image2bytes(image: np.array, format='PNG') -> io.BytesIO:
        pil_image = Image.fromarray(image)
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format=format)
        img_byte_arr.seek(0)
        return img_byte_arr

    def predict_human(self, img: np.array):
        try:
            response = requests.post(
                f"{self._server_url}:{PORT}/predict_human",
                files={
                    "image": ("image", self._convert_image2bytes(img), "image/png")
                }
            )
            response.raise_for_status()
            return np.array(response.json().get("coordinates"))
        except Exception as exc:
            self._logger.critical("Cannot send request to model server. Error - {}".format(exc))
    
    def predict_bottles(self, img: np.array):
        try:
            self._logger.debug("Sending request to model server")
            response = requests.post(
                f"{self._server_url}:{PORT}/predict_bottles",
                files={
                    "image": ("image", self._convert_image2bytes(img), "image/png")
                }
            )
            response.raise_for_status()
            self._logger.debug("Bottles preds received")
            return np.array(response.json().get("coordinates"))
        except Exception as exc:
            self._logger.critical("Cannot send request to model server. Error - {}".format(exc))

    def predict_boxes(self, img: np.array):
        try:
            self._logger.debug("Sending request to model server")
            response = requests.post(
                f"{self._server_url}:{PORT}/predict_boxes",
                files={
                    "image": ("image", self._convert_image2bytes(img), "image/png")
                }
            )
            response.raise_for_status()
            self._logger.debug("Boxes preds received")
            return np.array(response.json().get("coordinates"))
        except Exception as exc:
            self._logger.critical("Cannot send request to model server. Error - {}".format(exc))


