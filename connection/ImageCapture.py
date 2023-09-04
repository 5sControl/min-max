import numpy as np
import cv2
from logging import Logger
import requests


class ImageCapture:
    def __init__(self, path, **kwargs):
        self._camera_ip = path
        self._username = kwargs.get('username')
        self._password = kwargs.get('password')
        self._logger : Logger = kwargs.get('logger')
        self._server_url = kwargs.get("server_url")
        self._prev_img = None
        if not self._username or not self._password:
            self.logger.warning("Empty password or username")

    def get_snapshot(self) -> cv2.Mat:
        try:
            resp = requests.get(f"http://{self._server_url}:7777/get_snapshot", params={"camera_ip": self._camera_ip})
            image = resp.content
            img_array = np.frombuffer(image, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return image
        except Exception as exc:
            self._logger.warning(f"Empty image.\n {exc} \n Skipping iteration...")
            return None
