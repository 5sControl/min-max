import httplib2
import numpy as np
import cv2
from skimage.metrics import structural_similarity
from logging import Logger


class HTTPLIB2Capture:
    def __init__(self, path, **kwargs):
        self._http_connection = httplib2.Http(".cache")
        self._camera_url = path
        self._username = kwargs.get('username', None)
        self._password = kwargs.get('password', None)
        self._logger : Logger = kwargs.get('logger')
        self._prev_img = None
        if not self._username or not self._password:
            self.logger.warning("Empty password or username")

    def get_snapshot(self) -> cv2.Mat:
        self._http_connection.add_credentials(self._username, self._password)
        try:
            resp, content = self._http_connection.request(
                self._camera_url, 
                "GET", 
                body="foobar"
            )
            img_array = np.frombuffer(content, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if self._prev_img is not None:
                self._prev_img = cv2.cvtColor(self._prev_img, cv2.COLOR_BGR2GRAY)
                image_cp = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
                ssim_value = structural_similarity(self._prev_img, image_cp, full=True)[0] * 100
                self._prev_img = image
                if ssim_value > 85:
                    self._logger.debug("Similar images. Skipping...")
                    return None
            return image
        except Exception as exc:
            self._logger.warning(f"Empty image.\n {exc} \n Skipping iteration...")
            return None
