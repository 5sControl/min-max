import httplib2
import numpy as np
import cv2


class HTTPLIB2Capture:
    def __init__(self, path, **kwargs):
        self._http_connection = httplib2.Http(".cache")
        self._camera_url = path
        self._username = kwargs.get('username', None)
        self._password = kwargs.get('password', None)
        self._logger = kwargs.get('logger')
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
            return image
        except Exception:
            self._logger.warning("Empty image. Skipping iteration...")
            return None
