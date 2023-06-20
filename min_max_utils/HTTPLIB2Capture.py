import httplib2
import numpy as np
import cv2


class HTTPLIB2Capture:
    def __init__(self, path, **kwargs):
        self.h = httplib2.Http(".cache")
        self.camera_url = path
        self.username = 'admin'
        self.password = kwargs.get('password', None)
        self.logger = kwargs.get('logger')
        if self.username is None or self.password is None:
            self.logger.warning("Empty password or username")

    def get_snapshot(self):
        self.h.add_credentials(self.username, self.password)
        print(self.username, self.password)
        try:
            resp, content = self.h.request(
                self.camera_url, "GET", body="foobar")
            img_array = np.frombuffer(content, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return image
        except Exception:
            self.logger.warning("Empty image. Skipping iteration...")
            print('test 2')
            return None
