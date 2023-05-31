import httplib2
import numpy as np
import cv2


class HTTPLIB2Capture:
    def __init__(self, path, **kwargs):
        self.h = httplib2.Http(".cache")
        self.camera_url = path
        self.username = kwargs.get('username', None)
        self.password = kwargs.get('password', None)

    def get_snapshot(self):
        try:
            self.h.add_credentials(self.username, self.password)
            resp, content = self.h.request(
                self.camera_url, "GET", body="foobar")
            img_array = np.frombuffer(content, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            assert image is not None, 'Image Not Found ' + self.camera_url
            return image
        except Exception as e:
            print("Data exc - ", e)
            return None
