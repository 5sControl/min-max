from pathlib import Path
import httplib2
import numpy as np
import cv2
h = httplib2.Http(".cache")


class HTTPLIB2Capture:  # for inference
    def __init__(self, path, **kwargs):
        self.camera_url = path
        self.username = kwargs.get('username', None)
        self.password = kwargs.get('password', None)

    def get_snapshot(self):
        try:
            h.add_credentials(self.username, self.password)
            resp, content = h.request(self.camera_url, "GET", body="foobar")
            nparr = np.frombuffer(content, np.uint8)
            img0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            assert img0 is not None, 'Image Not Found ' + self.camera_url
            return self.camera_url, img0
        except Exception as e:
            print("Data exc - ", e)
            return None, None
