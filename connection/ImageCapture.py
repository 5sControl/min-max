import httplib2
import numpy as np
import cv2
from skimage.metrics import structural_similarity
from logging import Logger
import socketio


sio = socketio.Client()


class ImageCapture:
    def __init__(self, path, **kwargs):
        self._camera_url = path
        self._username = kwargs.get('username', None)
        self._password = kwargs.get('password', None)
        self._logger : Logger = kwargs.get('logger')
        self._prev_img = None

        global sio
        sio.connect(kwargs.get("server_url"))
        sio.wait()

        if not self._username or not self._password:
            self.logger.warning("Empty password or username")

    @sio.on('snapshot_updated')
    def handle_server_data(data):
        return data.get('camera_ip'), data.get('screenshot')

    def get_snapshot(self) -> cv2.Mat:
        self._http_connection.add_credentials(self._username, self._password)
        try:
            print(self.handle_server_data())
            exit(1)
            img_array = np.frombuffer(content, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if self._prev_img is not None:
                self._prev_img = cv2.cvtColor(self._prev_img, cv2.COLOR_BGR2GRAY)
                image_cp = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
                ssim_value = structural_similarity(self._prev_img, image_cp, full=True)[0] * 100
                self._prev_img = image
            else:
                ssim_value = 0. 
            return image, ssim_value
        except Exception as exc:
            self._logger.warning(f"Empty image.\n {exc} \n Skipping iteration...")
            return None, None
