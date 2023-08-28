import httplib2
import numpy as np
import cv2
from skimage.metrics import structural_similarity
from logging import Logger
import socketio


sio = socketio.AsyncClient()
images = {}


class ImageCapture:
    def __init__(self, path, **kwargs):
        self._camera_url = path
        self._username = kwargs.get('username')
        self._password = kwargs.get('password')
        self._logger : Logger = kwargs.get('logger')
        self._prev_img = None
        if not self._username or not self._password:
            self.logger.warning("Empty password or username")

    def get_snapshot(self) -> cv2.Mat:
        try:
            global images
            image = images[self._camera_url]
            img_array = np.frombuffer(image, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return image
        except Exception as exc:
            self._logger.warning(f"Empty image.\n {exc} \n Skipping iteration...")
            return None

@sio.event
async def connect():
    print("Connection")

@sio.event
async def snapshot_updated(data):
    camera_url, screen = data.get("camera_ip"), data.get("screenshot")
    global images
    images[camera_url] = screen

async def run_sio(url):
    await sio.connect(url)
    await sio.wait()
