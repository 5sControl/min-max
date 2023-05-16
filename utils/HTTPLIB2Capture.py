from pathlib import Path
import httplib2
import numpy as np
import cv2
h = httplib2.Http(".cache")


class HTTPLIB2Capture:  # for inference
    def __init__(self, path, img_size=640, stride=32, username='username', password='password'):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        self.camera_url = path
        self.username = username
        self.password = password
        files = [path]

        images = [path]
        videos = []
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        try:
            path = self.camera_url

            self.count += 1
            h.add_credentials(self.username, self.password)
            resp, content = h.request(path, "GET", body="foobar")

            if resp.status == 200:
                # Decode the JPEG image data into a NumPy array
                nparr = np.frombuffer(content, np.uint8)
                img0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            else:
                print('Error: Could not retrieve image')
            assert img0 is not None, 'Image Not Found ' + path

            return path, None, img0, self.cap
        except:
            return None, None, None, None

    def __len__(self):
        return self.nf  # number of files
