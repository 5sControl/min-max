import os
import yolov7
import numpy as np
import cv2
import httplib2
import time

username = os.environ.get("username")
password = os.environ.get("password")
camera_url = os.environ.get("camera_url")
areas = os.environ.get("areas")

# mocks
# username = 'admin'
# password = 'just4Taqtile'
# camera_url = 'http://192.168.1.64/onvif-http/snapshot?Profile_1'
# areas = [
#     {
#         "itemId": 1,
#         "coords": [{
#             "x1": 123,
#             "x2": 232,
#             "y1": 213,
#             "y2": 232
#         }],
#         "itemName": "Item name 1"
#     },
#     {
#         "itemId": 2,
#         "coords": [{
#             "x1": 333,
#             "x2": 444,
#             "y1": 111,
#             "y2": 232
#         }],
#         "itemName": "Item name 2"
#     }
# ]

h = httplib2.Http(".cache")
h.add_credentials(username, password)

model = yolov7.load('min_max_v0.2.6.pt')
# set model parameters
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.classes = None  # (optional list) filter by class


def run():
    while True:
        time.sleep(1)
        resp, content = h.request(camera_url, "GET", body="foobar")
        if resp.status == 200:
            nparr = np.frombuffer(content, np.uint8)
            img0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            print('Error: Could not retrieve image')
            continue

        results = model(img0)
        predictions = results.pred[0]
        boxes = predictions[:, :4]  # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]
        print(categories, 'categories')


if __name__ == '__main__':
    run()
