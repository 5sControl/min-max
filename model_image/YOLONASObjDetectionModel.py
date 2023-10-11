from super_gradients.training import models
from super_gradients.common.object_names import Models
import torch
import numpy as np


class YOLONASObjDetectionModel:
    def __init__(self, model_path: str, conf_thresh: float, iou_thresh: float, classes: list, img_size: int) -> None:
        # self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self._model = models.get(model_path, pretrained_weights='coco', num_classes=80)
        self._model.eval()
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.classes = classes
        self.img_size = img_size

    @torch.no_grad()
    def __call__(self, img: np.array) -> list:
        results = self._model.predict(img, fuse_model=False)
        prediction = results[0].prediction
        labels = prediction.labels.astype(int) == self.classes[0]
        boxes = prediction.bboxes_xyxy[labels]
        confs = np.expand_dims(prediction.confidence.astype(float)[labels], -1)
        #confs = 
        coords_with_confs = np.hstack((boxes, confs))
        return coords_with_confs