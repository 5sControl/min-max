from ultralytics import YOLO
import torch
import numpy as np


class YOLOv8ObjDetectionModel:
    def __init__(self, path: str, conf_thresh: float, iou_thresh: float, classes: list) -> None:
        self.model = YOLO(path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.classes = classes

    @torch.no_grad()
    def __call__(self, img: np.array, classes: list = None) -> list:
        results = self.model(
            source=img,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            max_det=600,
            classes=self.classes if classes is None else classes,
            verbose=False
        )[0].boxes
        coords_with_confs = torch.hstack((results.xyxy, results.conf.unsqueeze(-1)))
        return coords_with_confs
    