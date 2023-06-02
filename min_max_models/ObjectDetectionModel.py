from ultralytics import YOLO
import torch


class ObjDetectionModel:
    def __init__(self, path: str, device, conf_thresh, iou_thresh, classes) -> None:
        self.model = YOLO(path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.classes = classes

    @torch.no_grad()
    def __call__(self, img) -> list:
        results = self.model(
            source=img,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            max_det=300,
            classes=self.classes,
            verbose=False
        )[0].boxes
        n_boxes = len(results)
        return [n_boxes, results.xyxy]
    