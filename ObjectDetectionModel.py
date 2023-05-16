from typing import Any
import torch
from utils.general import non_max_suppression
from models.experimental import attempt_load
from utils.torch_utils import TracedModel


class ObjDetectModel:
    def __init__(self, path: str, device, conf_thresh, iou_thresh, classes) -> None:
        model = attempt_load(path, map_location=device)
        self.stride = int(model.stride.max())
        self.model = TracedModel(model, device)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.classes = classes

    @torch.no_grad()
    def __call__(self, img) -> Any:
        preds = self.model(img, False)[0]
        nms_pred = non_max_suppression(
            preds,
            self.conf_thresh,
            self.iou_thresh,
            classes=self.classes,
            agnostic=False
        )
        count = 0
        result = []
        for detections in nms_pred:
            count += len(detections)
        return count
