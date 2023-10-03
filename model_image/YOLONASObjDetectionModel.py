from super_gradients.training import models
import torch
import numpy as np


class YOLONASObjDetectionModel:
    def __init__(self, model_path: str, conf_thresh: float, iou_thresh: float, classes: list, img_size: int) -> None:
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self._model = models.get(model_path, pretrained_weights='coco').to(device)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.classes = classes
        self.img_size = img_size

    @torch.no_grad()
    def __call__(self, img: np.array, classes: list = None) -> list:
        print(img.shape)
        results = self._model(img)
        print(results)
        coords_with_confs = torch.hstack((results.xyxy, results.conf.unsqueeze(-1)))
        return coords_with_confs