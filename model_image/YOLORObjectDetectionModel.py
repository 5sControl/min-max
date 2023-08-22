import torch
from yolor.model import get_model
import numpy as np
from yolor.utils.datasets import letterbox
from yolor.utils.general import non_max_suppression, scale_coords


class YOLORObjectDetectionModel:
    def __init__(self, model_path: str, config_path: str, conf_thresh, iou_thresh, classes, **kwargs) -> None:
        self.model, self.device = get_model(model_path, config_path)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.classes = classes
        self.logger = kwargs.get("logger")

    def __preprocess_image__(self, img: np.array) -> np.array:
        self.img_shape = img.shape
        img = letterbox(img.copy(), new_shape=1280, auto_size=64)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        img = img.unsqueeze(0)
        return img

    @torch.no_grad()
    def __call__(self, img: np.array) -> list:
        img = self.__preprocess_image__(img)
        pred = self.model(img, augment=False)[0]
        pred = non_max_suppression(
            pred, self.conf_thresh, self.iou_thresh, classes=self.classes, agnostic=False)[0]
        pred[:, :4] = scale_coords(
            img.shape[2:], pred[:, :4], self.img_shape).round()
        return pred[:, :5]
