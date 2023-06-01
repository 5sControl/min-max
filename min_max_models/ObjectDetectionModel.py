from ultralytics import YOLO


class ObjDetectModel:
    def __init__(self, path: str, device, conf_thresh, iou_thresh, classes) -> None:
        model = YOLO(path)
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
        result = nms_pred[0]
        for detections in nms_pred:
            count += len(detections)
        return [count, result]
