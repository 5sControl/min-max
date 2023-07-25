import torch
from .utils.torch_utils import select_device
from .models.models import Darknet


def get_model(weights, cfg):
    imgsz = 1280
    device = select_device('cpu')
    model = Darknet(cfg, imgsz)
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    model.to(device).eval()
    return model, device
