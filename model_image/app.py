from PIL import Image
from flask import Flask, jsonify, request
from flask_configs.load_configs import *
from ObjectDetectionModel import ObjDetectionModel
import numpy as np
import colorlog
import logging
import os


app = Flask(__name__)
human_model = ObjDetectionModel(HUMAN_MODEL_PATH, CONF_THRES, IOU_THRES, CLASSES)
box_model = ObjDetectionModel(BOX_MODEL_PATH, CONF_THRES, IOU_THRES, CLASSES)


logger = logging.getLogger('min_max_logger')
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'CRITICAL': 'bold_red,bg_white',
        }))
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False

@app.route('/predict_human', methods=['POST'])
def predict_human():
    if request.method == 'POST':
        image = np.array(request.json['image']).astype(np.float32)
        n_boxes, coords = human_model(image)
        return jsonify(
            {
                "n_boxes": n_boxes,
                "coordinates": coords.tolist()
            }
        )
    
@app.route('/predict_boxes', methods=['POST'])
def predict_boxes():
    if request.method == 'POST':
        image = np.array(request.json['image']).astype(np.float32)
        n_boxes, coords = box_model(image)
        logger.info("Request to predict_boxes: " + str(n_boxes))
        return jsonify(
            {
                "n_boxes": n_boxes,
                "coordinates": coords.tolist()
            }
        )
