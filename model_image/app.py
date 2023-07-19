from PIL import Image
from flask import Flask, jsonify, request
from flask_configs.load_configs import *
from ObjectDetectionModel import ObjDetectionModel
import numpy as np
import colorlog
import logging
import io


app = Flask(__name__)
human_model = ObjDetectionModel(HUMAN_MODEL_PATH, CONF_THRES, IOU_THRES, CLASSES)
box_model = ObjDetectionModel(BOX_MODEL_PATH, CONF_THRES, IOU_THRES, CLASSES)
bottle_model = ObjDetectionModel(BOTTLE_MODEL_PATH, CONF_THRES, IOU_THRES, CLASSES)


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

convert_bytes2image = lambda bytes: np.array(Image.open(io.BytesIO(bytes)), dtype=np.uint8)

@app.route('/predict_human', methods=['POST'])
def predict_human():
    if request.method == 'POST':
        image = convert_bytes2image(request.files["image"].read()).astype(np.float32)
        coords = human_model(image)
        return jsonify(
            {
                "coordinates": coords.tolist()
            }
        )
    
@app.route('/predict_boxes', methods=['POST'])
def predict_boxes():
    if request.method == 'POST':
        image = convert_bytes2image(request.files["image"].read()).astype(np.float32)
        coords = box_model(image)
        return jsonify(
            {
                "coordinates": coords.tolist()
            }
        )

@app.route('/predict_bottles', methods=['POST'])
def predict_bottles():
    if request.method == 'POST':
        image = convert_bytes2image(request.files["image"].read()).astype(np.float32)
        coords = bottle_model(image)
        return jsonify(
            {
                "coordinates": coords.tolist()
            }
        )
