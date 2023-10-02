from PIL import Image
from flask import Flask, jsonify, request
from flask_configs.load_configs import bottle_model_configs, box_model_configs, human_model_configs
from ObjectDetectionModel import YOLOv8ObjDetectionModel
from YOLONASObjDetectionModel import YOLONASObjDetectionModel
import numpy as np
import colorlog
import logging
import io

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

app = Flask(__name__)
human_model = YOLONASObjDetectionModel(**human_model_configs)
box_model = YOLOv8ObjDetectionModel(**box_model_configs)
bottle_model = YOLONASObjDetectionModel(**bottle_model_configs)


convert_bytes2image = lambda bytes: np.array(Image.open(io.BytesIO(bytes)), dtype=np.uint8)

@app.route('/predict_human', methods=['POST'])
def predict_human():
    if request.method == 'POST':
        image = convert_bytes2image(request.files["image"].read()).astype(np.float32)
        coords = human_model(image)
        logger.info(f"request to predict_human: {len(coords)}")
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
        logger.info(f"request to predict_boxes: {len(coords)}")
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
        logger.info(f"request to predict_bottles: {len(coords)}")
        return jsonify(
            {
                "coordinates": coords.tolist()
            }
        )
