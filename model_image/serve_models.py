from PIL import Image
from flask import Flask, jsonify, request
from flask_configs.load_configs import *
from ObjectDetectionModel import ObjDetectionModel
import numpy as np


app = Flask(__name__)
human_model = ObjDetectionModel(HUMAN_MODEL_PATH, CONF_THRES, IOU_THRES, CLASSES)
box_model = ObjDetectionModel(BOX_MODEL_PATH, CONF_THRES, IOU_THRES, CLASSES)


@app.route('/predict_human', methods=['POST'])
def predict_human():
    if request.method == 'POST':
        img_file = request.files['image']
        image = np.array(Image.open(img_file))
        n_boxes, coords = human_model(
            source=image,
            conf=CONF_THRES,
            iou=IOU_THRES,
            max_det=600,
            classes=CLASSES,
            verbose=False
        )
        print(n_boxes)
        return jsonify(
            {
                "n_boxes": n_boxes,
                "coordinates": coords.tolist()
            }
        )
    
@app.route('/predict_boxes', methods=['POST'])
def predict_boxes():
    if request.method == 'POST':
        img_file = request.files['image']
        image = np.array(Image.open(img_file))
        n_boxes, coords = box_model(
            source=image,
            conf=CONF_THRES,
            iou=IOU_THRES,
            max_det=600,
            classes=CLASSES,
            verbose=False
        )
        return jsonify(
            {
                "n_boxes": n_boxes,
                "coordinates": coords.tolist()
            }
        )

if __name__ == '__main__':
    app.run(debug=False, port=PORT, load_dotenv=False)
