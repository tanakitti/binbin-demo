from flask import Flask, request, jsonify
from DetectTrash import detectObjectFromImage
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import cv2
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def home():
    path = "f:/_Archieve/FinalSeniorProject/ModelEval/Images/Plastic/test26.jpg"
    img = Image.open(path)
    im_file = BytesIO()
    img.save(im_file, format="JPEG")
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    im_b64 = base64.b64encode(im_bytes)
    
    return im_b64

@app.route("/predict", methods=["POST"])
def predict():
    result = 0
    if request.method == "POST":    		
        input_value = request.form["input_value"]
        im_bytes = base64.b64decode(input_value)   # im_bytes is a binary image
        im_file = BytesIO(im_bytes)  # convert image to file-like object
        img = Image.open(im_file)
        
        name,score = detectObjectFromImage(np.array(img))
    return jsonify(
		name = name,
    score = float(score)
	),200


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8082)