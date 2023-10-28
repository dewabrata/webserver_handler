#todolist https://github.com/Mikubill/sd-webui-controlnet/wiki/API --> tambah controlnet
#reference https://towardsdatascience.com/stable-diffusion-as-an-api-5e381aec1f6

import json
import base64
import os
import io
import requests
import numpy as np
import cv2
from flask import Flask, jsonify, request
from flask_cors import CORS

import uuid
from process import process_masking

#https://github.com/ternaus/cloths_segmentation --> segmentasi pakaian
import albumentations as albu
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from cloths_segmentation.pre_trained_models import create_model

# IMG2IMG_URL = 'http://127.0.0.1:7860/sdapi/v1/img2img'



def submit_post(url: str, data: dict):
    """
    Submit a POST request to the given URL with the given data.
    """
    print(json.dumps(data))
    return requests.post(url, data=json.dumps(data))

def _b64encode(x: bytes) -> str:
    return base64.b64encode(x).decode("utf-8")


def fileImg2b64(png_filepath):
    with open(png_filepath, "rb") as image_file:
        # Membaca file dan mengonversinya menjadi base64
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string.decode('utf-8')



def img2b64(img):
    """
    Convert a PIL image to a base64-encoded string.
    """
    buffered = io.BytesIO()
    img.save(buffered, format='PNG')
    return _b64encode(buffered.getvalue())


def save_encoded_image(b64_image: str, output_path: str):
    """
    Save the given image to the given output path.
    """
    with open(output_path, "wb") as image_file:
        image_file.write(base64.b64decode(b64_image))



app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

@app.route('/api/islive', methods=['GET'])
def isLive():
    return jsonify({"status": "OK"})


@app.route('/api/mask', methods=['POST'])
def receive_state():
    """Endpoint to receive state data and file."""
    # Handle file upload
    print("STARTING MASKING")

    try:
        # Parse JSON data
        data = request.get_json()
        if data is None or 'image' not in data:
            return jsonify({'error': 'Invalid input, please send JSON with "image" key containing Base64 encoded image data.'}), 400

       
        imgArray = process_masking(data['image'])

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
   

    
   
    return jsonify({"mask": imgArray})





if __name__ == '__main__':
    app.run(debug=True)

   
    