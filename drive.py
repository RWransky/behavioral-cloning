import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
import cv2
from flask import Flask, render_template
import matplotlib.pyplot as plt
from io import BytesIO

from highway_unit import *
from model_helpers import *
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data['steering_angle']
    # The current throttle of the car
    throttle = data['throttle']
    # The current speed of the car
    speed = data['speed']
    # The current image from the center camera of the car
    imgString = data['image']
    img = base64.b64decode(imgString)
    npimg = np.fromstring(img, dtype=np.uint8)
    image = cv2.imdecode(npimg, 1)
    scaled_image = image_helper(image)
    steering_angle = float(model.predict(scaled_image, batch_size=1))
    throttle = 0.2
    print(steering_angle, throttle)
    send_control(float(steering_angle), throttle)


@sio.on('connect')
def connect(sid, environ):
    print('connect ', sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


# Helper method to resize image to 80x260x3 and convert to YUV
def image_helper(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img = img[80:, 40:300, :]
    return img[np.newaxis, ...]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
        help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # model = model_from_json(jfile.read(), {'HighwayUnit': HighwayUnit()})
        model = model_from_json(jfile.read(), custom_objects={'intensity_norm': intensity_norm})

    model.compile(optimizer='adam', loss='mse')
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
