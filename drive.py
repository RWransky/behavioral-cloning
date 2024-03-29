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

from warp import *
from threshold import *
from pipeline import *

from network import *
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
import keras.activations


# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf

keras.activations.intensity_norm = intensity_norm

prev_angle = 0
prev_image = None
new_image = None
start = True


sio = socketio.Server()
app = Flask(__name__)
model = None


class Memory:
    def __init__(self):
        self.prev_angle = 0
        self.prev_image = np.zeros((1, 40, 80, 3))

    def record(self, angle, image):
        self.prev_image = image
        self.prev_angle = angle

    def fetch(self):
        return self.prev_angle, self.prev_image


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
    new_image = image_helper(image)

    prev_angle, prev_image = memory.fetch()
    prev_angle = np.array(prev_angle)

    outputs = model.predict([prev_image, new_image, prev_angle[np.newaxis, ...]], batch_size=1)
    steering_angle = float(outputs[0])
    memory.record(steering_angle, new_image)

    # throttle = float(outputs[0][1])
    send_control(steering_angle, 0.8)


@sio.on('connect')
def connect(sid, environ):
    print('connect ', sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


# Helper method for image input
def image_helper(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = img[int(img.shape[0]/3):,:]
    # result = pipeline.process(img)
    result = img
    result = cv2.resize(result, (80, 40))
    return np.uint8(result[np.newaxis, ...])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
        help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # model = model_from_json(jfile.read(), {'HighwayUnit': HighwayUnit()})
        model = model_from_json(jfile.read(), {'intensity_norm': intensity_norm})

    model.compile(optimizer='adam', loss='mse')
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    memory = Memory()

    pipeline = Pipeline()

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
