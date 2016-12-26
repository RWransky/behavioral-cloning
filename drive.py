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

train_mean_red = np.load('train_mean_red.npy')
train_std_red = np.load('train_std_red.npy')
train_mean_green = np.load('train_mean_green.npy')
train_std_green= np.load('train_std_green.npy')
train_mean_blue = np.load('train_mean_blue.npy')
train_std_blue = np.load('train_std_blue.npy')
# Normalize imaging data
train_mean = [train_mean_red, train_mean_green, train_mean_blue]
train_std = [train_std_red, train_std_green, train_std_blue]
print(train_std)

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
    angle = rescale_angle(steering_angle)
    throttle = 0.2
    print(angle, throttle)
    send_control(float(angle), throttle)


@sio.on('connect')
def connect(sid, environ):
    print('connect ', sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit('steer', data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


# Helper method to resize image to 20x80x3 and apply normalization
def image_helper(image):
    img = cv2.resize(np.uint8(image), (80, 20))
    img = intensity_normalization(img)
    # for i in range(3):
    #     img[:, :, i] = (img[:, :, i] - train_mean[i])/train_std[i]
    return img[np.newaxis, ...]


# Convert angles from bins -> [-25,25]
def rescale_angle(angle, lower_angle=-25, upper_angle=25):
    print(angle)
    return angle*(upper_angle - lower_angle) + lower_angle
    # Uncomment section below for bin to angle conversion
    # label = np.argmax(bins, 1)[0]
    # return 0.5*label - 25


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
        help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(jfile.read(), {'HighwayUnit': HighwayUnit()})

    model.compile(optimizer='adam', loss='mean_squared_error')
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
