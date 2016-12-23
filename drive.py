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
from io import BytesIO

from network import *

# Fix error with Keras and TensorFlow
import tensorflow as tf
tf.python.control_flow_ops = tf


sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None


def predict(image, path='./weights'):
    tf.reset_default_graph()
    mainN = Network()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
        return sess.run([mainN.probs],
            feed_dict={mainN.input_layer: np.uint8(image)})


@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    img = base64.b64decode(imgString)
    npimg = np.fromstring(img, dtype=np.uint8)
    image = cv2.imdecode(npimg, 1)
    scaled_image = image_helper(image)
    steering_angle = float(model.predict(scaled_image, batch_size=1))
    angle = rescale_angle(steering_angle)
    throttle = 0.2
    print(angle, throttle)
    send_control(angle, throttle)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__()
    }, skip_sid=True)


# Helper method to resize image to 20x80x3 and convert to HSV colorspace
def image_helper(image):
    img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    return cv2.resize(np.uint8(img), (80, 20))


# Convert angles from [0.1,0.9] -> [-25,25]
def rescale_angle(angle):
    return (angle-0.5)/0.016


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
        help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        model = model_from_json(json.load(jfile))

    model.compile(optimizer='adam', loss='mean_squared_error')
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
