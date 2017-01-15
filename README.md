Autonomous Driving via Behavioral Cloning
==================================

Background
-------------------
This Keras model was built to satisfy the requirements for a project in Udacity's Self-Driving Car Nanodegree Program. A Python [driving simulator](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f290_simulator-macos/simulator-macos.zip) simulator was provided (Note: link points to MacOS version only). This simulator provides the ability to gather training data and an autonmous mode. Training mode produces 7 main things:
- Image from center of car
- Image from left of car
- Image from right of car
- Steering angle (already normalized between [-1, 1] for real-world values of [-25, 25])
- Speed
- Throttle (already normalized between [0, 1])
- Brake

Model Data Description
-----------------------
This model requires three inputs:
- Previous center image (binarized lane mask)
- Previous steering angle (already normalized to [-1, 1])
- Current center image (binarized lane mask)

Both images undergo three transforms before being fed into the model. First, a perspective transform is done to convert the image perspective from straight ahead to birds-eye view. Next, a Sobel transform thresholded by magnitude values is done. Finally, the resulting binary mask is cropeed to be 40x80. See images below for visualizations.

**Source Image**
![center_image](https://cloud.githubusercontent.com/assets/13735131/21959385/34264b22-da93-11e6-94ad-9bcfa52aee55.png)

**Perspective Transform**
![warped](https://cloud.githubusercontent.com/assets/13735131/21959384/342656e4-da93-11e6-8537-e289dfd7054f.png)

**Sobel Magnitude Threshold (Final Input)**
![sobel_thresh](https://cloud.githubusercontent.com/assets/13735131/21959383/34256d1a-da93-11e6-86bd-15cdad04cabf.png)


Model Description
----------------------
The core model is a multi-branched convolutional regressive neural network. It directly outputs the normalized steering angle between [-1, 1]. Below is a diagram of the full model. More explicit details can be found in `network.py`.

![model](https://cloud.githubusercontent.com/assets/13735131/21959636/064cac66-da9b-11e6-9a1d-c6fd69bb577f.png)

Training Description
----------------------
The model was trained using a dataset from [track 1](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip), a dataset from track 2, and a dataset from track 1 that exhibited lane recovery maneuvers. Each dataset emphasized normal driving conditions with the car staying on the track at all times.

Because this model is regressive in nature, the training methodology involved reducing the mean squared error loss computed between the true angle vs. the predicted angle.

An Adam optimizer was used with 100 training epochs with batch sizes of 512.

To avoid overfitting, dropout regularization was incorporated throughout many of the model's layers. Additionally, the initial dataset was split into a training set and a validation set, with 10% of all data belonging to the validation set.

The final model that was saved was based upon the best mean squared error loss recorded for the validation dataset.
