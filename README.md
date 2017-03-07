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
- Previous center image (RGB cropped and intensity normalized)
- Previous steering angle (already normalized to [-1, 1])
- Current center image (RGB cropped and intensity normalized)

Both images undergo transforms before being fed into the model. First, the image is cropped to remove unnecessary data from above the road. This cropped image is then shrunk to be 40x80. The image stays in the RGB color channel. Before being fed into the model, each image is intesity normalized to be within the range of [0, 1].

**Source Image**
![figure_1](https://cloud.githubusercontent.com/assets/13735131/23663675/4fe248a0-0321-11e7-8bbc-daefdb238e93.png)

**Cropped and Resized**
![cropped](https://cloud.githubusercontent.com/assets/13735131/23663674/4fe1dee2-0321-11e7-923d-4f7e58fcdbe2.png)

Model Description
----------------------
The core model is a multi-branched convolutional regressive neural network. It directly outputs the normalized steering angle between [-1, 1]. Below is a diagram of the full model. More explicit details can be found in `network.py`.

![model](https://cloud.githubusercontent.com/assets/13735131/21959636/064cac66-da9b-11e6-9a1d-c6fd69bb577f.png)

Training Description
----------------------
The model was trained using a dataset from [track 1](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) and a dataset from track 1 that exhibited lane recovery maneuvers. Each dataset emphasized normal driving conditions with the car staying on the track at all times.

Because this model is regressive in nature, the training methodology involved reducing the mean squared error loss computed between the true angle vs. the predicted angle.

An Adam optimizer was used with 100 training epochs with batch sizes of 300.

To avoid overfitting, dropout regularization was incorporated throughout many of the model's layers. Additionally, the initial dataset was split into a training set and a validation set, with 1% of all data belonging to the validation set. Only 1% was used because the total dataset size was over 15,000 images.

The final model that was saved was based upon the best mean squared error loss recorded for the validation dataset.
