import numpy as np


def accuracy(labels, predictions):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0])


def intensity_normalization(image):
    image = image.astype('float')
    return image / 255.0
