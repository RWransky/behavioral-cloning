import pandas as pd
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import cv2
import os

base_path = os.getcwd()


def pull_data(track):
    # load csv file
    df = pd.read_csv('{0}/{1}/driving_log.csv'.format(base_path, track))
    center_imgs = df[['center']].values
    file_paths = np.copy(center_imgs)
    for i in range(center_imgs.shape[0]):
        file_paths[i] = '{0}/{1}/{2}'.format(base_path, track, center_imgs[i][0])
    angles = df[['angle']].values
    return file_paths[0:-1], angles[0:-1]


def get_training_data():
    imgs1, angles1 = pull_data('track_1')
    imgs2, angles2 = pull_data('track_2')
    # stack two sources into one
    img_files = np.vstack((imgs1, imgs2))
    labels = np.vstack((angles1, angles2))
    images = convert_paths_to_images(img_files)
    plt.imshow(images[1300])
    plt.show()


def convert_paths_to_images(files):
    img_array = np.zeros((files.shape[0], 80, 320, 3))
    for i in range(files.shape[0]):
        img_array[i] = convert_to_image(files[i][0])
    return img_array


def convert_to_image(image):
    return io.imread(image)


def main():
    get_training_data()

if __name__ == "__main__":
    main()
