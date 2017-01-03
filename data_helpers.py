import pandas as pd
import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import os

base_path = os.getcwd()


def pull_data(track):
    # load csv file
    df = pd.read_csv('{0}/{1}/driving_log.csv'.format(base_path, track))
    # remove last entry in data frame
    df = df[0:-2]

    # specify different parsing logic for drifting training
    if track == 'drifting':
        df = df.reindex(np.random.permutation(df.index))
        angles = df[['steering']].values
        center_imgs = df[['center']].values
        file_paths = np.copy(center_imgs)
        for i in range(center_imgs.shape[0]):
            file_paths[i] = '{0}/{1}/{2}'.format(base_path, track, center_imgs[i][0])
        throttles = df[['throttle']].values
    else:
        # pull all records that have a steering angle of 0 and find how many there are
        zero_records = df.loc[df['steering'] == 0]
        num_zero = len(zero_records)
        percent_remove_zero = 0.25
        subset_zero = zero_records[int(num_zero*percent_remove_zero):]
        nonzero_records = df.loc[df['steering'] != 0]
        # combine two data frames
        df_final = pd.concat([subset_zero, nonzero_records])
        df_final = df_final.reindex(np.random.permutation(df_final.index))
        center_imgs = df_final[['center']].values
        file_paths = np.copy(center_imgs)
        for i in range(center_imgs.shape[0]):
            file_paths[i] = '{0}/{1}/{2}'.format(base_path, track, center_imgs[i][0])
        angles = df_final[['steering']].values
        throttles = df_final[['throttle']].values
    return file_paths, angles, throttles


def get_training_data():
    imgs1, angles1, throttles1 = pull_data('track_1')
    imgs2, angles2, throttles2 = pull_data('drifting')
    # stack two sources into one
    img_files = np.vstack((imgs1, imgs2))
    angles = np.vstack((angles1, angles2))
    throttles = np.vstack((throttles1, throttles2))
    # convert inputs and outputs
    # angles = convert_continous_angles_to_bins(angles)
    images = convert_paths_to_images(img_files)
    # split data into training and validation datasets
    train_data, validate_data, train_angles, validate_angles, train_throttles, validate_throttles = split_data(images, angles, throttles)
    # combine outputs
    train_outputs = np.hstack((train_angles, train_throttles))
    validate_outputs = np.hstack((validate_angles, validate_throttles))
    return train_data, validate_data, train_outputs, validate_outputs


def convert_paths_to_images(files):
    img_array = np.zeros((files.shape[0], 80, 260, 2), dtype=float)
    for i in range(files.shape[0]):
        img_array[i] = convert_to_image(files[i][0])
    return img_array


def convert_to_image(image):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img[80:, 40:300, 1:]


def split_data(images, angles, throttles):
    return train_test_split(images, angles, throttles, test_size=0.1)


def main():
    get_training_data()

if __name__ == "__main__":
    main()
