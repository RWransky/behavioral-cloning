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
        angles = df[['angle']].values
        center_imgs = df[['center']].values
        file_paths = np.copy(center_imgs)
        for i in range(center_imgs.shape[0]):
            file_paths[i] = '{0}/{1}/{2}'.format(base_path, track, center_imgs[i][0])
    else:
        # pull all records that have a steering angle of 0 and find how many there are
        zero_records = df.loc[df['angle'] == 0]
        num_zero = len(df.loc[df['angle'] == 0])
        percent_keep_zero = 0.5
        subset_zero = zero_records[int(num_zero*percent_keep_zero):]
        nonzero_records = df.loc[df['angle'] != 0]
        # combine two data frames
        df_final = pd.concat([subset_zero, nonzero_records])
        df_final = df_final.reindex(np.random.permutation(df_final.index))
        center_imgs = df_final[['center']].values
        file_paths = np.copy(center_imgs)
        for i in range(center_imgs.shape[0]):
            file_paths[i] = '{0}/{1}/{2}'.format(base_path, track, center_imgs[i][0])
        angles = df_final[['angle']].values
    return file_paths, angles


def get_training_data():
    imgs1, angles1 = pull_data('track_1')
    imgs2, angles2 = pull_data('drifting')
    # stack two sources into one
    img_files = np.vstack((imgs1, imgs2))
    angles = np.vstack((angles1, angles2))
    # convert inputs and outputs
    angles = convert_continous_angles_to_bins(angles)
    images = convert_paths_to_images(img_files)
    # split data into training and validation datasets
    train_data, validate_data, train_angles, validate_angles = split_data(images, angles)
    return train_data, validate_data, train_angles, validate_angles


# Convert steering angles of range -25 to 25 and map to 0.1 to 0.9
def reformat_continous_angles(angles, lower_angle=-25, upper_angle=25):
    reformat_angles = np.zeros(angles.size)
    for i in range(angles.shape[0]):
        reformat_angles[i] = 0.016 * angles[i] + 0.5
    return reformat_angles


# Convert steering angles to categorical bins
def convert_continous_angles_to_bins(labels, lower_angle=-25, upper_angle=25):
    bin_labels = np.zeros(labels.size)
    lower_range = lower_angle
    label_count = 0
    while lower_range < upper_angle + 0.5:
        indx = np.where((labels < lower_range + 0.5).all(axis=1) & (labels >= lower_range).all(axis=1))
        bin_labels[indx] = label_count
        lower_range += 0.5
        label_count += 1
    return np.uint8(bin_labels)


def convert_paths_to_images(files):
    img_array = np.zeros((files.shape[0], 20, 80, 3), dtype=np.uint8)
    for i in range(files.shape[0]):
        img_array[i] = convert_to_image(files[i][0])
    return img_array


def convert_to_image(image):
    img = io.imread(image)
    return cv2.resize(np.uint8(img), (80, 20))


def split_data(images, angles):
    return train_test_split(images, angles, test_size=0.1)


def main():
    get_training_data()

if __name__ == "__main__":
    main()
