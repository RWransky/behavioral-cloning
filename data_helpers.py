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
    df = df[0:-1]
    # pull all records that have a steering angle of 0 and find how many there are
    zero_records = df.loc[df['angle'] == 0]
    num_zero = len(df.loc[df['angle'] == 0])
    percent_keep_zero = num_zero/float(len(df))
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
    imgs2, angles2 = pull_data('track_2')
    # stack two sources into one
    img_files = np.vstack((imgs1, imgs2))
    labels = np.vstack((angles1, angles2))
    # convert inputs and outputs
    labels = convert_continous_angles_to_bins(labels)
    images = convert_paths_to_images(img_files)
    # split data into training and validation datasets
    train_data, validate_data, train_labels, validate_labels = split_data(images, labels)
    return train_data, validate_data, train_labels, validate_labels


# Convert steering angles of range -25 to 25 and map to 0 to 1
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
    img = cv2.cvtColor(io.imread(image), cv2.COLOR_RGB2HSV)
    return cv2.resize(np.uint8(img), (80, 20))


def split_data(images, labels):
    return train_test_split(images, labels, test_size=0.1)


def main():
    get_training_data()

if __name__ == "__main__":
    main()
