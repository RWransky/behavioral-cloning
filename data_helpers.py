import pandas as pd
import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import os

from warp import *
from sobel import *

base_path = os.getcwd()


def pull_data(track):
    # load csv file
    df = pd.read_csv('{0}/{1}/driving_log.csv'.format(base_path, track))
    # remove last entry in data frame
    df = df[0:-2]
    # pull all records that have a steering angle of 0 and find how many there are
    zero_records = df.loc[df['steering'] == 0]
    num_zero = len(zero_records)
    percent_remove_zero = 0.1
    subset_zero = zero_records[int(num_zero*percent_remove_zero):]
    nonzero_records = df.loc[df['steering'] != 0]
    # combine two data frames
    df_final = pd.concat([subset_zero, nonzero_records])
    df_final = df_final.reindex(np.random.permutation(df_final.index))
    center_imgs = df_final[['center']].values
    left_imgs = df_final[['left']].values
    right_imgs = df_final[['right']].values
    file_paths = []
    for i in range(center_imgs.shape[0]):
        file_paths.append('{0}/{1}/{2}'.format(base_path, track, center_imgs[i][0].lstrip()))
        # file_paths.append('{0}/{1}/{2}'.format(base_path, track, left_imgs[i][0].lstrip()))
        # file_paths.append('{0}/{1}/{2}'.format(base_path, track, right_imgs[i][0].lstrip()))

    angles = df_final[['steering']].values
    throttles = df_final[['throttle']].values
    return file_paths, angles, throttles


def get_training_data():
    imgs1, angles1, throttles1 = pull_data('drifting')
    imgs2, angles2, throttles2 = pull_data('track_1_more')
    imgs3, angles3, throttles3 = pull_data('track_1')
    imgs4, angles4, throttles4 = pull_data('track_2')
    # stack two sources into one
    img_files = concat_data(imgs1, imgs2, imgs3, imgs4)
    angles = merge_data(angles1, angles2, angles3, angles4)
    throttles = merge_data(throttles1, throttles2, throttles3, throttles4)
    # convert inputs and outputs
    # angles = convert_continous_angles_to_bins(angles)
    images = convert_paths_to_images(img_files)
    # split data into training and validation datasets
    train_data, validate_data, train_angles, validate_angles, train_throttles, validate_throttles = split_data(images, angles, throttles)
    # combine outputs
    train_outputs = np.hstack((train_angles, train_throttles))
    validate_outputs = np.hstack((validate_angles, validate_throttles))
    return train_data, validate_data, train_angles, validate_angles


def merge_data(d1, d2, d3, d4):
    merge1 = np.vstack((d1, d2))
    merge2 = np.vstack((d3, d4))
    return np.vstack((merge1, merge2))


def concat_data(d1, d2, d3, d4):
    merge1 = np.concatenate([d1, d2])
    merge2 = np.concatenate([d3, d4])
    return np.concatenate([merge1, merge2])


def convert_paths_to_images(files):
    img_array = np.zeros((files.shape[0], 160, 320, 1), dtype=float)
    for i in range(files.shape[0]):
        img_array[i] = convert_to_image(files[i])[..., np.newaxis]
    return img_array


def convert_to_image(image):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    warped = warp_perspective(np.uint8(img))
    thresh = mag_threshold(warped, sobel_kernel=11, thresh=(30, 130))
    # plt.imshow(warp_perspective(dir_threshold(np.uint8(img))))
    # plt.imshow(mag_threshold(warped, sobel_kernel=11, thresh=(30, 130)))
    # plt.show()
    return np.float32(thresh)


def split_data(images, angles, throttles):
    return train_test_split(images, angles, throttles, test_size=0.1)


def main():
    get_training_data()

if __name__ == "__main__":
    main()
