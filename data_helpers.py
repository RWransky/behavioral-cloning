import pandas as pd
import numpy as np
from skimage import io
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import os

from warp import *
from threshold import *

base_path = os.getcwd()


def pull_data(track):
    # load csv file
    df = pd.read_csv('{0}/{1}/driving_log.csv'.format(base_path, track))
    # remove last entry in data frame
    df = df[0:-2]
    # # pull all records that have a steering angle of 0 and find how many there are
    # zero_records = df.loc[df['steering'] == 0]
    # num_zero = len(zero_records)
    # percent_remove_zero = 0.55
    # subset_zero = zero_records[int(num_zero*percent_remove_zero):]
    # nonzero_records = df.loc[df['steering'] != 0]
    # # combine two data frames
    # df_final = pd.concat([subset_zero, nonzero_records])
    # df_final = df_final.reindex(np.random.permutation(df_final.index))
    center_imgs = df[['center']].values
    # left_imgs = df[['left']].values
    # right_imgs = df[['right']].values

    angles = df[['steering']].values
    throttles = df[['throttle']].values
    file_paths_present = []
    file_paths_past = []
    angles_arr_present = []
    angles_arr_past = []
    # throttles_arr = []
    for i in range(center_imgs.shape[0]):
        if angles[i] == 0:
            if i % 2 == 0:
                if i == 0:
                    file_paths_past.append('{0}/{1}/{2}'.format(base_path, track, center_imgs[i][0].lstrip()))
                    angles_arr_past.append(angles[i])
                else:
                    file_paths_past.append('{0}/{1}/{2}'.format(base_path, track, center_imgs[i-1][0].lstrip()))
                    angles_arr_past.append(angles[i-1])

                file_paths_present.append('{0}/{1}/{2}'.format(base_path, track, center_imgs[i][0].lstrip()))
                angles_arr_present.append(angles[i])
        else:
            if i == 0:
                file_paths_past.append('{0}/{1}/{2}'.format(base_path, track, center_imgs[i][0].lstrip()))
                angles_arr_past.append(angles[i])
            else:
                file_paths_past.append('{0}/{1}/{2}'.format(base_path, track, center_imgs[i-1][0].lstrip()))
                angles_arr_past.append(angles[i-1])

            file_paths_present.append('{0}/{1}/{2}'.format(base_path, track, center_imgs[i][0].lstrip()))
            angles_arr_present.append(angles[i])

        # throttles_arr.append(throttles[i])
        # file_paths.append('{0}/{1}/{2}'.format(base_path, track, left_imgs[i][0].lstrip()))
        # angles_arr.append(angles[i])
        # throttles_arr.append(throttles[i])
        # file_paths.append('{0}/{1}/{2}'.format(base_path, track, right_imgs[i][0].lstrip()))
        # angles_arr.append(angles[i])
        # throttles_arr.append(throttles[i])

    return file_paths_present, angles_arr_present, file_paths_past, angles_arr_past


def get_training_data():
    imgs1_pres, angles1_pres, imgs1_past, angles1_past = pull_data('track_1_more')
    imgs2_pres, angles2_pres, imgs2_past, angles2_past = pull_data('track_1')
    imgs3_pres, angles3_pres, imgs3_past, angles3_past = pull_data('track_1')
    imgs4_pres, angles4_pres, imgs4_past, angles4_past = pull_data('drifting')
    # stack two sources into one
    img_files_pres = concat_data(imgs1_pres, imgs2_pres, imgs3_pres, imgs4_pres)
    angles_pres = concat_data(angles1_pres, angles2_pres, angles3_pres, angles4_pres)

    img_files_past = concat_data(imgs1_past, imgs2_past, imgs3_past, imgs4_past)
    angles_past = concat_data(angles1_past, angles2_past, angles3_past, angles4_past)
    # throttles = concat_data(throttles1, throttles2, throttles3, throttles4)
    # convert inputs and outputs
    # angles = convert_continous_angles_to_bins(angles)
    images_pres = convert_paths_to_images(img_files_pres)
    images_past = convert_paths_to_images(img_files_past)
    # split data into training and validation datasets
    train_img_pres, validate_img_pres, train_angles_pres, validate_angles_pres, train_img_past, validate_img_past, train_angles_past, validate_angles_past = split_data(images_pres, angles_pres, images_past, angles_past)
    # combine outputs
    # train_outputs = np.hstack((train_angles, train_throttles))
    # validate_outputs = np.hstack((validate_angles, validate_throttles))
    return train_img_pres, validate_img_pres, train_angles_pres, validate_angles_pres, train_img_past, validate_img_past, train_angles_past, validate_angles_past


def merge_data(d1, d2, d3, d4):
    merge1 = np.vstack((d1, d2))
    merge2 = np.vstack((d3, d4))
    return np.vstack((merge1, merge2))


def concat_data(d1, d2, d3, d4):
    merge1 = np.concatenate([d1, d2])
    merge2 = np.concatenate([d3, d4])
    return np.concatenate([merge1, merge2])


def convert_paths_to_images(files):
    img_array = np.zeros((files.shape[0], 160, 320, 3), dtype=np.uint8)
    for i in range(files.shape[0]):
        img_array[i] = convert_to_image(files[i])[...]
    return img_array


def convert_to_image(image):
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    warped = warp_perspective(np.uint8(img))
    # thresh = combine_thresholds(img, k_size_sobel=3, thresh_sobel=(30, 150),
    #                             k_size_mag=3, thresh_mag=(30, 255),
    #                             k_size_dir=3, thresh_dir=(0.5, 1.25))
    # color_grad_thresh = combine_color_grad_thresholds(img, thresh,
    #                                                   space=cv2.COLOR_RGB2HLS,
    #                                                   channel=2, thresh=(30, 150))
    # result = color_grad_thresh[60:145, :]
    # result = img[60:145, :, :]
    # result = cv2.resize(result, (80, 80))
    result = warped
    # plt.imshow(np.uint8(result)/float(255.0))
    # plt.show()
    # plt.imshow(result[0:85, :])
    # # plt.imshow(cv2.resize(result, (100, 60)))
    # plt.imshow(cv2.resize(result, (80, 80)))
    # plt.show()
    return np.uint8(result[...])


def split_data(images_pres, angles_pres, images_past, angles_past):
    return train_test_split(images_pres, angles_pres,
                            images_past, angles_past, test_size=0.1)


def main():
    get_training_data()

if __name__ == "__main__":
    main()
