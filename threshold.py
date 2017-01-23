import cv2
import numpy as np


def sobel_abs_threshold(img, direction='x', sobel_kernel=3, thresh=(0, 255)):
    # 1) Convert to grayscale assumes RGB input
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given direction = 'x' or 'y'
    if direction == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel/float(np.max(abs_sobel)))
    # 5) Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    mask = np.zeros_like(scaled_sobel)
    mask[(scaled_sobel <= thresh[1]) & (scaled_sobel >= thresh[0])] = 1
    return mask


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # 1) Convert to grayscale assumes RGB input
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    dir = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    mask = np.zeros_like(dir)
    lower = thresh[0]
    upper = thresh[1]
    mask[(dir <= upper) & (dir >= lower)] = 1
    return mask


def mag_threshold(img, sobel_kernel=3, thresh=(0, 255)):
    # 1) Convert to grayscale assumes RGB input
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Calculate the magnitude
    mag = np.sqrt(sobel_x*sobel_x + sobel_y*sobel_y)
    # 5) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * mag/float(np.max(mag)))
    # 6) Create a binary mask where mag thresholds are met
    sbinary = np.zeros_like(scaled_sobel)
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    sbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sbinary


def color_threshold(img, space=cv2.COLOR_RGB2HLS, channel=2, thresh=(0, 255)):
    transform_img = cv2.cvtColor(img, space)
    single_channel = transform_img[:, :, channel]
    mask = np.zeros_like(single_channel)
    mask[(single_channel >= thresh[0]) & (single_channel <= thresh[1])] = 1
    return mask


def combine_thresholds(img, k_size_sobel=3, thresh_sobel=(0, 255), k_size_dir=3, thresh_dir=(0, np.pi/2), k_size_mag=3, thresh_mag=(0, 255)):
    sobel_x = sobel_abs_threshold(img, direction='x', sobel_kernel=k_size_sobel, thresh=thresh_sobel)
    sobel_y = sobel_abs_threshold(img, direction='y', sobel_kernel=k_size_sobel, thresh=thresh_sobel)
    dir_mask = dir_threshold(img, sobel_kernel=k_size_dir, thresh=thresh_dir)
    mag_mask = mag_threshold(img, sobel_kernel=k_size_mag, thresh=thresh_mag)

    combine = np.zeros_like(mag_mask)
    combine[((sobel_x == 1) | (sobel_y == 1)) | ((mag_mask == 1) & (dir_mask == 1))] = 1
    return combine


def combine_color_grad_thresholds(img, grad_thres, space=cv2.COLOR_RGB2HLS, channel=2, thresh=(0, 255)):
    color_thresh = color_threshold(img, space=space, channel=channel, thresh=thresh)
    mask = np.zeros_like(color_thresh)
    mask[(grad_thres == 1) & (color_thresh == 1)] = 1
    return mask

