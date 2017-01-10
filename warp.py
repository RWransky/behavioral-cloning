import cv2
import numpy as np

# Define source points taken from image of straight lanes
src_pts = np.float32([[0, 115], [320, 115], [230, 90], [65, 90]])


# Note image is undistorted image
def warp_perspective(image):
    img_size = (image.shape[1], image.shape[0])
    dst_pts = np.float32([[0, 160], [320, 160],
                         [320, 0], [0, 0]])
    trans_mtx = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, trans_mtx, img_size)
