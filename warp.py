import cv2
import numpy as np

# src coordinates
src = np.float32([
    [0, 75],
    [0, 115],
    [320, 115],
    [320, 75]
])

# dest coordinates
dst = np.float32([
    [-50, 0],
    [10, 120],
    [300, 120],
    [370, 0]
])


# Note image is undistorted image
def get_trans_mtx(image):
    img_size = (image.shape[1], image.shape[0])
    trans_mtx = cv2.getPerspectiveTransform(src, dst)
    return trans_mtx


def warp_perspective(image, M):
    return cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_NEAREST)
