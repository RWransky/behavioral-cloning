from warp import *
from threshold import *
from lanes import *
import numpy as np
import matplotlib.pyplot as plt


class Pipeline():
    def __init__(self):
        image = cv2.imread('seed_images/seed1.jpg')
        # plt.imshow(image)
        # plt.show()
        self.M = get_trans_mtx(image)
        self.Minv = np.linalg.inv(self.M)
        # plt.imshow(warp_perspective(image, self.M))
        # plt.show()

    def process(self, image):
        # img = warp_perspective(image, self.M)
        blur = cv2.bilateralFilter(image, 25, 100, 100)
        color_thresh = color_threshold(blur, space=cv2.COLOR_RGB2HSV,
                                    channel=2, thresh=(90, 170))
        return color_thresh[int(image.shape[0]/3):, :]
