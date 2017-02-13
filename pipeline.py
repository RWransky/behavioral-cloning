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
        img = warp_perspective(image, self.M)
        blur = cv2.bilateralFilter(img, 25, 100, 100)
        # blur = warp_perspective(image, self.M)
        thresh = combine_thresholds(blur, k_size_sobel=5, thresh_sobel=(20, 200),
                                    k_size_mag=5, thresh_mag=(30, 200),
                                    k_size_dir=5, thresh_dir=(0.7, 1.3))
        color_grad_thresh = combine_color_grad_thresholds(blur, thresh,
                                                          space=cv2.COLOR_RGB2HLS,
                                                          channel=2, thresh=(100, 250))
        color_thresh = color_threshold(blur, space=cv2.COLOR_RGB2HSV,
                                    channel=2, thresh=(90, 130))
        # plt.imshow(color_thresh)
        # plt.show()
        transformed_img = color_thresh
        # transformed_img = warp_perspective(color_thresh, self.M)
        result = process_image_for_lanes(transformed_img, image, self.Minv)
        return result
