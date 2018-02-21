import cv2
import numpy as np
import os
from sklearn.utils import shuffle

class Img_Augmentation:

    def augment_brightness_camera_images(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image = np.array(image, dtype=np.float64)
        random_bright = .5 + np.random.uniform()
        image[:, :, 2] = image[:, :, 2] * random_bright
        image[:, :, 2][image[:, :, 2] > 255] = 255
        image = np.array(image, dtype=np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        return image

    def add_random_shadow(self, image):
        top_y = 320 * np.random.uniform()
        top_x = 0
        bot_x = 160
        bot_y = 320 * np.random.uniform()
        image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        shadow_mask = 0 * image_hls[:, :, 1]
        X_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][0]
        Y_m = np.mgrid[0:image.shape[0], 0:image.shape[1]][1]

        shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1
        if np.random.randint(2) == 1:
            random_bright = .5
            cond1 = shadow_mask == 1
            cond0 = shadow_mask == 0
            if np.random.randint(2) == 1:
                image_hls[:, :, 1][cond1] = image_hls[:, :, 1][cond1] * random_bright
            else:
                image_hls[:, :, 1][cond0] = image_hls[:, :, 1][cond0] * random_bright
        image = cv2.cvtColor(image_hls, cv2.COLOR_HLS2RGB)
        return image

    def rnd_flip(self, image):
        """
        Flips a Image randomly, 
        :param image: 
        :return: 
        """
        ind_flip = np.random.randint(-1,2)
        image = cv2.flip(image, ind_flip)
        return image

    def rnd_rotation(self, image, range=[0,180]):
        """
        Rotates a Image a random degree between -25..+25
        :param image: Image to rotate
        :return: 
        """
        img = image
        num_rows, num_cols = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2),  np.random.randint(range[0],range[1]), 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
        return img_rotation

    def __getFileArray(self , directory):
        return  os.listdir(directory)

    def createRandomArray(self, directory,  percent):
        inputarray= self.__getFileArray(directory)
        inputarray= shuffle(inputarray)
        length = int(len(inputarray) * percent)
        return inputarray[:length]
