#!/usr/bin/env python3
"""
Created on Mon Jan 15 13:31:59 2018

@author: J. Guymont

"""
from PIL import Image
import numpy as np

class IMGPreprocessing(object):
    """image preprocessor

    Arguments:
        mode: (String) either "L" for balack and white or "RGB" for color
        flatten: (Boolean) should the image be transform into a 1-d array (vs original 2-d)
    """
    def __init__(self, mode, flatten=False):
        self.mode = mode
        self.flatten = flatten

    def open_image(self, image_path):
        """open a jpg file as a color or a black and white image

        Arguments:
            image_path: (String) path to the image to be transform
        """
        with Image.open(image_path) as img:
            _image = img.convert(self.mode)
        return _image

    def jpg_to_array(self, image):
        """transform a jpg file into an array

        Arguments:
            image: (String) a jpg image
        """
        array_image = np.asarray(image, np.float)
        return array_image

    def scale(self, array_image):
        """scale all the pixel between -1 and 1

        Arguments:
            array_image: (Array) array representation of an image
                i.e. a matrix of the same dimension as the
                original image with pixel value as element
        """
        min_value = np.min(array_image)
        max_value = np.max(array_image)
        scaled_image = (array_image - min_value)/(max_value - min_value)*2 - 1
        return scaled_image

    def normalize(self, image_array):
        """normalize an array

        Arguments:
            array_image: (Array) array representation of an image
        """
        mean = image_array.mean()
        std = image_array.std()
        return (image_array - mean)/std

    def flatten_array(self, array_image):
        """convert a 2 dimensional array into a 1 dimensional array
        so it can be feed to a neural network

        Arguments:
            array_image: (Array) array representation of an image
        """
        img_dim = array_image.shape[0]
        return array_image.ravel().reshape(1, img_dim**2)

    def transform(self, image_path):
        """wrapper method

        Arguments:
            image_path: (String) path to the image to be transform
        """
        image = self.open_image(image_path)
        image = self.jpg_to_array(image)
        image = self.normalize(image)
        if self.flatten:
            image = self.flatten_array(image)
        return image

if __name__ == '__main__':
    IMG_PATH = './data/jpg/trainingSet/0/img_1.jpg'

    img = IMGPreprocessing(mode="L").open_image(IMG_PATH)
    img.show()

    preprocess = IMGPreprocessing(mode="L").transform
    img = preprocess(IMG_PATH)
    print(img)
