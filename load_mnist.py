#!/usr/bin/env python3
"""
Created on Mon Jan 18 2018

@author: J. Guymont
"""

import os
import numpy as np
from tqdm import tqdm

from preprocessing import IMGPreprocessing

class MNIST(IMGPreprocessing):
    """load mnist dataset and and apply 
    preprocessing from module IMGPreprocessing

    Arguments:
        mode
    """

    def __init__(self, mode, img_path, flatten=False):

        self.img_path = img_path

        super(MNIST, self).__init__(mode=mode, flatten=False)

    def mnist_digit(self, digit):
        """load all image of the specified digit. 
        This method suppose that all image of a same
        digits are in a folder '/path_to_mnist_data/`digit`/' 
        """
        data = []
        _img_dir = '{}/{}'.format(self.img_path, digit)
        _img_list = os.listdir(_img_dir)
        for img in _img_list:
            img_path = '{}/{}'.format(_img_dir, img)
            data.append(self.transform(img_path))
        return data

    def preprocess(self):
        """apply preprocessing to inputs 
        and store all data in a list.
        
        create a list of corresponding targets
        """
        _inputs = []
        _targets = []
        for i in tqdm(range(10)):
            _cur_digits = self.mnist_digit(i)
            _num_digit_example = len(_cur_digits)
            _inputs.extend(_cur_digits)
            _targets.extend([i]*_num_digit_example)
        return _inputs, _targets

    def save(self, data, dest_path):
        """save data"""
        np.save(dest_path, data)
        return None

if __name__ == '__main__':

    MNIST_DATA_PATH = './data/jpg/trainingSet'

    mnist = MNIST(mode='L', img_path=MNIST_DATA_PATH)
    inputs, targets = mnist.preprocess()
    mnist.save(inputs, './data/inputs')
    mnist.save(targets, './data/targets')
