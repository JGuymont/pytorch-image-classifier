#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:31:59 2018
@author: Jonathan Guymont
"""
import pickle
import gzip
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def load_mnist():
    """
    load MNIST data and split into
    predefined train/dev/test
    """

    with gzip.open('./data/mnist.pkl.gz', 'rb') as f:

        _train_set, _valid_set, _test_set = pickle.load(f, encoding='latin1')

        _train_x, _train_y = _train_set
        _valid_x, _valid_y = _valid_set
        _test_x, _test_y = _test_set

    return [_train_x, _train_y, _valid_x, _valid_y, _test_x, _test_y]
        
if __name__ == '__main__':
    data = load_mnist()
    _train_x = data[0]
    plt.imshow(_train_x[0].reshape((28, 28)))
    plt.show()

