#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:31:59 2018
@author: Jonathan Guymont
"""
import pickle
import gzip
import matplotlib.pyplot as plt
import random

import torch
from torch.autograd import Variable

def normalize(x):

    mean = x.mean()
    std = x.std()

    normalized_x = (x - mean)/std

    return normalized_x

def load_mnist(train_size=1.):
    """
    load MNIST data and split into
    predefined train/dev/test
    """

    with gzip.open('./data/mnist.pkl.gz', 'rb') as f:

        _train_set, _valid_set, _test_set = pickle.load(f, encoding='latin1')

        _train_size = round(train_size*len(_train_set[0]))
        _valid_size = round(train_size*len(_valid_set[0]))
        _test_size = len(_test_set[0])

        _train_x, _train_y = _train_set
        valid_x, valid_y = _valid_set
        test_x, test_y = _test_set

    _train_ix = range(_train_size)
    _valid_ix = range(_valid_size)
    _test_ix = range(_test_size)

    return [normalize(_train_x)[_train_ix, :], _train_y[_train_ix], 
            normalize(valid_x)[_valid_ix, :], valid_y[_valid_ix], 
            normalize(test_x)[_test_ix, :], test_y[_test_ix]]

def batch_sampler(x, y, batch_size=1):
    """
    Data loader. Combines a dataset and a sampler, and
    provides an iterators over the training dataset.

    Args:
        batch_size (int, optional): how many samples per batch to load (default: 1).
    """

    _data_size = len(y)
    _feature_size = x.shape[1]

    _examples = range(_data_size)

    dataloader = []

    for _ in range(int(_data_size/batch_size)):

        #: randomly select examples for current SGD iteration
        _mini_batch = random.sample(_examples, batch_size)

        #: remove current example from the list of examples
        _examples = [example for example in _examples if example not in _mini_batch]

        #: Convert torch tensor to Variable
        _batch_x = Variable(torch.Tensor(x[_mini_batch, :])).view(batch_size, 1, 28, 28)
        _batch_y = Variable(torch.LongTensor(y[_mini_batch]))

        dataloader.append((_batch_x, _batch_y))

    return dataloader


if __name__ == '__main__':
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist()
    
    TRAINLOADER = batch_sampler(train_x, train_y, batch_size=64)

    # plt.imshow(train_x[0].reshape((28, 28)))
    # plt.show()
