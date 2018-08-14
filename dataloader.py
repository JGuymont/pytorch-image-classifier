#!/usr/bin/env python3
"""
Created on Mon Jan 15 13:31:59 2018

@author: J. Guymont

"""

import random
import numpy as np

import torch
from torch.autograd import Variable

class Dataloader:
    """
    Data loader. Combines a dataset and a sampler, and
    provides an iterators over the training dataset.

    Args:
        batch_size (int, optional): how many samples per batch to load (default: 1).
    """
    def __init__(self):
        pass

    def split_ix(self, data_size, split):
        """return shuffled data split train/dev/test
        Args:
            data_size: (Int) number of examples in the dataset
            split: (list of float) train/dev/split in pct (e.g. [.7, .15, .15])

        Return:
            list of index for each data set
        """

        #: set data size based on specified split
        _train_size = round(data_size*split[0])
        _valid_size = round((data_size - _train_size)*split[1])
        _test_size = data_size - _train_size - _valid_size

        _examples = range(data_size)

        _train_ix = random.sample(_examples, _train_size)

        #: remove training datat index
        _examples = [example for example in _examples if example not in _train_ix]

        _valid_ix = random.sample(_examples, _valid_size)

        #: test index are equal to the remainaing
        #: index (after removing train and dev index)
        _test_ix = [example for example in _examples if example not in _valid_ix]

        return _train_ix, _valid_ix, _test_ix

    def split_data(self, inputs, targets, split):
        """split a dataset according to a list of index

        Args:
            inputs: (list) a list of array
            targets: (list) a list of corresponding target label
            split_index: (list) list of index for each data set
        
        Return:
            a list of datasets split in a train/dev/test format
        """

        _data_size = len(targets)

        _train_ix, _valid_ix, _test_ix = self.split_ix(_data_size, split)

        _train_x = inputs[_train_ix]
        _train_y = targets[_train_ix]

        _valid_x = inputs[_valid_ix]
        _valid_y = targets[_valid_ix]

        _test_x = inputs[_test_ix]
        _test_y = targets[_test_ix]

        return _train_x, _train_y, _valid_x, _valid_y, _test_x, _test_y

    def data_loader(self, inputs, targets, batch_size):
        """provides an iterator over a dataset"""
        _data_size = len(targets)
        _examples = range(_data_size)

        dataloader = []

        for _ in range(int(_data_size/batch_size)):

            #: randomly select examples for current SGD iteration
            _mini_batch = random.sample(_examples, batch_size)

            #: remove current example from the list of examples
            _examples = [example for example in _examples if example not in _mini_batch]

            #: Convert array to tensor of size [batch_size, 1, img_size, img_size]
            _batch_x = Variable(torch.Tensor(inputs[_mini_batch, :])).view(batch_size, 1, 28, 28)
            _batch_y = Variable(torch.LongTensor(targets[_mini_batch]))

            dataloader.append((_batch_x, _batch_y))

        return dataloader

if __name__ == '__main__':

    inputs, targets = np.load('./data/inputs.npy'), np.load('./data/targets.npy')

    train_x, train_y, valid_x, valid_y, test_x, test_y = Dataloader().split_data(inputs, targets, split=[.7, .15, .15])

    trainloader = Dataloader().data_loader(train_x, train_y, batch_size=64)

    devloader = Dataloader().data_loader(valid_x, valid_y, batch_size=64)

    testloader = Dataloader().data_loader(test_x, test_y, batch_size=64)