#!/usr/bin/env python3
"""Convnet Classifier"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

# Main classifier that subclasses nn.Module
class Classifier(nn.Module):
    """Convnet Classifier"""
    def __init__(self):

        self.input_size = 28
        self.num_classes = 10
        self.channel_sizes = [1, 10]
        self.num_cnn_layer = len(self.channel_sizes) - 1
        self.kernel_sizes = [10]
        self.dense_input_size = self.compute_dense_input_size()

        self.cnn_architecture = self.build_cnn()

        super(Classifier, self).__init__()

        self.conv = nn.Sequential(
            #: series of cnn layers.
            #: the cnn graph is used to learn the filters.
            #: Filters are matrix of parameters that will 
            #: give high value output when convolve with
            #: image that contain certain pattern detect
            #: by this particular filter (e.g. an edge detector)
            self.cnn_architecture
        )

        self.dense = nn.Sequential(
            #: fully connected network that will learn how to
            #: classify based on the result of the convolution
            #: with the different filters
            nn.Linear(self.dense_input_size, self.num_classes),
            nn.ReLU(),
        )

    def build_cnn(self):
        """Construct the cnn architecture based
        on provided hyperparameters"""
        _cnn_architecture = OrderedDict()
        for i in range(self.num_cnn_layer):
            _cnn_architecture['conv{}'.format(i)] = nn.Conv2d(self.channel_sizes[i], self.channel_sizes[i+1], self.kernel_sizes[i])
            _cnn_architecture['relu{}'.format(i)] = nn.ReLU()
        return _cnn_architecture

    def compute_dense_input_size(self, stride=1, dilation=1):
        #: return the size of the collapsed cnn output tensor 
        #: into one dimension. This tensor will be the feed
        #: to the fully connected network
        L_out = self.input_size
        for k in self.kernel_sizes:
            L_out = (L_out - dilation*(k - 1) - 1)/stride + 1    
        return self.channel_sizes[-1]*L_out**2

    def flatten(self, x):
        #: Return a copy of the tensor of shape 
        #: (batch_size, num_channels, N, N) collapsed 
        #: into one dimension. This tensor will be the feed
        #: to the fully connected network 
        return x.view(-1, self.dense_input_size)     

    def forward(self, x):
        """return a vector of lenght 20 with a score
        for each digits. The digit with the highest score
        is the predicted one"""
        x = self.conv(x)
        x = self.flatten(x)
        x = self.dense(x)
        return F.log_softmax(x, dim=1)

    def evaluate(self, dataloader):
        """
        evaluate accuracy
        """
        total = 0
        correct = 0

        for (inputs, targets) in dataloader:

            _outputs = self.forward(inputs)
            _, predicted = torch.max(_outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

        accuracy = 100.*correct/total

        return accuracy
