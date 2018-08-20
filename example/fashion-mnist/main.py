#!/usr/bin/env/python3
"""
Created on Sat Aug 18 2018

@author: J. Guymont
"""
import time
import random
import pickle
import numpy as np 
import matplotlib.pyplot as plt
import argparse

import torch

from context import image_classifier
from image_classifier.models.cnn import Classifier 

def argparser():
    """Training settings"""
    parser = argparse.ArgumentParser(description='PyTorch Fashion MNIST Example')

    parser.add_argument('--model_name', type=str, default='default',
                        help='The model will be save under this name.')
    
    parser.add_argument('--input_size', type=int, default=28,
                        help='Image size in pixels')
    
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes')
    
    parser.add_argument('--verbose', type=bool, default=True,
                        help='print training and dev accuracy during training')

    # -- cnn architecture
    parser.add_argument('--input_channel', type=int, default=1,
                        help='Number of input channel (e.g. 1 for greyscale and 3 for RGB image')

    parser.add_argument('--channel_sizes', type=lambda s: [int(n) for n in s.split()],
                        help='Sizes of the cnn layers.')
                    
    parser.add_argument('--kernel_sizes', type=lambda s: [int(n) for n in s.split()],
                        help='Sizes of the filters.')

    parser.add_argument('--dropout', type=lambda s: [float(n) for n in s.split()], default=[0.],
                        help='Dropout percentage at each conv layer. None means no dropout.')
    
    parser.add_argument('--max_pool', type=lambda s: [int(n) for n in s.split()], default=[0],
                        help='List of pooling kernel size. None means no pooling')
    
    parser.add_argument('--batch_norm', type=lambda s: [bool(n) for n in s.split()], default=[False],
                        help='whether to apply batch normalization')

    # -- linear layers architecture

    parser.add_argument('--num_dense_layer', type=int, default=1,
                        help='Number of fully connected layers')
    
    parser.add_argument('--hidden_layer_sizes', type=lambda s: [int(n) for n in s.split()], default=[],
                        help='Size of the hidden layer. Only if num_dense_layer > 1')

    parser.add_argument('--dense_dropout', type=lambda s: [float(n) for n in s.split()], default=[0.],
                        help='Size of the hidden layer. Only if num_dense_layer > 1')
    
    # -- training parameters 

    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'],
                        help='Which optimizer to use.')

    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')

    return parser.parse_args()

if __name__ == '__main__':

    with open('./example/fashion-mnist/data/dataloaders.pkl', 'rb') as f:
        dataloaders = pickle.load(f)

    trainloader = dataloaders['trainloader']
    devloader = dataloaders['devloader']
    testloader = dataloaders['testloader']

    args = argparser()

    clf = Classifier(args)
    print(clf)

    clf.train(trainloader, devloader)     
    clf.save_model(clf, './example/fashion-mnist/models', args.model_name)
    