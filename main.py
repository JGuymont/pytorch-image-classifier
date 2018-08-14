#!/usr/bin/env python

import time
import random
import numpy as np 
import matplotlib.pyplot as plt
import argparse

import torch
from load_mnist import load_mnist, batch_sampler
from cnn import Classifier

def argparser():
    """Training settings"""
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--train_size', type=float, default=1.,
                        help='proportion of the data to use')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    return parser.parse_args()

def train(args, model, trainloader, devloader, optimizer, criterion):
    """
    """
    for epoch in range(args.epochs):
        start_time = time.time()
        losses = []

        # Train
        for (_inputs, _targets) in trainloader:

            optimizer.zero_grad()
            _outputs = model.forward(_inputs)

            loss = criterion(_outputs, _targets)
            loss.backward() #
            optimizer.step() 
            losses.append(loss.item())
        
        _train_acc = model.evaluate(trainloader)
        _valid_acc = model.evaluate(devloader)

        print('Epoch : {:.0f} Loss : {:.3f} Train acc : {:.2f} Dev acc : {:.2f}'.format(
            epoch+1, np.mean(losses), _train_acc, _valid_acc
            ))
        print("--- %s seconds ---" % (time.time() - start_time))

def main():

    args = argparser()

    train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist(args.train_size)

    trainloader = batch_sampler(train_x, train_y, batch_size=args.batch_size)
    devloader = batch_sampler(valid_x, valid_y, batch_size=args.test_batch_size)
    testloader = batch_sampler(test_x, test_y, batch_size=args.test_batch_size)

    _model = Classifier()

    # _optimizer = torch.optim.Adam(_model.parameters(), lr=args.lr)
    _optimizer = torch.optim.SGD(_model.parameters(), lr=args.lr, momentum=args.momentum)

    _criterion = torch.nn.NLLLoss()

    train(args, _model, trainloader, devloader, _optimizer, _criterion)

    test_acc = _model.evaluate(testloader)

    print('test acc:', test_acc)

    torch.save(_model, './models/cnn')


if __name__ == '__main__':

    main()


