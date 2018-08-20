import os
import time

from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    
    def build_conv_net(self):
        """Construct the cnn architecture based on provided hyperparameters"""
        
        _cnn_architecture = OrderedDict()
        
        _cnn_architecture['batch_norm{}'.format(0)] = nn.BatchNorm2d(self.input_channel)
        
        # cnn layer 1
        # --------------------------
        _cnn_architecture['conv{}'.format(1)] = nn.Conv2d(self.input_channel, self.channel_sizes[0], self.kernel_sizes[0])
        _cnn_architecture['relu{}'.format(1)] = nn.ReLU()
        
        if self.max_pool[0] > 0:
            _cnn_architecture['max_pool{}'.format(1)] = nn.MaxPool2d(self.max_pool[0])
        
        if self.dropout[0] > 0.:
            _cnn_architecture['dropout{}'.format(1)] = nn.Dropout(p=self.dropout[0])
        
        if self.batch_norm[0]:
            _cnn_architecture['batch_norm{}'.format(1)] = nn.BatchNorm2d(self.channel_sizes[0])
        
        if self.num_cnn_layer == 1:
            return _cnn_architecture
        # --------------------------
        
        for i in range(1, self.num_cnn_layer):
            
            # Add convolutional layer
            _cnn_architecture['conv{}'.format(i+1)] = nn.Conv2d(self.channel_sizes[i-1], self.channel_sizes[i], self.kernel_sizes[i])
            
            # Add activation function (relu)
            _cnn_architecture['relu{}'.format(i+1)] = nn.ReLU()
            
            # Add pooling
            if self.max_pool[i] > 0:
                _cnn_architecture['max_pool{}'.format(i+1)] = nn.MaxPool2d(self.max_pool[i])
            
            # Add dropout
            if self.dropout[i] > 0.:
                _cnn_architecture['dropout{}'.format(i+1)] = nn.Dropout(p=self.dropout[i])
                        
            # Apply batch normalization
            if self.batch_norm[i]:
                _cnn_architecture['batch_norm{}'.format(i+1)] = nn.BatchNorm2d(self.channel_sizes[i])
            
        return _cnn_architecture
    
    def build_dense_net(self):
        
        _dense_architecture = OrderedDict()
        
        # Dense layer 1
        # ------------------------------------------------
        if self.num_dense_layer == 1:
            _dense_architecture['fc1'] = nn.Linear(self.dense_input_size, self.num_classes)
            _dense_architecture['softmax'] = nn.LogSoftmax(dim=1)
            return _dense_architecture
        else:
            _dense_architecture['fc1'] = nn.Linear(self.dense_input_size, self.hidden_layer_sizes[0])
            _dense_architecture['relu1'] = nn.ReLU()
            if self.dense_dropout[0] > 0.:
                _dense_architecture['dropout{}'.format(1)] = nn.Dropout(p=self.dense_dropout[0])
        # ------------------------------------------------
        
        # Dense hidden layers
        # ------------------------------------------------
        for i in range(1, self.num_dense_layer-1):
            
            # Add fully connected layer
            _dense_architecture['fc{}'.format(i+1)] = nn.Linear(self.hidden_layer_sizes[i-1], self.hidden_layer_sizes[i])
            _dense_architecture['relu{}'.format(i+1)] = nn.ReLU()
            
            # Add dropout
            if self.dense_dropout[i] > 0.:
                _dense_architecture['dropout{}'.format(i)] = nn.Dropout(p=0.5)
        # ------------------------------------------------
        
        # Last linear layer
        _dense_architecture['fc{}'.format(self.num_dense_layer)] = nn.Linear(self.hidden_layer_sizes[-1], self.num_classes)
        _dense_architecture['softmax'] = nn.LogSoftmax(dim=1)
        return _dense_architecture
    
    def _compute_dense_input_size(self, stride=1, dilation=1):
        #: return the size of the collapsed cnn output tensor 
        #: into one dimension. This tensor will be the feed
        #: to the fully connected network
        out_size = self.input_size
        for i in range(self.num_cnn_layer):
            # adjust for convolution
            out_size = (out_size - dilation*(self.kernel_sizes[i] - 1) - 1)/stride + 1    
            # adjust for pooling
            if self.max_pool[i] is not None:
                out_size = np.floor(out_size/self.max_pool[i]).astype(int)
        return int(self.channel_sizes[-1]*out_size**2)

    
    def __init__(self, args):
        
        self.input_size = args.input_size
        
        self.input_channel = args.input_channel
        self.channel_sizes = args.channel_sizes
        self.num_cnn_layer = len(self.channel_sizes)
        self.kernel_sizes = args.kernel_sizes
        self.max_pool = args.max_pool
        self.dropout = args.dropout
        self.batch_norm = args.batch_norm
        
        self.dense_input_size = self._compute_dense_input_size()
        self.num_dense_layer = args.num_dense_layer
        self.hidden_layer_sizes = args.hidden_layer_sizes
        self.dense_dropout = args.dense_dropout
        self.num_classes = args.num_classes
        
        self.cnn_architecture = self.build_conv_net()
        
        super(CNN, self).__init__()
        
        self.conv = nn.Sequential(
            #: series of cnn layers.
            #: the cnn graph is used to learn the filters.
            #: A filters is a matrix of parameters that will 
            #: give high value output when convolve with 
            #: image that contain certain pattern detect
            #: by this particular (e.g. an edge detector)
            self.cnn_architecture
        )
        
        self.dense = nn.Sequential(
            #: fully connected network that will learn how to
            #: classify based on the result of the convolution
            #: with the different filters
            nn.Linear(self.dense_input_size, self.num_classes),
        )

    def _flatten(self, x):
        #: Return a copy of the tensor of shape 
        #: (batch_size, num_channels, N, N) collapsed 
        #: into one dimension. This tensor will be the feed
        #: to the fully connected network 
        return x.view(-1, self.dense_input_size)
        
    def forward(self, x):
        """return a vector of lenght 10 with a score
        for each digits. The digit with the highest score
        is the predicted one"""
        x = self.conv(x)
        x = self._flatten(x)
        x = self.dense(x)
        return F.log_softmax(x, dim=1)

class Classifier(CNN):
    """Convnet Classifier"""

    def __init__(self, args):

        self.verbose = args.verbose
        self.num_epochs = args.epochs

        super(Classifier, self).__init__(args)

        if args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        elif args.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=args.lr, momentum=args.momentum)

        self.loss_function = nn.NLLLoss()

    def train(self, trainloader, devloader):
        """Train the model.

        Arguments:
            trainloader:
            devloader:
        """
        for epoch in range(self.num_epochs):
            start_time = time.time()
            losses = []

            # Train
            for (_inputs, _targets) in trainloader:
                
                # make sure the gradient are all set to 0
                # before minibatch training
                self.optimizer.zero_grad() # reset derivative to 0
                
                # comput network prediction for the current batch
                _outputs = self.forward(_inputs)
                
                # compulte loss
                loss = self.loss_function(_outputs, _targets)
                
                # Use autograd to compute the derivative of the loss w.r.t 
                # all Tensors with requires_grad=True. After calling `loss.backward()`, 
                # conv_weight.grad, dense_weight.grad, and dense_bias.grad 
                # will be Tensors equal to the gradient of the loss with respect 
                # to the filters of the cnn layer, the weight of the fully connected layer, and 
                # the bias of the fully connected layer respectively.
                loss.backward()

                # Apply gradient descent to all the leaned parameters
                # The derivative of the loss is giving us the direction
                # where the funtion increase. Thus we go in the 
                # opposite direction. Using torch.no_grad() tells pytorch
                # to not include thes operation in the computational graph.
                # Instead, gradient descent is goning to be applied `inplace`.
                self.optimizer.step() 

                losses.append(loss.item())
            
            # compute epoch accuracy
            _train_acc = self.evaluate(trainloader)
            _valid_acc = self.evaluate(devloader)

            if self.verbose:
                print('Epoch: {:.0f}\tLoss: {:.3f}\tTrain acc: {:.2f}\tDev acc: {:.2f}'.format(
                    epoch+1, np.mean(losses), _train_acc, _valid_acc
                    ))
                print("--- %s seconds ---" % (time.time() - start_time))
    
    @staticmethod
    def save_model(model, model_dir, model_name):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(model, '{}/{}'.format(model_dir, model_name))
    
    @staticmethod
    def load_model(model_dir, model_name):
        model_path = '{}/{}'.format(model_dir, model_name)
        model = torch.load(model_path)
        return model
    
    def predict(self, x):
        """Make a prediction.
        
        Arguments
            x: (Tensor or Array) input to be classified. 
                Most have shape (n, 1, 28, 28)
        Return 
            int: class 
        """
        if torch.is_tensor(x) is False:
            x = torch.Tensor(x)
        _outputs = self.forward(x)
        _, pred = torch.max(_outputs.data, 1)
        return pred
    
    def evaluate(self, dataloader):
        """Evaluate accuracy.
        
        """
        total = 0.
        correct = 0.

        for (inputs, targets) in dataloader:

            _outputs = self.forward(inputs)
            _, predicted = torch.max(_outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

        accuracy = 100.*correct/total

        return accuracy