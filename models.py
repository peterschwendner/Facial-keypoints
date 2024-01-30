## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220 with W=224 picture dimensions
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        self.conv1 = nn.Conv2d(1, 32, 5)
#        self.conv1_bn = nn.BatchNorm2d(32)  # Batch normalization for conv1
#        self.conv1.weight = torch.nn.init.xavier_uniform_(self.conv1.weight)
        
        # after one pool layer, this becomes (32, 110, 110)
        self.pool = nn.MaxPool2d(2, 2)

        # second conv layer: 32 inputs, 64 outputs, 5x5 conv
        ## output size = (110-F)/S +1 = (110-3)/1 +1 = 108
        # the output tensor will have dimensions: (64, 108, 108)
        self.conv2 = nn.Conv2d(32, 64, 3)
#        self.conv2_bn = nn.BatchNorm2d(64)  # Batch normalization for conv2
#        self.conv2.weight = torch.nn.init.xavier_uniform_(self.conv2.weight)

        # after another pool layer this becomes (64, 54, 54)
        #self.pool = nn.MaxPool2d(2, 2)
        # third conv layer: 64 inputs, 64 outputs, 3x3 conv
        ## output size = (54-F)/S +1 = (54-3)/1 +1 = 52
        # the output tensor will have dimensions: (64, 52, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        # after another pool layer this becomes (64, 26, 26)

        self.fc1 = nn.Linear(64*26*26,512)
#        self.fc1_bn = nn.BatchNorm1d(256)  # Batch normalization for fc1
#        self.fc1.weight = torch.nn.init.xavier_uniform_(self.fc1.weight)
        
        self.drop = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(512,136)
#        self.fc2.weight = torch.nn.init.xavier_uniform_(self.fc2.weight)
    
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1(x)))
        #print(x.shape) # => torch.Size([10, 32, 110, 110])
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape) # => torch.Size([10, 64, 53, 53])
        x = self.pool(F.relu(self.conv3(x)))
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        #print(x.shape) # => torch.Size([10, 179776])
        # two linear layers with dropout in between
        x = self.fc1(x)
        #print(x.shape)  # => torch.Size([10, 256])
        x = self.drop(x)
        #print(x.shape) # torch.Size([10, 256])
        x = self.fc2(x)
        #print(x.shape) # torch.Size([10, 136])
        # a modified x, having gone through all the layers of your model, should be returned
        return x