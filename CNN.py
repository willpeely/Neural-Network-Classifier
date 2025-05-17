import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tranforms
from torch.utils.data import DataLoader
import time

# Setting the variables for convolutional layers
convolutional_kernel_size = 3    
convolutional_stride = 1
convolutional_padding = 1
convolutional_activation = nn.ReLU()
convolutional_pool = nn.MaxPool2d(kernel_size=2, stride=2)

# Used to create a convolutional layer with defined input and output channels
def convolutional_layer(input_channels, output_channels, pool=True):
    """ Returns a convolutional layer.

    Args:
        input_channels (int): The number of inputs into the layer.
        output_channels (int): The number of outputs the layer produces.
        pool (bool, optional): If True the convolutional layer will use pooling. Defaults to True.
    """
    layers = [
        nn.Conv2d(
            in_channels=input_channels, 
            out_channels=output_channels,
            kernel_size=convolutional_kernel_size,
            stride=convolutional_stride,
            padding=convolutional_padding
        ),

        nn.BatchNorm2d(
            num_features=output_channels
        ),

        convolutional_activation
    ]
    if pool:
        layers.append(convolutional_pool)

    return nn.Sequential(*layers)