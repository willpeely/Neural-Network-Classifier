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

# fully connected variables
fully_connected_activation = nn.ReLU()
fully_connected_dropout = 0.5

# Used to create a fully connected layer with defined input and output features
def fully_connected_layer(input_features, output_features, dropout=True):
    """ Returns a fully connected layer.

    Args:
        input_features (int): The number of inputs into the layer.
        output_features (int): The number of outputs produced by the layer.
        dropout (bool, optional): If true the fully connected layer will use dropout. Defaults to True.
    """
    layers = [
        nn.Linear(
            in_features=input_features, 
            out_features=output_features
        ),
        fully_connected_activation
    ]
    if dropout:
        layers.append(nn.Dropout(fully_connected_dropout))

    return nn.Sequential(*layers)

# Used to get the flattened size after going through convolutional layers
def get_flattened_size(convolutional_layers):
    dummy = torch.zeros(1, 3, 32, 32)  
    conv_output = convolutional_layers(dummy)
    flattened_size = conv_output.view(1, -1).size(1)

    return flattened_size

# Data variables
image_size = 32
RGB_channels = 3
image_classes = 10

# Convolutional neural network class
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.convolutional_layers = nn.Sequential(
            convolutional_layer(input_channels=RGB_channels, output_channels=32),
            convolutional_layer(input_channels=32, output_channels=64),
        )

        flattened_size = get_flattened_size(self.convolutional_layers)

        self.fully_connected_layers = nn.Sequential(
            nn.Flatten(),
            fully_connected_layer(flattened_size, 128),
            nn.Linear(128, image_classes)
        )

    def forward(self, x):
        x = self.convolutional_layers(x)
        x = self.fully_connected_layers(x)
        return x
