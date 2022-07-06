'''
Networks with different architectures.
'''

import inspect
import math
import sys

import numpy as np
import torch
from torch import nn

from setup import INPUT_CHANNELS, INPUT_SIZE, OUTPUT_CHANNELS, OUTPUT_SIZE


# Return a sequence of layers related to convolution.
def convolution(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs, padding_mode='replicate'),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        )

# Return a sequence of layers related to deconvolution.
def deconvolution(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        )


class ClassifierCnn(nn.Module):
    def __init__(self, input_channels: int, number_classes: int):
        super().__init__()

        self.convolution_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False),
        )
        self.convolution_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False),
        )
        self.convolution_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False),
        )
        self.convolution_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False),
        )
        self.convolution_5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False),
        )
        self.linear = nn.Linear(in_features=256*7*7, out_features=number_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.float()

        x = self.convolution_1(x)
        x = self.convolution_2(x)
        x = self.convolution_3(x)
        x = self.convolution_4(x)
        x = self.convolution_5(x)
        x = self.linear(x.view(x.shape[0], -1))
        x = self.softmax(x)

        return x


# GAN based on: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

class GanGenerator(nn.Module):
    def __init__(self, input_channels: int, number_features: int, output_channels: int, embedding_size: int):
        super().__init__()

        self.latent_size = input_channels
        self.embedding_size = embedding_size

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(self.latent_size, number_features * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(number_features * 16),
            nn.ReLU(inplace=True),

            # nn.ConvTranspose2d(number_features * 32, number_features * 16, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(number_features * 16),
            # nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(number_features * 16, number_features * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(number_features * 8),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(number_features * 8, number_features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(number_features * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(number_features * 4, number_features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(number_features * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(number_features * 2, number_features * 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(number_features * 1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(number_features * 1, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x, label):
        label_embedding = nn.Embedding(self.embedding_size, self.latent_size)(label.flatten())
        return self.layers(x * label_embedding.reshape((*label_embedding.shape, 1, 1)))
        # return self.layers(x)

class GanDiscriminator(nn.Module):
    def __init__(self, number_features: int, output_channels: int, embedding_size: int):
        super().__init__()

        self.embedding_size = embedding_size

        self.layers = nn.Sequential(
            # nn.Conv2d(output_channels, number_features * 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Conv2d(output_channels + 1, number_features * 1, kernel_size=4, stride=2, padding=1, bias=False),  # Add 1 to account for label channel
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(number_features * 1, number_features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(number_features * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(number_features * 2, number_features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(number_features * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(number_features * 4, number_features * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(number_features * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(number_features * 8, number_features * 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(number_features * 16),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            # nn.Conv2d(number_features * 16, number_features * 32, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.BatchNorm2d(number_features * 32),
            # nn.LeakyReLU(negative_slope=0.2, inplace=True),
            
            nn.Conv2d(number_features * 16, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, label: int):
        label_embedding = nn.Embedding(self.embedding_size, np.prod(x.size()[2:]))(label)
        label_embedding = torch.reshape(label_embedding.flatten(), (x.shape[0], 1, *x.shape[2:4]))
        # Concatenate along the channel dimension.
        return self.layers(torch.cat((x, label_embedding), axis=1))
        # return self.layers(x)


# Store all classes defined in this module in a dictionary.
networks = []
start_lines = {}
for name, obj in inspect.getmembers(sys.modules[__name__]):
    if inspect.isclass(obj):
        # Preserve the order the classes are defined in.
        _, start_line = inspect.getsourcelines(obj)
        start_lines[name] = start_line
        networks.append((name, obj))
networks = sorted(networks, key=lambda _: start_lines[_[0]])
networks = dict(networks)