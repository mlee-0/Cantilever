'''
Networks with different architectures.
'''

import math

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


class FullyCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution_1 = convolution(
            in_channels=INPUT_CHANNELS, out_channels=4, kernel_size=3, stride=2, padding=1,
            )
        self.convolution_2 = convolution(
            in_channels=4, out_channels=8, kernel_size=3, stride=2, padding=1,
            )
        self.convolution_3 = convolution(
            in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1,
            )
        self.convolution_4 = convolution(
            in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1,
            )
        self.convolution_5 = convolution(
            in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1,
            )
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.linear = nn.Linear(in_features=8192, out_features=np.prod(OUTPUT_SIZE))

    def forward(self, x):
        x = x.float()
        # Convolution.
        x = self.convolution_1(x)
        x = self.convolution_2(x)
        x = self.convolution_3(x)
        x = self.convolution_4(x)
        x = self.convolution_5(x)
        # Fully connected.
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = x.reshape(-1, *OUTPUT_SIZE)
        return x

class UNetCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolution_1 = convolution(
            in_channels=INPUT_CHANNELS, out_channels=8, kernel_size=3, stride=3, padding=1,
            )
        self.convolution_2 = convolution(
            in_channels=8, out_channels=16, kernel_size=3, stride=3, padding=1,
            )
        self.deconvolution_1 = deconvolution(
            in_channels=16, out_channels=8, kernel_size=7, stride=3, padding=1,
            )
        self.deconvolution_2 = deconvolution(
            in_channels=8, out_channels=4, kernel_size=7, stride=3, padding=1,
            )
        self.deconvolution_3 = deconvolution(
            in_channels=4, out_channels=OUTPUT_CHANNELS, kernel_size=7, stride=3, padding=1,
            )
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpooling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.linear = nn.Linear(in_features=11342, out_features=np.prod(OUTPUT_SIZE))

    def forward(self, x):
        x = x.float()
        # Convolution.
        x = self.convolution_1(x)
        size_1 = x.size()
        x, indices_1 = self.pooling(x)
        x = self.convolution_2(x)
        size_2 = x.size()
        x, indices_2 = self.pooling(x)
        # Deconvolution.
        # x = self.unpooling(x, indices_2, output_size=size_2)
        x = self.deconvolution_1(x)
        # x = self.unpooling(x, indices_1, output_size=size_1)
        x = self.deconvolution_2(x)
        x = self.deconvolution_3(x)
        

        # Fully connected.
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        x = x.reshape(-1, *OUTPUT_SIZE)
        return x

class AutoencoderCnn(nn.Module):
    def __init__(self):
        super().__init__()
        
        # self.convolution_1 = convolution(
        #     in_channels=INPUT_CHANNELS, out_channels=4, kernel_size=1, stride=2, padding=0
        #     )
        self.convolution_2 = convolution(
            in_channels=INPUT_CHANNELS, out_channels=8, kernel_size=1, stride=2, padding=0
            )
        self.convolution_3 = convolution(
            in_channels=8, out_channels=16, kernel_size=1, stride=2, padding=0
            )
        self.convolution_4 = convolution(
            in_channels=16, out_channels=32, kernel_size=1, stride=2, padding=0
            )
        self.convolution_5 = convolution(
            in_channels=32, out_channels=64, kernel_size=1, stride=2, padding=0
            )
        self.convolution_6 = convolution(
            in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0
            )
        # self.deconvolution_2 = deconvolution(
        #     in_channels=128, out_channels=64, kernel_size=1, stride=2, padding=0, output_padding=(1,1)
        #     )
        self.deconvolution_3 = deconvolution(
            in_channels=128, out_channels=64, kernel_size=1, stride=2, padding=0, output_padding=(1,0)
            )
        self.deconvolution_4 = deconvolution(
            in_channels=64, out_channels=32, kernel_size=1, stride=2, padding=0, output_padding=(0,0)
            )
        self.deconvolution_5 = deconvolution(
            in_channels=32, out_channels=16, kernel_size=1, stride=2, padding=0, output_padding=(0,0)
            )
        self.deconvolution_6 = deconvolution(
            in_channels=16, out_channels=8, kernel_size=1, stride=2, padding=0, output_padding=(0,1)
            )
        self.deconvolution_7 = deconvolution(
            in_channels=8, out_channels=OUTPUT_CHANNELS, kernel_size=1, stride=2, padding=0, output_padding=(1,1)
            )
        # self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # self.unpooling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0)
        self.linear = nn.Linear(in_features=5000, out_features=5000)
        
        self.flatten = nn.Flatten()
        self.autoencoder = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(in_features=1024, out_features=64),
            # nn.Linear(in_features=24, out_features=12),
            # nn.Linear(in_features=32, out_features=16),
            # nn.Linear(in_features=12, out_features=6),
            # nn.Linear(in_features=6, out_features=3),
            # nn.Linear(in_features=3, out_features=6),
            # nn.Linear(in_features=6, out_features=12),
            # nn.Linear(in_features=16, out_features=32),
            # nn.Linear(in_features=12, out_features=24),
            nn.Linear(in_features=64, out_features=1024),
            # nn.ReLU(),
            )
        # self.unflatten = nn.Unflatten(1, (128, 1, 3))

    def forward(self, x):
        x = x.float()
        # Convolution.
        # x = self.convolution_1(x)
        # print(x.size())
        # x = self.dropout(x)
        # x, indices_1 = self.pooling(x)
        x = self.convolution_2(x)
        # print(x.size())
        # x = self.dropout(x)
        # x, indices_2 = self.pooling(x)
        x = self.convolution_3(x)
        # print(x.size())
        x = self.convolution_4(x)
        # print(x.size())
        x = self.convolution_5(x)
        # print(x.size())
        x = self.convolution_6(x)
        # print(x.size())
        # x = self.convolution_7(x)
        # print(x.size())
        # Autoencoding.
        size_encoding = x.size()
        # x = self.flatten(x)
        x = self.autoencoder(x.flatten())
        x = x.reshape(size_encoding)  #x = self.unflatten(x)
        # Deconvolution.
        # print(x.size())
        # x = self.unpooling(x, indices_2, output_size=size_2)
        # print(x.size())
        # x = self.deconvolution_1(x)
        # x = self.dropout(x)
        # x = x[..., :-1, :]
        # print(x.size())
        # x = self.unpooling(x, indices_1, output_size=size_1)
        # print(x.size())
        # x = self.deconvolution_2(x)
        # x = self.dropout(x)
        # print(x.size())
        x = self.deconvolution_3(x)
        # print(x.size())
        x = self.deconvolution_4(x)
        # print(x.size())
        x = self.deconvolution_5(x)
        # print(x.size())
        x = self.deconvolution_6(x)
        # print(x.size())
        x = self.deconvolution_7(x)
        # print(x.size())
        
        # x = x.view(x.size(0), -1)
        # x = self.linear(x)
        x = x.reshape((1, *OUTPUT_SIZE))

        return x