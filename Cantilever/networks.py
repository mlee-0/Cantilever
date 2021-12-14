'''
Networks with different architectures.
'''

import numpy as np
from torch import nn

from setup import INPUT_CHANNELS, INPUT_SIZE, OUTPUT_CHANNELS, OUTPUT_SIZE


class BaseCnn(nn.Module):
    """Base class with common helper methods used by all CNN classes."""

    # Return a sequence of layers related to convolution.
    @staticmethod
    def convolution(in_channels, out_channels, **kwargs):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs, padding_mode='replicate'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )
    
    # Return a sequence of layers related to deconvolution.
    @staticmethod
    def deconvolution(in_channels, out_channels, **kwargs):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            )

class UNetCnn(BaseCnn):
    def __init__(self):
        super().__init__()
        self.convolution_1 = self.convolution(
            in_channels=INPUT_CHANNELS, out_channels=8, kernel_size=3, stride=3, padding=1,
            )
        self.convolution_2 = self.convolution(
            in_channels=8, out_channels=16, kernel_size=3, stride=3, padding=1,
            )
        self.deconvolution_1 = self.deconvolution(
            in_channels=16, out_channels=8, kernel_size=3, stride=3, padding=1,
            )
        self.deconvolution_2 = self.deconvolution(
            in_channels=8, out_channels=4, kernel_size=3, stride=3, padding=1,
            )
        self.deconvolution_3 = self.deconvolution(
            in_channels=4, out_channels=OUTPUT_CHANNELS, kernel_size=3, stride=3, padding=1,
            )
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpooling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.linear = nn.Linear(in_features=55, out_features=np.prod(OUTPUT_SIZE))

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

class AutoencoderCnn(BaseCnn):
    def __init__(self):
        super().__init__()
        
        self.convolution_1 = self.convolution(
            in_channels=INPUT_CHANNELS, out_channels=8, kernel_size=3, stride=3, padding=0
            )
        self.convolution_2 = self.convolution(
            in_channels=8, out_channels=16, kernel_size=3, stride=3, padding=1
            )
        self.deconvolution_1 = self.deconvolution(
            in_channels=16, out_channels=8, kernel_size=3, stride=3, padding=1, output_padding=(1,0)
            )
        self.deconvolution_2 = self.deconvolution(
            in_channels=8, out_channels=OUTPUT_CHANNELS, kernel_size=3, stride=3, padding=0, output_padding=(0,1)
            )
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpooling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.autoencoder = nn.Sequential(
            nn.Linear(in_features=48, out_features=24),
            nn.Linear(in_features=24, out_features=12),
            nn.Linear(in_features=12, out_features=24),
            nn.Linear(in_features=24, out_features=48),
            )

    def forward(self, x):
        x = x.float()
        x = nn.functional.pad(x, (0, 1, 1, 1))  # Add one row to bottom, and one column on both left and right
        # Convolution.
        x = self.convolution_1(x)
        size_1 = x.size()
        x, indices_1 = self.pooling(x)
        x = self.convolution_2(x)
        size_2 = x.size()
        x, indices_2 = self.pooling(x)
        # Autoencoding.
        size_encoding = x.size()
        x = self.autoencoder(x.flatten())
        x = x.reshape(size_encoding)
        # Deconvolution.
        x = self.unpooling(x, indices_2, output_size=size_2)
        x = self.deconvolution_1(x)
        x = self.unpooling(x, indices_1, output_size=size_1)
        x = self.deconvolution_2(x)
        x = x[..., :-1, :]  # Remove last row added before first convolution

        return x