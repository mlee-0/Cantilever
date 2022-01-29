'''
Networks with different architectures.
'''

import math

import numpy as np
import torch
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

class FullyCnn(BaseCnn):
    def __init__(self):
        super().__init__()
        self.convolution_1 = self.convolution(
            in_channels=INPUT_CHANNELS, out_channels=64, kernel_size=3, stride=2, padding=1,
            )
        self.convolution_2 = self.convolution(
            in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1,
            )
        self.convolution_3 = self.convolution(
            in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1,
            )
        # self.convolution_4 = self.convolution(
        #     in_channels=32, out_channels=64, kernel_size=3, stride=3, padding=1,
        #     )
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        self.linear = nn.Linear(in_features=23296, out_features=np.prod(OUTPUT_SIZE))

    def forward(self, x):
        x = x.float()
        # Convolution.
        x = self.convolution_1(x)
        x = self.convolution_2(x)
        x = self.convolution_3(x)
        # x = self.convolution_4(x)
        # Fully connected.
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        x = x.reshape(-1, *OUTPUT_SIZE)
        return x


class UNet(nn.Module):
    """Modified from: https://towardsdatascience.com/u-net-b229b32b4a71"""

    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            )
        return block
    
    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.ConvTranspose2d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
        return block
    
    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        block = nn.Sequential(
            nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel),
            nn.ReLU(),
            nn.BatchNorm2d(mid_channel),
            nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            )
        return block
    
    def __init__(self, in_channel, out_channel):
        super().__init__()

        # Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=64)
        self.conv_maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_encode2 = self.contracting_block(64, 128)
        self.conv_maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv_encode3 = self.contracting_block(128, 256)
        # self.conv_maxpool3 = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(kernel_size=3, in_channels=128, out_channels=256),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(kernel_size=3, in_channels=256, out_channels=256),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
            )
        
        # Decode
        # self.conv_decode3 = self.expansive_block(512, 256, 128)
        self.conv_decode2 = self.expansive_block(256, 128, 64)
        self.final_layer = self.final_block(128, 64, out_channel)
        
    def crop_and_concat(self, upsampled, bypass, crop=False):
        if crop:
            vertical = (bypass.size()[2] - upsampled.size()[2]) / 2
            horizontal = (bypass.size()[3] - upsampled.size()[3]) / 2
            bypass = nn.functional.pad(
                bypass,
                (-math.floor(horizontal), -math.ceil(horizontal), -math.floor(vertical), -math.ceil(vertical))
                )
        return torch.cat((upsampled, bypass), 1)
    
    def forward(self, x):
        x = x.float()
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        # encode_block3 = self.conv_encode3(encode_pool2)
        # encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool2)
        # Decode
        # decode_block3 = self.crop_and_concat(bottleneck1, encode_block3, crop=True)
        # cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(bottleneck1, encode_block2, crop=True)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1, crop=True)
        final_layer = self.final_layer(decode_block1)
        return final_layer

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
            in_channels=16, out_channels=8, kernel_size=7, stride=3, padding=1,
            )
        self.deconvolution_2 = self.deconvolution(
            in_channels=8, out_channels=4, kernel_size=7, stride=3, padding=1,
            )
        self.deconvolution_3 = self.deconvolution(
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
        self.dropout = nn.Dropout(0.8)
        
        self.flatten = nn.Flatten()
        self.autoencoder = nn.Sequential(
            nn.Linear(in_features=48, out_features=24),
            nn.Linear(in_features=24, out_features=12),
            nn.Linear(in_features=12, out_features=24),
            nn.Linear(in_features=24, out_features=48),
            )
        self.unflatten = nn.Unflatten(1, (16, 1, 3))

    def forward(self, x):
        x = x.float()
        x = nn.functional.pad(x, (0, 1, 1, 1))  # Add one row to bottom, and one column on both left and right
        # Convolution.
        x = self.convolution_1(x)
        x = self.dropout(x)
        size_1 = x.size()
        x, indices_1 = self.pooling(x)
        x = self.convolution_2(x)
        x = self.dropout(x)
        size_2 = x.size()
        x, indices_2 = self.pooling(x)
        # Autoencoding.
        size_encoding = x.size()
        x = self.flatten(x)
        x = self.autoencoder(x)
        x = self.unflatten(x)  #x = x.reshape(size_encoding)
        # Deconvolution.
        x = self.unpooling(x, indices_2, output_size=size_2)
        x = self.deconvolution_1(x)
        x = self.dropout(x)
        x = self.unpooling(x, indices_1, output_size=size_1)
        x = self.deconvolution_2(x)
        x = self.dropout(x)
        x = x[..., :-1, :]  # Remove last row added before first convolution

        return x