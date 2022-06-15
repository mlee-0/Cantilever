"""
Networks with different architectures.
"""

import inspect
import sys
from typing import Tuple

import numpy as np
import torch
from torch import nn


class Nie(nn.Module):
    """Based on: https://arxiv.org/pdf/1808.08914.pdf"""

    def __init__(self, input_channels: int, input_size: Tuple[int, int], output_channels: int):
        super().__init__()

        TRACK_RUNNING_STATS = False
        MOMENTUM = 0.1

        # Number of features in the output of the first layer.
        nf = 32

        self.convolution_1 = nn.Sequential(
            nn.Conv2d(input_channels, nf*1, kernel_size=9, stride=1, padding="same"),
            nn.BatchNorm2d(nf*1, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
            nn.ReLU(inplace=True),
        )
        # Reduces both the height and width by half.
        self.convolution_2 = nn.Sequential(
            nn.Conv2d(nf*1, nf*2, kernel_size=4, stride=2, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(nf*2, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
            nn.ReLU(inplace=True),
        )
        # Reduces both the height and width by half.
        self.convolution_3 = nn.Sequential(
            nn.Conv2d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(nf*4, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
            nn.ReLU(inplace=True),
        )

        # Convenience functions for returning residual blocks and squeeze-and-excitation blocks.
        residual_block = lambda: nn.Sequential(
            nn.Conv2d(nf*4, nf*4, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(nf*4, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
            nn.Conv2d(nf*4, nf*4, kernel_size=3, padding="same"),
            nn.BatchNorm2d(nf*4, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
        )
        se_block = lambda kernel_size: nn.Sequential(
            nn.AvgPool2d(kernel_size=kernel_size),
            nn.Flatten(),
            nn.Linear(nf*4, nf*4//16),
            nn.ReLU(inplace=True),
            nn.Linear(nf*4//16, nf*4),
            nn.Sigmoid(),
        )
        
        output_size_residual = (round(input_size[0] / 2 / 2), round(input_size[1] / 2 / 2))
        self.residual_1 = residual_block()
        self.se_1 = se_block(output_size_residual)
        self.residual_2 = residual_block()
        self.se_2 = se_block(output_size_residual)
        self.residual_3 = residual_block()
        self.se_3 = se_block(output_size_residual)
        self.residual_4 = residual_block()
        self.se_4 = se_block(output_size_residual)
        self.residual_5 = residual_block()
        self.se_5 = se_block(output_size_residual)

        self.deconvolution_1 = nn.Sequential(
            nn.ConvTranspose2d(nf*4, nf*2, kernel_size=(4,2), stride=(1,2), padding=(0,2), output_padding=(0,0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(nf*2, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
        )
        self.deconvolution_2 = nn.Sequential(
            nn.ConvTranspose2d(nf*2, nf*1, kernel_size=(3,2), stride=2, padding=1, output_padding=(0,0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(nf*1, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
        )
        self.deconvolution_3 = nn.Sequential(
            nn.Conv2d(nf*1, output_channels, kernel_size=9, stride=1, padding="same"),
            # nn.BatchNorm2d(OUTPUT_CHANNELS, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, value_load: float = None):
        batch_size = x.size()[0]

        x = x.float()
        conv_1 = self.convolution_1(x)
        conv_2 = self.convolution_2(conv_1)
        x = self.convolution_3(conv_2)
        
        residual = self.residual_1(x)
        se = self.se_1(residual)
        x = x + residual * se.reshape((batch_size, -1, 1, 1))
        residual = self.residual_2(x)
        se = self.se_2(residual)
        x = x + residual * se.reshape((batch_size, -1, 1, 1))
        residual = self.residual_3(x)
        se = self.se_3(residual)
        x = x + residual * se.reshape((batch_size, -1, 1, 1))
        residual = self.residual_4(x)
        se = self.se_4(residual)
        x = x + residual * se.reshape((batch_size, -1, 1, 1))
        residual = self.residual_5(x)
        se = self.se_5(residual)
        x = x + residual * se.reshape((batch_size, -1, 1, 1))

        # Add load value.
        if value_load is not None:
            x = x + value_load

        x = self.deconvolution_1(x)
        # x = self.deconvolution_2(conv_2[..., 1:-1, 2:-2] + x)
        # x = self.deconvolution_3(conv_1[..., 2:-3, 5:-5] + x)
        # x = self.deconvolution_2(torch.cat((conv_2[..., 1:-1, 2:-2], x), dim=1))
        # x = self.deconvolution_3(torch.cat((conv_1[..., 2:-3, 5:-5], x), dim=1))
        x = self.deconvolution_2(x)
        x = self.deconvolution_3(x)
    
        return x

class Nie3d(nn.Module):
    def __init__(self, input_channels: int, input_size: Tuple[int, int, int], output_channels: int):
        super().__init__()

        TRACK_RUNNING_STATS = False
        MOMENTUM = 0.1

        # Number of features in the output of the first layer.
        nf = 16

        self.convolution_1 = nn.Sequential(
            nn.Conv3d(input_channels, nf*1, kernel_size=9, stride=1, padding="same"),
            nn.BatchNorm3d(nf*1, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
            nn.ReLU(inplace=True),
        )
        # Reduces all 3 dimensions by half.
        self.convolution_2 = nn.Sequential(
            nn.Conv3d(nf*1, nf*2, kernel_size=4, stride=2, padding=1, padding_mode="replicate"),
            nn.BatchNorm3d(nf*2, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
            nn.ReLU(inplace=True),
        )
        # Reduces all 3 dimensions by half.
        self.convolution_3 = nn.Sequential(
            nn.Conv3d(nf*2, nf*4, kernel_size=4, stride=2, padding=1, padding_mode="replicate"),
            nn.BatchNorm3d(nf*4, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
            nn.ReLU(inplace=True),
        )

        output_size_residual = [round(_ / 2 / 2) for _ in input_size]
        residual_block = lambda: nn.Sequential(
            nn.Conv3d(nf*4, nf*4, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(nf*4, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
            nn.Conv3d(nf*4, nf*4, kernel_size=3, padding="same"),
            nn.BatchNorm3d(nf*4, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
        )
        se_block = lambda kernel_size: nn.Sequential(
            nn.AvgPool3d(kernel_size=kernel_size),
            nn.Flatten(),
            nn.Linear(nf*4, nf*4//16),
            nn.ReLU(inplace=True),
            nn.Linear(nf*4//16, nf*4),
            nn.Sigmoid(),
        )
        
        self.residual_1 = residual_block()
        self.se_1 = se_block(output_size_residual)
        self.residual_2 = residual_block()
        self.se_2 = se_block(output_size_residual)
        self.residual_3 = residual_block()
        self.se_3 = se_block(output_size_residual)
        self.residual_4 = residual_block()
        self.se_4 = se_block(output_size_residual)
        self.residual_5 = residual_block()
        self.se_5 = se_block(output_size_residual)

        self.deconvolution_1 = nn.Sequential(
            nn.ConvTranspose3d(nf*4, nf*2, kernel_size=(4,2,4), stride=(1,2,1), padding=(0,2,0), output_padding=(0,0,0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(nf*2, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
        )
        self.deconvolution_2 = nn.Sequential(
            nn.ConvTranspose3d(nf*2, nf*1, kernel_size=(3,2,3), stride=2, padding=1, output_padding=(0,0,0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(nf*1, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
        )
        self.deconvolution_3 = nn.Sequential(
            nn.Conv3d(nf*1, output_channels, kernel_size=9, stride=1, padding="same"),
            # nn.BatchNorm3d(OUTPUT_CHANNELS, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, value_load: float = None):
        batch_size = x.size()[0]

        x = x.float()
        conv_1 = self.convolution_1(x)
        conv_2 = self.convolution_2(conv_1)
        x = self.convolution_3(conv_2)
        
        residual = self.residual_1(x)
        se = self.se_1(residual)
        x = x + residual * se.reshape((batch_size, -1, 1, 1, 1))
        residual = self.residual_2(x)
        se = self.se_2(residual)
        x = x + residual * se.reshape((batch_size, -1, 1, 1, 1))
        residual = self.residual_3(x)
        se = self.se_3(residual)
        x = x + residual * se.reshape((batch_size, -1, 1, 1, 1))
        residual = self.residual_4(x)
        se = self.se_4(residual)
        x = x + residual * se.reshape((batch_size, -1, 1, 1, 1))
        residual = self.residual_5(x)
        se = self.se_5(residual)
        x = x + residual * se.reshape((batch_size, -1, 1, 1, 1))

        # Add load value.
        if value_load is not None:
            x = x + value_load

        x = self.deconvolution_1(x)
        x = self.deconvolution_2(x)
        x = self.deconvolution_3(x)
    
        return x

class FullyCnn(nn.Module):
    def __init__(self, input_channels: int, output_size: Tuple[int, int], output_channels: int):
        super().__init__()
        self.output_size = output_size
        self.output_channels = output_channels

        self.convolution_1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.convolution_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        self.linear = nn.Linear(in_features=32*5*10, out_features=np.prod(output_size))

    def forward(self, x):
        x = x.float()

        x = self.convolution_1(x)
        x = self.convolution_2(x)
        x = self.linear(x.view(x.shape[0], -1))
        
        return x.reshape((x.shape[0], self.output_channels, *self.output_size))

class UNetCnn(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()

        # Preserves height and length dimensions.
        self.encoding_1 = self.two_convolutions(input_channels, 16)
        # Reduces both the height and width by half.
        self.downsize_1 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        # Preserves height and length dimensions.
        self.encoding_2 = self.two_convolutions(16, 32)
        # Reduces both the height and width by half.
        self.downsize_2 = nn.MaxPool2d(kernel_size=4, stride=2, padding=1)
        # Preserves height and length dimensions.
        self.encoding_3 = self.two_convolutions(32, 64)

        self.upsize_2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=(2,3), output_padding=(1,1))
        self.decoding_2 = self.two_convolutions(32*2, 32)
        self.upsize_1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=(1,2), output_padding=(0,1))
        self.decoding_1 = self.two_convolutions(16*2, 16)

        self.final_convolution = nn.Conv2d(in_channels=16, out_channels=output_channels, kernel_size=1, stride=1, padding="same")

    def two_convolutions(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Return a Sequential with two convolution layers that preserve the height and length dimensions."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.float()

        encoding_1 = self.encoding_1(x)
        x = self.downsize_1(encoding_1)
        encoding_2 = self.encoding_2(x)
        x = self.downsize_2(encoding_2)
        x = self.encoding_3(x)
        upsize_2 = self.upsize_2(x)
        x = self.decoding_2(torch.cat(
            (encoding_2[..., 1:-1, 2:-2], upsize_2), dim=1
        ))
        upsize_1 = self.upsize_1(x)
        x = self.decoding_1(torch.cat(
            (encoding_1[..., 2:-3, 5:-5], upsize_1), dim=1
        ))
        x = self.final_convolution(x)
        
        return x

class AutoencoderCnn(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        
        self.convolution_1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.convolution_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=(2,2), padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.convolution_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(2,2), padding=1, padding_mode="replicate"),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.autoencoder = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(in_features=5*10*64, out_features=64*8),
            nn.Linear(in_features=64*8, out_features=64*4),
            nn.Linear(in_features=64*4, out_features=64*2),
            # nn.Linear(in_features=32, out_features=16),
            # nn.Linear(in_features=16, out_features=8),
            # nn.Linear(in_features=6, out_features=3),
            # nn.Linear(in_features=3, out_features=6),
            # nn.Linear(in_features=8, out_features=16),
            # nn.Linear(in_features=16, out_features=32),
            nn.Linear(in_features=64*2, out_features=64*4),
            nn.Linear(in_features=64*4, out_features=64*8),
            nn.Linear(in_features=64*8, out_features=5*10*64),
            # nn.ReLU(),
        )

        self.deconvolution_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=(2,2), padding=(2,3), output_padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
        )
        self.deconvolution_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=(2,2), padding=(1,2), output_padding=(0,1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
        )
        self.deconvolution_3 = nn.Sequential(
            nn.Conv2d(16, output_channels, kernel_size=9, stride=1, padding=4),  # output_padding=(0,0)),
            # nn.BatchNorm2d(output_channels, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
            nn.ReLU(inplace=True),
        )

        # self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # self.unpooling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # self.dropout = nn.Dropout(0)
        # self.linear = nn.Linear(in_features=1250, out_features=1250)
        
        # self.flatten = nn.Flatten()
        # self.autoencoder = nn.Sequential(
        #     # nn.ReLU(),
        #     nn.Linear(in_features=1024, out_features=64),
        #     nn.Linear(in_features=64, out_features=3),
        #     # nn.Linear(in_features=32, out_features=16),
        #     # nn.Linear(in_features=12, out_features=6),
        #     # nn.Linear(in_features=6, out_features=3),
        #     # nn.Linear(in_features=3, out_features=6),
        #     # nn.Linear(in_features=6, out_features=12),
        #     # nn.Linear(in_features=16, out_features=32),
        #     nn.Linear(in_features=3, out_features=64),
        #     nn.Linear(in_features=64, out_features=1024),
        #     # nn.ReLU(),
        #     )
        # # self.unflatten = nn.Unflatten(1, (128, 1, 3))

    def forward(self, x):
        x = x.float()
        # Convolution.
        x = self.convolution_1(x)
        # print(x.size())
        # x = self.dropout(x)
        # x, indices_1 = self.pooling(x)
        x = self.convolution_2(x)
        # print(x.size())
        # x = self.dropout(x)
        # x, indices_2 = self.pooling(x)
        x = self.convolution_3(x)
        # print(x.size())
        # x = self.convolution_4(x)
        # print(x.size())
        # x = self.convolution_5(x)
        # print(x.size())
        # x = self.convolution_6(x)
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
        x = self.deconvolution_1(x)
        # x = self.dropout(x)
        # x = x[..., :-1, :]
        # print(x.size())
        # x = self.unpooling(x, indices_1, output_size=size_1)
        # print(x.size())
        x = self.deconvolution_2(x)
        # x = self.dropout(x)
        # print(x.size())
        x = self.deconvolution_3(x)
        # print(x.size())
        # x = self.deconvolution_4(x)
        # print(x.size())
        # x = self.deconvolution_5(x)
        # print(x.size())
        # x = self.deconvolution_6(x)
        # print(x.size())
        
        # x = x.view(x.size(0), -1)
        # x = self.linear(x)
        # x = x.reshape((1, *OUTPUT_SIZE))

        return x

# GAN based on: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
class GanGenerator(nn.Module):
    def __init__(self, input_channels: int, number_features: int, output_channels: int):
        super().__init__()

        self.latent_size = input_channels

        self.deconvolution_1 = nn.Sequential(
            nn.ConvTranspose2d(self.latent_size, number_features * 4, kernel_size=(2,4), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(number_features * 4),
            nn.ReLU(inplace=True),
        )
        self.deconvolution_2 = nn.Sequential(
            nn.ConvTranspose2d(number_features * 4, number_features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(number_features * 2),
            nn.ReLU(inplace=True),
        )
        self.deconvolution_3 = nn.Sequential(
            nn.ConvTranspose2d(number_features * 2, number_features * 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(number_features * 1),
            nn.ReLU(inplace=True),
        )
        self.deconvolution_4 = nn.Sequential(
            nn.ConvTranspose2d(number_features * 1, output_channels, kernel_size=3, stride=2, padding=(1,2), output_padding=(0,1), bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.deconvolution_1(x)
        x = self.deconvolution_2(x)
        x = self.deconvolution_3(x)
        x = self.deconvolution_4(x)
        return x

class GanDiscriminator(nn.Module):
    def __init__(self, number_features: int, output_channels: int):
        super().__init__()

        self.convolution_1 = nn.Sequential(
            nn.Conv2d(output_channels, number_features * 1, kernel_size=3, stride=2, padding=(1,2), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.convolution_2 = nn.Sequential(
            nn.Conv2d(number_features * 1, number_features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(number_features * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.convolution_3 = nn.Sequential(
            nn.Conv2d(number_features * 2, number_features * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(number_features * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        # Reduces image size to (1, 1).
        self.convolution_4 = nn.Sequential(
            nn.Conv2d(number_features * 4, 1, kernel_size=(2,4), stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.convolution_1(x)
        x = self.convolution_2(x)
        x = self.convolution_3(x)
        x = self.convolution_4(x)
        return x


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