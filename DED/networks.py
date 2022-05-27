'''
Networks with different architectures.
'''

import inspect
import math
import sys

import numpy as np
import torch
from torch import nn

from setup import INPUT_CHANNELS, INPUT_SIZE, OUTPUT_CHANNELS, OUTPUT_SIZE, EMBEDDING_SIZE


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
    def __init__(self, input_channels: int, output_size: int):
        super().__init__()
        self.output_size = output_size

        PADDING_MODE = "replicate"
        KERNEL_SIZE = 3
        STRIDE = 2
        PADDING = 1

        self.convolution = nn.Sequential(
            nn.Conv2d(in_channels=input_channels * 1, out_channels=input_channels * 2, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING, padding_mode=PADDING_MODE),
            nn.BatchNorm2d(input_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False),

            nn.Conv2d(in_channels=input_channels * 2, out_channels=input_channels * 4, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING, padding_mode=PADDING_MODE),
            nn.BatchNorm2d(input_channels * 4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False),

            nn.Conv2d(in_channels=input_channels * 4, out_channels=input_channels * 8, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING, padding_mode=PADDING_MODE),
            nn.BatchNorm2d(input_channels * 8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False),
            
            # nn.Conv2d(in_channels=input_channels * 8, out_channels=input_channels * 16, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING, padding_mode=PADDING_MODE),
            # nn.BatchNorm2d(input_channels * 16),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False),
            
            # nn.Conv2d(in_channels=input_channels * 16, out_channels=input_channels * 32, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING, padding_mode=PADDING_MODE),
            # nn.BatchNorm2d(input_channels * 32),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False),
        )

        self.linear = nn.Linear(in_features=24, out_features=output_size)

        # self.convolution_1 = convolution(
        #     in_channels=input_channels, out_channels=input_channels*2, kernel_size=3, stride=2, padding=1,
        #     )
        # self.convolution_2 = convolution(
        #     in_channels=input_channels*2, out_channels=input_channels*4, kernel_size=3, stride=2, padding=1,
        #     )
        # self.convolution_3 = convolution(
        #     in_channels=input_channels*4, out_channels=input_channels*8, kernel_size=3, stride=2, padding=1,
        #     )
        # self.convolution_4 = convolution(
        #     in_channels=input_channels*8, out_channels=input_channels*16, kernel_size=3, stride=2, padding=1,
        #     )
        # self.convolution_5 = convolution(
        #     in_channels=input_channels*16, out_channels=input_channels*32, kernel_size=3, stride=2, padding=1,
        #     )
        # self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=False)
        # self.linear = nn.Linear(in_features=288, out_features=output_size)

    def forward(self, x):
        x = x.float()

        x = self.convolution(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        x = x.reshape(-1, self.output_size, 1, 1)

        # # Convolution.
        # x = self.convolution_1(x)
        # x = self.pooling(x)
        # x = self.convolution_2(x)
        # x = self.pooling(x)
        # x = self.convolution_3(x)
        # x = self.pooling(x)
        # # x = self.convolution_4(x)
        # # x = self.pooling(x)
        # # x = self.convolution_5(x)
        # # x = self.pooling(x)
        # # Fully connected.
        # x = x.view(x.size(0), -1)
        # x = self.linear(x)
        # x = x.reshape(-1, *OUTPUT_SIZE)

        return x


# GAN based on: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

class GanGenerator(nn.Module):
    def __init__(self, input_channels: int, number_features: int, output_channels: int):
        super().__init__()

        self.latent_size = input_channels

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(self.latent_size, number_features * 8, kernel_size=4, stride=1, padding=0, bias=False),
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

    def forward(self, x, label: int):
        label_embedding = nn.Embedding(EMBEDDING_SIZE, self.latent_size)(label)
        return self.layers(x * label_embedding.flatten().reshape(x.size()))

class GanDiscriminator(nn.Module):
    def __init__(self, number_features: int, output_channels: int):
        super().__init__()

        self.layers = nn.Sequential(
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

            nn.Conv2d(number_features * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x, label: int):
        label_embedding = nn.Embedding(EMBEDDING_SIZE, x.numel())(label)
        label_embedding = torch.reshape(label_embedding.flatten(), x.size())
        # Concatenate along the channel dimension.
        return self.layers(torch.cat((x, label_embedding), axis=1))


# Conditional GAN based on: https://learnopencv.com/conditional-gan-cgan-in-pytorch-and-tensorflow/
class ConditionalGanGenerator(nn.Module):
    def __init__(self, input_channels: int, number_features: int, output_channels: int):
        super().__init__()

        EMBEDDING_SIZE = 100

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(input_channels, number_features * 8, kernel_size=4, stride=1, padding=0, bias=False),
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

    def forward(self, x):
        return self.layers(x)


class ConditionalGanDiscriminator(nn.Module):
    def __init__(self, number_features: int, output_channels: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(output_channels, number_features * 1, kernel_size=4, stride=2, padding=1, bias=False),
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

            nn.Conv2d(number_features * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        return self.layers(x)


class Nie(nn.Module):
    def __init__(self):
        super().__init__()

        TRACK_RUNNING_STATS = False
        MOMENTUM = 0.1

        self.convolution_1 = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, 16, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(16, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
            nn.ReLU(inplace=True),
        )
        self.convolution_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=(2,2), padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(32, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
            nn.ReLU(inplace=True),
        )
        self.convolution_3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=(2,2), padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(64, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
            nn.ReLU(inplace=True),
        )
        self.se_1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
        )
        self.se_2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(OUTPUT_SIZE[1]//4, OUTPUT_SIZE[2]//4)),
            nn.Flatten(),
            nn.Linear(64, 4),
            nn.ReLU(inplace=True),
            nn.Linear(4, OUTPUT_SIZE[2]//4),
            nn.Sigmoid(),
        )
        self.deconvolution_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=(2,2), padding=1, output_padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
        )
        self.deconvolution_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=(2,2), padding=1, output_padding=(1,1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
        )
        self.deconvolution_3 = nn.Sequential(
            nn.Conv2d(16, OUTPUT_CHANNELS, kernel_size=9, stride=1, padding=4),  # output_padding=(0,0)),
            # nn.BatchNorm2d(OUTPUT_CHANNELS, momentum=MOMENTUM, track_running_stats=TRACK_RUNNING_STATS),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        batch_size = x.size()[0]
        x = x.float()
        # print(x.size())
        x = self.convolution_1(x)
        # print(x.size())
        x = self.convolution_2(x)
        # print(x.size())
        x = self.convolution_3(x)
        # print(x.size())
        se_1 = self.se_1(x)
        se_2 = self.se_2(se_1)
        x = x + se_1 * se_2.reshape((batch_size, 1, 1, -1))
        se_1 = self.se_1(x)
        se_2 = self.se_2(se_1)
        x = x + se_1 * se_2.reshape((batch_size, 1, 1, -1))
        se_1 = self.se_1(x)
        se_2 = self.se_2(se_1)
        x = x + se_1 * se_2.reshape((batch_size, 1, 1, -1))
        se_1 = self.se_1(x)
        se_2 = self.se_2(se_1)
        x = x + se_1 * se_2.reshape((batch_size, 1, 1, -1))
        se_1 = self.se_1(x)
        se_2 = self.se_2(se_1)
        x = x + se_1 * se_2.reshape((batch_size, 1, 1, -1))
        # print(x.size())
        x = self.deconvolution_1(x)
        # print(x.size())
        x = self.deconvolution_2(x)
        # print(x.size())
        x = self.deconvolution_3(x)
        # print(x.size())
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
            in_channels=INPUT_CHANNELS, out_channels=8, kernel_size=3, stride=2, padding=1
            )
        self.convolution_3 = convolution(
            in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1
            )
        self.convolution_4 = convolution(
            in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1
            )
        self.convolution_5 = convolution(
            in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1
            )
        self.convolution_6 = convolution(
            in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
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
            in_channels=16, out_channels=OUTPUT_CHANNELS, kernel_size=1, stride=2, padding=0, output_padding=(0,1)
            )
        # self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        # self.unpooling = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0)
        self.linear = nn.Linear(in_features=1250, out_features=1250)
        
        self.flatten = nn.Flatten()
        self.autoencoder = nn.Sequential(
            # nn.ReLU(),
            nn.Linear(in_features=1024, out_features=64),
            nn.Linear(in_features=64, out_features=3),
            # nn.Linear(in_features=32, out_features=16),
            # nn.Linear(in_features=12, out_features=6),
            # nn.Linear(in_features=6, out_features=3),
            # nn.Linear(in_features=3, out_features=6),
            # nn.Linear(in_features=6, out_features=12),
            # nn.Linear(in_features=16, out_features=32),
            nn.Linear(in_features=3, out_features=64),
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
        
        # x = x.view(x.size(0), -1)
        # x = self.linear(x)
        x = x.reshape((1, *OUTPUT_SIZE))

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