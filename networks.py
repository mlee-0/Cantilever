"""Define network architectures."""


import inspect
import sys
from typing import Tuple

import torch
from torch import nn


def print_model_summary(model) -> None:
    """Print information about a model."""
    print(f"\n{type(model).__name__}")
    print(f"\tTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\tLearnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")


class StressNet(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, model_size: int):
        super().__init__()

        # Number of features in the output of the first layer.
        c = model_size

        self.convolution_1 = nn.Sequential(
            nn.Conv2d(input_channels, c*1, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            nn.BatchNorm2d(c*1),
            nn.ReLU(inplace=True),
        )
        # Reduces both the height and width by half.
        self.convolution_2 = nn.Sequential(
            nn.Conv2d(c*1, c*2, kernel_size=3, stride=2, padding=1, padding_mode='zeros'),
            nn.BatchNorm2d(c*2),
            nn.ReLU(inplace=True),
        )
        # Reduces both the height and width by half.
        self.convolution_3 = nn.Sequential(
            nn.Conv2d(c*2, c*4, kernel_size=3, stride=2, padding=1, padding_mode='zeros'),
            nn.BatchNorm2d(c*4),
            nn.ReLU(inplace=True),
        )

        # Convenience function for creating residual blocks.
        residual = lambda: nn.Sequential(
            nn.Conv2d(c*4, c*4, kernel_size=3, padding='same'),
            nn.BatchNorm2d(c*4),
            nn.ReLU(inplace=False),
            nn.Conv2d(c*4, c*4, kernel_size=3, padding='same'),
            nn.BatchNorm2d(c*4),
        )
        
        self.residual_1 = residual()
        self.residual_2 = residual()
        self.residual_3 = residual()
        self.residual_4 = residual()
        self.residual_5 = residual()

        self.deconvolution_1 = nn.Sequential(
            nn.ConvTranspose2d(c*4, c*2, kernel_size=2, stride=2, padding=0, padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(c*2),
        )
        self.deconvolution_2 = nn.Sequential(
            nn.ConvTranspose2d(c*2, c*1, kernel_size=2, stride=2, padding=0, padding_mode='zeros'),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(c*1),
        )
        self.deconvolution_3 = nn.Sequential(
            nn.ConvTranspose2d(c*1, output_channels, kernel_size=3, stride=1, padding=1, padding_mode='zeros'),
            nn.ReLU(inplace=True),
        )

        print_model_summary(self)

    def forward(self, x):
        x = self.convolution_1(x)
        x = self.convolution_2(x)
        x = self.convolution_3(x)
        
        x = torch.relu(x + self.residual_1(x))
        x = torch.relu(x + self.residual_2(x))
        x = torch.relu(x + self.residual_3(x))
        x = torch.relu(x + self.residual_4(x))
        x = torch.relu(x + self.residual_5(x))

        x = self.deconvolution_1(x)
        x = self.deconvolution_2(x)
        x = self.deconvolution_3(x)

        return x

class StressNet3d(nn.Module):
    def __init__(self, input_channels: int, input_size: Tuple[int, int, int], output_channels: int):
        super().__init__()

        # Number of features in the output of the first layer.
        c = 16

        self.convolution_1 = nn.Sequential(
            nn.Conv3d(input_channels, c*1, kernel_size=9, stride=1, padding="same"),
            nn.BatchNorm3d(c*1),
            nn.ReLU(inplace=True),
        )
        # Reduces all 3 dimensions by half.
        self.convolution_2 = nn.Sequential(
            nn.Conv3d(c*1, c*2, kernel_size=4, stride=2, padding=1, padding_mode="replicate"),
            nn.BatchNorm3d(c*2),
            nn.ReLU(inplace=True),
        )
        # Reduces all 3 dimensions by half.
        self.convolution_3 = nn.Sequential(
            nn.Conv3d(c*2, c*4, kernel_size=4, stride=2, padding=1, padding_mode="replicate"),
            nn.BatchNorm3d(c*4),
            nn.ReLU(inplace=True),
        )

        output_size_residual = [round(_ / 2 / 2) for _ in input_size]
        residual_block = lambda: nn.Sequential(
            nn.Conv3d(c*4, c*4, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(c*4),
            nn.Conv3d(c*4, c*4, kernel_size=3, padding="same"),
            nn.BatchNorm3d(c*4),
        )
        se_block = lambda kernel_size: nn.Sequential(
            nn.AvgPool3d(kernel_size=kernel_size),
            nn.Flatten(),
            nn.Linear(c*4, c*4//16),
            nn.ReLU(inplace=True),
            nn.Linear(c*4//16, c*4),
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
            nn.ConvTranspose3d(c*4, c*2, kernel_size=(4,2,4), stride=(1,2,1), padding=(0,2,0), output_padding=(0,0,0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(c*2),
        )
        self.deconvolution_2 = nn.Sequential(
            nn.ConvTranspose3d(c*2, c*1, kernel_size=(3,2,3), stride=2, padding=1, output_padding=(0,0,0)),
            nn.ReLU(inplace=True),
            nn.BatchNorm3d(c*1),
        )
        self.deconvolution_3 = nn.Sequential(
            nn.Conv3d(c*1, output_channels, kernel_size=9, stride=1, padding="same"),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, value_load: float = None):
        batch_size = x.size(0)

        conv_1 = self.convolution_1(x)
        conv_2 = self.convolution_2(conv_1)
        conv_3 = self.convolution_3(conv_2)
        x = conv_3
        
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


# Store all classes defined in this module in a dictionary, used by the GUI.
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