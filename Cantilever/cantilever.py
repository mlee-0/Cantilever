import glob
import os
import random

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device.')


# Dataset size.
NUMBER_SAMPLES = 20
# Minimum and maximum values of parameters to be varied.
LOAD_RANGE = (1e3, 1e4)
ANGLE_RANGE = (0, 360)
# Increments used to generate samples within the above ranges.
LOAD_STEP = 20
ANGLE_STEP = 5
# Number of possible points within the ranges using the specified increments. The maximum value is excluded from the range.
LOAD_SAMPLE_COUNT = round((LOAD_RANGE[1] - LOAD_RANGE[0]) / LOAD_STEP)
ANGLE_SAMPLE_COUNT = round((ANGLE_RANGE[1] - ANGLE_RANGE[0]) / ANGLE_STEP)
print(f'Input images will be size {LOAD_SAMPLE_COUNT} by {ANGLE_SAMPLE_COUNT}.')

# Folders.
FOLDER_ROOT = 'Cantilever'
FOLDER_INPUTS = os.path.join(FOLDER_ROOT, 'Inputs')
FOLDER_OUTPUTS = os.path.join(FOLDER_ROOT, 'Outputs')

# Training hyperparameters.
BATCH_SIZE = 32
LEARNING_RATE = 0.1


# Return randomly generated sample points for loads and load angles and write them to a text file.
def generate_samples():
    load_samples = np.arange(LOAD_RANGE[0], LOAD_RANGE[1], LOAD_STEP)
    angle_samples = np.arange(ANGLE_RANGE[0], ANGLE_RANGE[1], ANGLE_STEP)
    # Randomize the ordering of the samples.
    random.shuffle(load_samples)
    random.shuffle(angle_samples)
    # Keep only the first N samples.
    load_samples, angle_samples = load_samples[:NUMBER_SAMPLES], angle_samples[:NUMBER_SAMPLES]
    # Write samples to text file.
    text = [f'Load: {load},  Angle: {angle}\n' for load, angle in zip(load_samples, angle_samples)]
    with open(os.path.join(FOLDER_ROOT, 'cantilever_samples.txt'), 'w') as file:
        file.writelines(text)
    return load_samples, angle_samples

# Generate images for each pair of sample points.
def generate_input_images(load_samples, angle_samples):
    for load, angle in zip(load_samples, angle_samples):
        load, angle = int(load), int(angle)
        image = np.zeros((LOAD_SAMPLE_COUNT, ANGLE_SAMPLE_COUNT))
        image[
            int(round((load - LOAD_RANGE[0]) / LOAD_STEP)),
            int(round((angle - ANGLE_RANGE[0]) / ANGLE_STEP))
            ] = 255
        filename = os.path.join(FOLDER_INPUTS, f'cantilever_{load}_{angle}.png')
        with Image.fromarray(image.astype(np.uint8), 'L').convert('1') as file:
            file.save(filename)

# load_samples, angle_samples = generate_samples()
# generate_input_images(load_samples, angle_samples)

class CantileverDataset(Dataset):
    def __init__(self, folder_inputs, folder_outputs):
        self.folder_inputs = folder_inputs
        self.folder_outputs = folder_outputs
        # Get all image filenames in the folder.
        self.filenames = glob.glob(os.path.join(folder_inputs, '*.png'))

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        image_input = np.asarray(Image.open(self.filenames[index]), np.uint8)
        image_output = np.asarray(Image.open('_'.join(['stress', self.filenames[index].split('_')[1:]])), np.uint8)
        sample = {'input': image_input, 'output': image_output}
        return sample

# A CNN that predicts the stress contour in a fixed-size cantilever beam with a point load that varies in magnitude and direction.
class StressContourNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            )
        self.linear = nn.Sequential(
            nn.Linear(),
        )


dataset = CantileverDataset(FOLDER_INPUTS, FOLDER_OUTPUTS)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)