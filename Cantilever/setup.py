"""
Information about parameters and functions for reading and writing files.
"""


import colorsys
from dataclasses import dataclass
import glob
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

try:
    from google.colab import drive  # type: ignore (forces Pylance in VS Code to ignore the missing import error)
except ModuleNotFoundError:
    GOOGLE_COLAB = False
else:
    GOOGLE_COLAB = True
    drive.mount("/content/drive")

@dataclass
class Parameter:
    """A dataclass that stores settings for a parameter."""
    # Name of the parameter.
    name: str
    # The minimum and maximum values between which samples are generated.
    low: float
    high: float
    # Spacing between adjacent values.
    step: int
    # Units for the parameter.
    units: str = ""
    # Number of decimal places to which sample values are rounded.
    precision: int = 0

# Settings for each parameter.
length = Parameter(low=2, high=4, step=0.1, precision=1, name="Length", units="m")
height = Parameter(low=1, high=2, step=0.1, precision=1, name="Height", units="m")
width = Parameter(low=1, high=2, step=0.1, precision=1, name="Width", units="m")
elastic_modulus = Parameter(low=190, high=210, step=1, precision=0, name="Elastic Modulus", units="GPa")
load = Parameter(low=500, high=1000, step=5, precision=0, name="Load", units="N")
angle_1 = Parameter(low=0, high=360, step=1, precision=0, name="Angle XY", units="Degrees")
angle_2 = Parameter(low=0, high=360, step=1, precision=0, name="Angle XZ", units="Degrees")

# Names of quantities that are derived from randomly generated values.
KEY_SAMPLE_NUMBER = "Sample Number"
KEY_X_LOAD = "Load X"
KEY_Y_LOAD = "Load Y"
KEY_Z_LOAD = "Load Z"
KEY_NODES_LENGTH = "Nodes Length"
KEY_NODES_HEIGHT = "Nodes Height"
KEY_NODES_WIDTH = "Nodes Width"

# Use the 3D FEA dataset.
is_3d = True

# Folders and files.
FOLDER_ROOT = "Cantilever" if not GOOGLE_COLAB else "drive/My Drive/Colab Notebooks"
if is_3d:
    FOLDER_TRAIN_LABELS = os.path.join(FOLDER_ROOT, "Train Labels 3D")
    FOLDER_TEST_LABELS = os.path.join(FOLDER_ROOT, "Test Labels 3D")
else:
    FOLDER_TRAIN_LABELS = os.path.join(FOLDER_ROOT, "Train Labels")
    FOLDER_TEST_LABELS = os.path.join(FOLDER_ROOT, "Test Labels")
FOLDER_RESULTS = os.path.join(FOLDER_ROOT, "Results")
FILENAME_SAMPLES_TRAIN = "samples_train.csv"
FILENAME_SAMPLES_TEST = "samples_test.csv"

# Number of digits used for numerical file names.
NUMBER_DIGITS = 6

# Size of input images (channel-height-width). Must have the same aspect ratio as the largest possible cantilever geometry.
INPUT_CHANNELS = 5 if is_3d else 3
INPUT_SIZE = (INPUT_CHANNELS, round(height.high / height.step), round(length.high / length.step))
assert (INPUT_SIZE[1] / INPUT_SIZE[2]) == (height.high / length.high), "Input image size must match aspect ratio of cantilever: {height.high}:{length.high}."
# Size of output images (channel-height-width) produced by the network. Each pixel corresponds to a single node in the FEA mesh.
OUTPUT_CHANNELS = 15 if is_3d else 1
OUTPUT_SIZE = (OUTPUT_CHANNELS, 15, 30)


def plot_histogram(values: np.ndarray, title=None) -> None:
    plt.figure()
    plt.hist(values, bins=100, rwidth=0.75, color='#0095ff')
    if title:
        plt.title(title)
    plt.show()

def split_training_validation(number_samples: int, training_split: float) -> Tuple[int, int]:
    """Return the number of samples in the training and validation datasets for the given ratio."""
    assert 0 < training_split < 1, f"Invalid training split: {training_split}."
    training_size = round(training_split * number_samples)
    validation_size = round((1 - training_split) * number_samples)
    assert training_size + validation_size == number_samples
    return training_size, validation_size

def generate_input_images(samples: pd.DataFrame, is_3d: bool) -> np.ndarray:
    """Return a 4D array of images for each of the specified sample values, with dimensions: [samples, channels, height, width]."""
    DATA_TYPE = np.uint8

    number_samples = len(samples)
    inputs = np.full((number_samples, *INPUT_SIZE), 0, dtype=DATA_TYPE)
    for i in range(number_samples):
        pixel_length = int(samples[KEY_NODES_LENGTH][i])
        pixel_height = int(samples[KEY_NODES_HEIGHT][i])
        pixel_width = int(samples[KEY_NODES_WIDTH][i])
        
        channels = []
        h, w = INPUT_SIZE[1:]

        # Create a channel with a white rectangle representing the length and height of the cantilever.
        channel = np.zeros((h, w), dtype=DATA_TYPE)
        channel[:pixel_height, :pixel_length] = 255
        channels.append(channel)

        # Create a channel with a white rectangle representing the height and width of the cantilever.
        if not is_3d:
            channel = np.zeros((h, w), dtype=DATA_TYPE)
            channel[:pixel_height, :pixel_width] = 255
            channels.append(channel)

        # Create a channel with a gray line of pixels representing the load magnitude and angle 1.
        channel = np.zeros((h, w), dtype=DATA_TYPE)
        r = np.arange(max(channel.shape))
        x = r * np.cos(samples[angle_1.name][i] * np.pi/180) + w/2
        y = r * np.sin(samples[angle_1.name][i] * np.pi/180) + h/2
        x = x.astype(int)
        y = y.astype(int)
        inside_image = (x >= 0) * (x < w) * (y >= 0) * (y < h)
        channel[y[inside_image], x[inside_image]] = 255 * (samples[load.name][i] / load.high)
        channel = np.flipud(channel)
        channels.append(channel)
        
        # Create a channel with a gray line of pixels representing the load magnitude and angle 2.
        if not is_3d:
            channel = np.zeros((h, w), dtype=DATA_TYPE)
            r = np.arange(max(channel.shape))
            x = r * np.cos(samples[angle_2.name][i] * np.pi/180) + w/2
            y = r * np.sin(samples[angle_2.name][i] * np.pi/180) + h/2
            x = x.astype(int)
            y = y.astype(int)
            inside_image = (x >= 0) * (x < w) * (y >= 0) * (y < h)
            channel[y[inside_image], x[inside_image]] = 255 * (samples[load.name][i] / load.high)
            channel = np.flipud(channel)
            channels.append(channel)
        
        # Create a channel with the elastic modulus distribution.
        channel = np.zeros((h, w), dtype=DATA_TYPE)
        channel[:pixel_height, :pixel_length] = 255 * (samples[elastic_modulus.name][i] / elastic_modulus.high)
        channels.append(channel)
        
        # # Create a channel with a vertical white line whose position represents the load magnitude. Leftmost column is 0, rightmost column is the maximum magnitude.
        # # image[0, :pixel_height, :pixel_length] = 255 * samples[load.name][i] / load.high
        # image[0, :, round(image.shape[2] * samples[load.name][i] / load.high) - 1] = 255
        
        # # Create a channel with the fixed boundary conditions.
        # image[3, :pixel_height, 0] = 255

        # Create the image and append it to the list.
        inputs[i, ...] = np.stack(channels, axis=0)
    
    return inputs

def generate_label_images(samples: pd.DataFrame, folder: str, is_3d: bool) -> np.ndarray:
    """Return a 4D array of images for the FEA text files found in the specified folder that correspond to the given samples, with dimensions: [samples, channels, height, width]."""
    number_samples = len(samples)
    
    # Get and sort all FEA filenames.
    filenames = glob.glob(os.path.join(folder, '*.txt'))
    filenames = sorted(filenames)

    # Only use the filenames that match the specified sample numbers. Assumes that filename numbers start from 1 and are contiguous.
    assert len(samples) <= len(filenames), f"The requested number of samples {len(samples)} exceeds the number of available files {len(filenames)}."
    filenames = [filenames[number-1] for number in samples[KEY_SAMPLE_NUMBER]]

    # Store all data in a single array, initialized with a default value. The order of values in the text files is determined by ANSYS.
    DEFAULT_VALUE = 0
    labels = np.full((number_samples, *OUTPUT_SIZE), DEFAULT_VALUE, dtype=float)
    for i, filename in enumerate(filenames):
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        # Assume each line contains the result followed by the corresopnding nodal coordinates, in the format: value, x, y, z. Round the coordinates to the specified number of digits to eliminate rounding errors from FEA.
        node_values = [
            [float(value) if i == 0 else round(float(value), 2) for i, value in enumerate(line.split(','))]
            for line in lines
        ]
        # Sort the values using the coordinates.
        node_values.sort(key=lambda _: (_[3], _[2], _[1]))

        image_channels = int(samples[KEY_NODES_WIDTH][i])
        image_height = int(samples[KEY_NODES_HEIGHT][i])
        image_length = int(samples[KEY_NODES_LENGTH][i])

        # Insert the values into the combined array, aligned top-left.
        if is_3d:
            labels[i, :image_channels, :image_height, :image_length] = np.reshape(
                [_[0] for _ in node_values],
                (image_channels, image_height, image_length),
            )
        else:
            labels[i, :, :image_height, :image_length] = np.reshape(
                [_[0] for _ in node_values if _[3] == 0.0],
                (1, image_height, image_length),
            )
        
        if (i+1) % 100 == 0:
            print(f"Reading label {i+1} / {number_samples}...", end='\r')
    print()


    return labels

def rgb_to_hue(array: np.ndarray) -> np.ndarray:
    """Convert a 3-channel RGB array into a 1-channel hue array with values in [0, 1]."""

    array = array / 255
    hue_array = np.zeros((array.shape[0], array.shape[1]))
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            hsv = colorsys.rgb_to_hsv(array[i, j, 0], array[i, j, 1], array[i, j, 2])
            hue_array[i, j] = hsv[0]
    return hue_array

def hsv_to_rgb(array: np.ndarray) -> np.ndarray:
    """Convert a 3-channel HSV array into a 3-channel RGB array."""

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            rgb = colorsys.hsv_to_rgb(array[i, j, 0], array[i, j, 1], array[i, j, 2])
            for k in range(3):
                array[i, j, k] = rgb[k] * 255
    return array

def array_to_colormap(array: np.ndarray, divide_by=None) -> np.ndarray:
    """Scale a 2D array of values to be inside [0, 1] and convert to a 3D color image."""
    # Make copy of array so that original array is not modified.
    array = np.copy(array)

    if divide_by:
        array /= divide_by
    else:
        array /= np.max(array)
    # Invert the values so that red represents high stresses.
    array = 1 - array
    # Constrain the values so that only colors from red to blue are shown, to match standard colors used in FEA.
    array = array * (240/360)
    # Convert the output to an RGB array.
    SATURATION, VALUE = 1, 2/3
    array = np.dstack((array, SATURATION * np.ones(array.shape, float), VALUE * np.ones(array.shape, float)))
    array = hsv_to_rgb(array)
    
    return array

def write_image(array: np.ndarray, filename: str) -> None:
    with Image.fromarray(array.astype(np.uint8)) as image:
        image.save(filename)


# Colors for plots.
class Colors:
    RED = "#ff4040"
    RED_DARK = "#9e2828"
    RED_LIGHT = "#ffb6b6"
    BLUE = "#0095ff"
    BLUE_DARK = "#005c9e"
    BLUE_LIGHT = "#9ed7ff"
    GRAY = "#808080"
    GRAY_DARK = "#404040"
    GRAY_LIGHT = "#bfbfbf"