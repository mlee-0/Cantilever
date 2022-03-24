'''
Information about parameters and functions for reading and writing files.
'''

import colorsys
from dataclasses import dataclass
import glob
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    from google.colab import drive
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
    units: str = ''
    # Number of decimal places to which sample values are rounded.
    precision: int = 0

# Settings for each parameter.
length = Parameter(low=2, high=4, step=0.1, precision=1, name='Length', units='m')
height = Parameter(low=1, high=2, step=0.1, precision=1, name='Height', units='m')
elastic_modulus = Parameter(low=190, high=210, step=1, precision=0, name='Elastic Modulus', units='GPa')
load = Parameter(low=500, high=1000, step=5, precision=0, name='Load', units='N')
angle = Parameter(low=0, high=360, step=1, precision=0, name='Angle', units='Degrees')
# Names of quantities that are saved but are not randomly generated.
KEY_SAMPLE_NUMBER = 'Sample Number'
KEY_X_LOAD = 'Load X'
KEY_Y_LOAD = 'Load Y'
KEY_IMAGE_LENGTH = 'Image Length'
KEY_IMAGE_HEIGHT = 'Image Height'

# Size of input images (channel-height-width). Must have the same aspect ratio as the largest possible cantilever geometry.
INPUT_CHANNELS = 3
INPUT_SIZE = (INPUT_CHANNELS, 200, 400)
assert (INPUT_SIZE[1] / INPUT_SIZE[2]) == (height.high / length.high), 'Input image size must match aspect ratio of cantilever: {height.high}:{length.high}.'
# Size of output images (channel-height-width) produced by the network. Output images produced by FEA will be resized to this size.
OUTPUT_CHANNEL_NAMES = ('stress',)  #('stress', 'displacement')
OUTPUT_CHANNELS = len(OUTPUT_CHANNEL_NAMES)
OUTPUT_SIZE = (OUTPUT_CHANNELS, 20, 40)

# Folders and files.
FOLDER_ROOT = 'Cantilever' if not GOOGLE_COLAB else 'drive/My Drive/Colab Notebooks'
FOLDER_TRAIN_LABELS = os.path.join(FOLDER_ROOT, 'Train Labels')
FOLDER_TEST_LABELS = os.path.join(FOLDER_ROOT, 'Test Labels')
FILENAME_SAMPLES_TRAIN = 'samples_train.csv'
FILENAME_SAMPLES_TEST = 'samples_test.csv'
FILENAME_SUBSET = 'subset.txt'

# Number of digits used for numerical file names.
NUMBER_DIGITS = 6

# Colors for plots.
class Colors:
    RED = '#ff4040'
    BLUE = '#0095ff'
    GRAY = '#bfbfbf'


def plot_histogram(values: np.ndarray, title=None) -> None:
    plt.figure()
    plt.hist(values, bins=100, rwidth=0.75, color='#0095ff')
    if title:
        plt.title(title)
    plt.show()

def get_sample_size(samples: dict) -> int:
    """Get the number of samples found in the specified dictionary."""

    sample_sizes = [len(_) for _ in samples.values()]
    low, high = min(sample_sizes), max(sample_sizes)
    assert low == high, 'Found different numbers of samples in the provided samples:  min. {low}, max. {high}.'
    return low

def split_training_validation(number_samples: int, training_split: float) -> Tuple[int, int]:
    """Return the number of samples in the training and validation datasets for the given ratio."""
    assert 0 < training_split < 1, f"Invalid training split: {training_split}."
    training_size = round(training_split * number_samples)
    validation_size = round((1 - training_split) * number_samples)
    assert training_size + validation_size == number_samples
    return training_size, validation_size

def generate_input_images(samples: dict) -> np.ndarray:
    """Return a 4D array of images for each of the specified sample values, with dimensions: [samples, channels, height, width]."""

    number_samples = get_sample_size(samples)
    inputs = np.full((number_samples, *INPUT_SIZE), 0, dtype=int)
    for i in range(number_samples):
        pixel_length, pixel_height = int(samples[KEY_IMAGE_LENGTH][i]), int(samples[KEY_IMAGE_HEIGHT][i])
        image = np.zeros(INPUT_SIZE)

        # Create a channel with a white rectangle representing the dimensions of the cantilever.
        image[0, :pixel_height, :pixel_length] = 255

        # Create a channel with a gray line of pixels representing the load magnitude and direction.
        r = np.arange(max(image.shape[1:]))
        x = r * np.cos(samples[angle.name][i] * np.pi/180) + image.shape[2]/2
        y = r * np.sin(samples[angle.name][i] * np.pi/180) + image.shape[1]/2
        x = x.astype(int)
        y = y.astype(int)
        inside_image = (x >= 0) * (x < image.shape[2]) * (y >= 0) * (y < image.shape[1])
        image[1, y[inside_image], x[inside_image]] = 255 * (samples[load.name][i] / load.high)
        image[1, :, :] = np.flipud(image[1, :, :])

        # # Create a channel with a vertical white line whose position represents the load magnitude. Leftmost column is 0, rightmost column is the maximum magnitude.
        # # image[0, :pixel_height, :pixel_length] = 255 * samples[load.name][i] / load.high
        # image[0, :, round(image.shape[2] * samples[load.name][i] / load.high) - 1] = 255
        
        # Create a channel with the elastic modulus distribution.
        image[2, :pixel_height, :pixel_length] = 255 * (samples[elastic_modulus.name][i] / elastic_modulus.high)
        
        # # Create a channel with the fixed boundary conditions.
        # image[3, :pixel_height, 0] = 255
        # Append the image to the list.
        inputs[i, ...] = image
    return inputs

def generate_label_images(samples: dict, folder: str) -> np.ndarray:
    """Return a 4D array of images for the FEA text files found in the specified folder that correspond to the given samples, with dimensions: [samples, channels, height, width]."""
    number_samples = get_sample_size(samples)
    stresses, displacements = read_labels(folder, samples[KEY_SAMPLE_NUMBER])

    # Store all stress data in a single array, initialized with a specific background value. The order of values in the text files is determined by ANSYS.
    BACKGROUND_VALUE = 0
    data = {'stress': stresses, 'displacement': displacements}
    labels = np.full((number_samples, *OUTPUT_SIZE), BACKGROUND_VALUE, dtype=float)
    for i in range(number_samples):
        for channel, channel_name in enumerate(OUTPUT_CHANNEL_NAMES):
            values = data[channel_name][i]
            # Initialize a 2D array.
            array = np.zeros((int(samples[KEY_IMAGE_HEIGHT][i]), int(samples[KEY_IMAGE_LENGTH][i])))
            # Determine the number of mesh divisions used in this sample.
            mesh_divisions = (int(samples[KEY_IMAGE_LENGTH][i]-1), int(samples[KEY_IMAGE_HEIGHT][i]-1))
            # Values for interior nodes.
            array[1:-1, 1:-1] = np.flipud(
                np.reshape(values[2*sum(mesh_divisions):], [_-1 for _ in mesh_divisions[::-1]], 'F')
                )
            # Values for corner nodes.
            array[-1, 0] = values[0]
            array[-1, -1] = values[1]
            array[0, -1] = values[1+mesh_divisions[0]]
            array[0, 0] = values[1+mesh_divisions[0]+mesh_divisions[1]]
            # Values for edge nodes.
            array[-1, 1:-1] = values[2:2+mesh_divisions[0]-1]
            array[1:-1, -1] = values[2+mesh_divisions[0]:2+mesh_divisions[0]+mesh_divisions[1]-1][::-1]
            array[0, 1:-1] = values[2+mesh_divisions[0]+mesh_divisions[1]:2+2*mesh_divisions[0]+mesh_divisions[1]-1][::-1]
            array[1:-1, 0] = values[2+2*mesh_divisions[0]+mesh_divisions[1]-1:2+2*mesh_divisions[0]+2*mesh_divisions[1]-2]
            # Insert the array.
            labels[i, channel, :array.shape[0], :array.shape[1]] = array

    return labels

def read_labels(folder: str, sample_numbers: list = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Return arrays of data from text files in the specified folder."""
    fea_filenames = glob.glob(os.path.join(folder, '*.txt'))
    fea_filenames = sorted(fea_filenames)
    # Only use the filenames that match the specified sample numbers. Assumes that filename numbers start from 1 and are contiguous.
    if sample_numbers:
        fea_filenames = [filename for number, filename in enumerate(fea_filenames, 1) if number in sample_numbers]
    sample_size = len(fea_filenames)

    stresses = [None] * sample_size
    displacements_x = [None] * sample_size
    displacements_y = [None] * sample_size
    for i, fea_filename in enumerate(fea_filenames):
        with open(fea_filename, 'r') as file:
            lines = file.readlines()
        stress, displacement_x, displacement_y = list(zip(
            *[[float(value) for value in line.split(',')] for line in lines]
            ))
        stresses[i] = stress
        displacements_x[i] = displacement_x
        displacements_y[i] = displacement_y
        if (i+1) % 1000 == 0:
            print(f"{i+1}/{sample_size}", end='\r')
    print()
    stresses = [np.array(stress) for stress in stresses]
    displacements = [np.sqrt(np.power(np.array(x), 2) + np.power(np.array(y), 2)) for x, y in zip(displacements_x, displacements_y)]

    return stresses, displacements

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