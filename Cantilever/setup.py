'''
Information about parameters and functions for reading and writing files.
'''

import colorsys
from dataclasses import dataclass
import glob
import os
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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
load = Parameter(low=500, high=1000, step=10, precision=0, name='Load', units='N')
angle = Parameter(low=0, high=355, step=5, precision=2, name='Angle', units='Degrees')
# Names of quantities that are not generated but are still stored in the text files.
key_x_load = 'Load X'
key_y_load = 'Load Y'
key_image_length = 'Image Length'
key_image_height = 'Image Height'

# Size of input images (channel-height-width). Must have the same aspect ratio as the largest possible cantilever geometry.
INPUT_CHANNELS = 4
INPUT_SIZE = (INPUT_CHANNELS, 50, 100)
assert (INPUT_SIZE[1] / INPUT_SIZE[2]) == (height.high / length.high), 'Input image size must match aspect ratio of cantilever: {height.high}:{length.high}.'
# Size of output images (channel-height-width) produced by the network. Output images produced by FEA will be resized to this size.
OUTPUT_CHANNELS = 1
OUTPUT_SIZE = (OUTPUT_CHANNELS, *INPUT_SIZE[1:3])

# Folders and files.
FOLDER_ROOT = 'Cantilever'
FOLDER_TRAIN_OUTPUTS = os.path.join(FOLDER_ROOT, 'Train Outputs')
FOLDER_TEST_OUTPUTS = os.path.join(FOLDER_ROOT, 'Test Outputs')
FILENAME_SAMPLES_TRAIN = 'samples_train.txt'
FILENAME_SAMPLES_TEST = 'samples_test.txt'

# Number of digits used for numerical file names.
NUMBER_DIGITS = 6


def generate_samples(number_samples, show_histogram=False) -> dict:
    """Generate sample values for each parameter and return them as a dictionary."""

    # Generate samples.
    samples = {}
    for parameter in [load, angle, length, height, elastic_modulus]:
        assert isinstance(parameter, Parameter)
        values = np.arange(parameter.low, parameter.high+parameter.step, parameter.step)
        samples[parameter.name] = np.round(
            random.choices(values, k=number_samples,),
            parameter.precision,
            )
    
    # Calculate the image size corresponding to the geometry.
    image_lengths = np.round(INPUT_SIZE[2] * (samples[length.name] / length.high))
    image_heights = np.round(INPUT_SIZE[1] * (samples[height.name] / height.high))
    samples[key_image_length] = image_lengths
    samples[key_image_height] = image_heights
    
    # Calculate the x- and y-components of the loads and corresponding angles.
    x_loads = np.round(
        np.cos(samples[angle.name] * (np.pi/180)) * samples[load.name] / (image_heights-1),
        load.precision
        )
    y_loads = np.round(
        np.sin(samples[angle.name] * (np.pi/180)) * samples[load.name] / (image_heights-1),
        load.precision
        )
    samples[key_x_load] = x_loads
    samples[key_y_load] = y_loads

    # Plot histograms for angle samples.
    if show_histogram:
        plt.figure()
        plt.hist(
            samples[angle.name],
            bins=round(angle.high-angle.low),
            rwidth=0.75,
            color='#0095ff',
            )
        plt.xticks(np.linspace(angle.low, angle.high, 5))
        plt.title(angle.name)
        plt.show()
    
    return samples

def write_samples(samples, filename) -> None:
    """Write the specified sample values to a text file."""

    number_samples = get_sample_size(samples)
    text = [None] * number_samples
    for i in range(number_samples):
        text[i] = ','.join(
            [f'{str(i+1).zfill(NUMBER_DIGITS)}'] + [f'{key}:{value[i]}' for key, value in samples.items()]
            ) + '\n'
    with open(os.path.join(FOLDER_ROOT, filename), 'w') as file:
        file.writelines(text)
    print(f'Wrote samples in {filename}.')

def read_samples(filename) -> dict:
    """Return the sample values found in the text file previously generated."""

    samples = {}
    filename = os.path.join(FOLDER_ROOT, filename)
    try:
        with open(filename, 'r') as file:
            for line in file.readlines():
                for data in line.split(',')[1:]:
                    key, value = data.split(':')
                    key, value = key.strip(), float(value)
                    if key in samples:
                        samples[key].append(value)
                    else:
                        samples[key] = [value]
    except FileNotFoundError:
        print(f'"{filename}" not found.')
        return
    else:
        print(f'Found samples in {filename}.')
        return samples

def generate_input_images(samples) -> List[np.ndarray]:
    """Return a list of images for each of the specified sample values."""

    number_samples = get_sample_size(samples)
    inputs = [None] * number_samples
    for i in range(number_samples):
        pixel_length, pixel_height = int(samples[key_image_length][i]), int(samples[key_image_height][i])
        image = np.zeros(INPUT_SIZE)
        # Create a channel with a gray line of pixels representing the load magnitude and direction.
        r = np.arange(max(image.shape[1:]))
        x = r * np.cos(samples[angle.name][i] * np.pi/180) + image.shape[2]/2
        y = r * np.sin(samples[angle.name][i] * np.pi/180) + image.shape[1]/2
        x = x.astype(int)
        y = y.astype(int)
        inside_image = (x >= 0) * (x < image.shape[2]) * (y >= 0) * (y < image.shape[1])
        image[0, y[inside_image], x[inside_image]] = 255 * (samples[load.name][i] / load.high)
        image[0, :, :] = np.flipud(image[0, :, :])
        # Create a channel with a white rectangle representing the dimensions of the cantilever.
        image[1, :pixel_height, :pixel_length] = 255
        # Create a channel with the elastic modulus distribution.
        image[2, :pixel_height, :pixel_length] = 255 * (samples[elastic_modulus.name][i] / elastic_modulus.high)
        # Create a channel with the fixed boundary conditions.
        image[3, :pixel_height, 0] = 255
        # Append the image to the list.
        inputs[i] = image
    return inputs

def get_sample_size(samples) -> int:
    """Get the number of samples found in the specified sample values."""

    sample_sizes = [len(_) for _ in samples.values()]
    low, high = min(sample_sizes), max(sample_sizes)
    assert low == high, 'Found different numbers of samples in the provided samples:  min. {low}, max. {high}.'
    return low
    
def write_ansys_script(samples, filename) -> None:
    """Write a text file containing ANSYS commands used to automate FEA and generate stress contour images."""

    number_samples = get_sample_size(samples)
    # Read the template script.
    with open(os.path.join(FOLDER_ROOT, 'ansys_template.lgw'), 'r') as file:
        lines = file.readlines()
    
    # Replace placeholder lines in the template script.
    with open(os.path.join(FOLDER_ROOT, filename), 'w') as file:
        # Initialize dictionary of placeholder strings (keys) and strings they should be replaced with (values).
        placeholder_substitutions = {}
        # Define names of variables.
        loop_variable = 'i'
        samples_variable = 'samples'
        # Add commands that define the array containing generated samples.
        commands_define_samples = [f'*DIM,{samples_variable},ARRAY,{9},{number_samples}\n']
        for i in range(number_samples):
            commands_define_samples.append(
                f'{samples_variable}(1,{i+1}) = {samples[load.name][i]},{samples[key_x_load][i]},{samples[key_y_load][i]},{samples[angle.name][i]},{samples[length.name][i]},{samples[height.name][i]},{samples[elastic_modulus.name][i]},{samples[key_image_length][i]},{samples[key_image_height][i]}\n'
                )
        placeholder_substitutions['! placeholder_define_samples\n'] = commands_define_samples
        # Add loop commands.
        placeholder_substitutions['! placeholder_loop_start\n'] = f'*DO,{loop_variable},1,{number_samples},1\n'
        placeholder_substitutions['! placeholder_loop_end\n'] = f'*ENDDO\n'
        # Add commands that format and create the output files.
        placeholder_substitutions['! placeholder_define_suffix\n'] = f'suffix = \'{"0"*NUMBER_DIGITS}\'\n'
        placeholder_substitutions['! placeholder_define_number\n'] = f'number = CHRVAL({loop_variable})\n'
        placeholder_substitutions['! placeholder_define_filename\n'] = f'filename = \'fea_%STRFILL(suffix,number,{NUMBER_DIGITS}-STRLENG(number)+1)%\'\n'
        # Substitute all commands into placeholders.
        for placeholder in placeholder_substitutions:
            command = placeholder_substitutions[placeholder]
            indices = [i for i, line in enumerate(lines) if line == placeholder]
            if isinstance(command, list):
                for index in indices:
                    for sub_command in command[::-1]:
                        lines.insert(index+1, sub_command)
                    del lines[index]
            else:
                for index in indices:
                    lines[index] = command
        # Write the file.
        file.writelines(lines)
        print(f'Wrote {filename}.')

def generate_label_images(samples, folder, normalization_values:tuple=(None,None), clip_high_stresses=False) -> Tuple[List[np.ndarray], List[float]]:
    """
    Return a list of images for each of the FEA text files and a list of maximum values found in each channel.

    `normalization_values`: Divide all quantities in each channel by these values. If not provided, use the maximum values found in the corresponding channels.
    `clip_high_stresses`: Reduce stresses above a threshold to the threshold.
    """
    number_samples = get_sample_size(samples)
    fea_filenames = glob.glob(os.path.join(folder, '*.txt'))
    fea_filenames = sorted(fea_filenames)
    assert len(fea_filenames) == number_samples, f'Found {len(fea_filenames)} .txt files in {folder}, but should be {number_samples}.'

    # Store all stress data in a single array, initialized with a specific background value.
    BACKGROUND_VALUE_INITIAL = -1
    BACKGROUND_VALUE = 0
    labels = np.full(
        (*OUTPUT_SIZE, len(fea_filenames)),
        BACKGROUND_VALUE_INITIAL,
        dtype=float,
        )
    for i, fea_filename in enumerate(fea_filenames):
        # Read the nodal stress and displacement values.
        with open(fea_filename, 'r') as file:
            raw_stress, displacement_x, displacement_y = list(zip(
                *[[float(value) for value in line.split(',')] for line in file.readlines()]
                ))
            displacement = np.sqrt(
                np.power(np.array(displacement_x), 2) + np.power(np.array(displacement_y), 2)
                )
        for channel, values in enumerate([raw_stress]): #enumerate([raw_stress, displacement]):
            # Initialize a 2D array.
            array = np.zeros((int(samples[key_image_height][i]), int(samples[key_image_length][i])))
            # Determine the number of mesh divisions used in this sample.
            mesh_divisions = (int(samples[key_image_length][i]-1), int(samples[key_image_height][i]-1))
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
            labels[channel, :array.shape[0], :array.shape[1], i] = array
    
    # Reduce stresses above a threshold to the threshold value to prevent a large portion of the dataset having values near zero.
    if clip_high_stresses:
        stresses = labels[0, ...][labels[0, ...] != BACKGROUND_VALUE_INITIAL]
        threshold_stress = np.mean(stresses) + 5 * np.std(stresses)
        print(f'Clipping stresses to reduce maximum from {np.max(stresses)} to {threshold_stress}.')
        labels[0, ...] = np.clip(labels[0, ...], None, threshold_stress)

    # Normalize values (<= 1) by dividing by the maximum value found among all samples.
    maxima = []
    for channel in range(OUTPUT_CHANNELS):
        maximum = np.max(labels[channel, ...])
        maxima.append(maximum)
        normalization_value = maximum if normalization_values[channel] is None else normalization_values[channel]
        assert normalization_value >= maximum, f'The value by which values in channel {channel} are divided {normalization_value} is less than the maximum value found {maximum}, which will cause normalized values to be > 1.'
        labels[channel, ...][labels[channel, ...] != BACKGROUND_VALUE_INITIAL] /= normalization_value
        labels[labels == BACKGROUND_VALUE_INITIAL] = BACKGROUND_VALUE

    return [labels[..., i] for i in range(labels.shape[-1])], tuple(maxima)

def rgb_to_hue(array) -> np.ndarray:
    """Convert a 3-channel RGB array into a 1-channel hue array with values in [0, 1]."""

    array = array / 255
    hue_array = np.zeros((array.shape[0], array.shape[1]))
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            hsv = colorsys.rgb_to_hsv(array[i, j, 0], array[i, j, 1], array[i, j, 2])
            hue_array[i, j] = hsv[0]
    return hue_array

def hsv_to_rgb(array) -> np.ndarray:
    """Convert a 3-channel HSV array into a 3-channel RGB array."""

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            rgb = colorsys.hsv_to_rgb(array[i, j, 0], array[i, j, 1], array[i, j, 2])
            for k in range(3):
                array[i, j, k] = rgb[k] * 255
    return array

def array_to_colormap(array) -> np.ndarray:
    """Convert a 2D array of values in [0, 1] to a color image."""

    # Invert the values so that red represents high stresses.
    array = 1 - array
    # Constrain the values so that only colors from red to blue are shown, to match standard colors used in FEA.
    array = array * (240/360)
    # Convert the output to an RGB array.
    SATURATION, VALUE = 1, 2/3
    array = np.dstack((array, SATURATION * np.ones(array.shape, float), VALUE * np.ones(array.shape, float)))
    array = hsv_to_rgb(array)
    return array