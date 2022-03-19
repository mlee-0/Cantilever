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
load = Parameter(low=500, high=1000, step=10, precision=0, name='Load', units='N')
angle = Parameter(low=0, high=360, step=None, precision=0, name='Angle', units='Degrees')
# Names of quantities that are not generated but are still stored in the text files.
key_x_load = 'Load X'
key_y_load = 'Load Y'
key_image_length = 'Image Length'
key_image_height = 'Image Height'

# Size of input images (channel-height-width). Must have the same aspect ratio as the largest possible cantilever geometry.
INPUT_CHANNELS = 3
INPUT_SIZE = (INPUT_CHANNELS, 250, 500)
assert (INPUT_SIZE[1] / INPUT_SIZE[2]) == (height.high / length.high), 'Input image size must match aspect ratio of cantilever: {height.high}:{length.high}.'
# Size of output images (channel-height-width) produced by the network. Output images produced by FEA will be resized to this size.
OUTPUT_CHANNEL_NAMES = ('stress',)  #('stress', 'displacement')
OUTPUT_CHANNELS = len(OUTPUT_CHANNEL_NAMES)
OUTPUT_SIZE = (OUTPUT_CHANNELS, 25, 50)

# Folders and files.
FOLDER_ROOT = 'Cantilever' if not GOOGLE_COLAB else 'drive/My Drive/Colab Notebooks'
FOLDER_TRAIN_OUTPUTS = os.path.join(FOLDER_ROOT, 'Train Labels')
FOLDER_VALIDATION_OUTPUTS = os.path.join(FOLDER_ROOT, 'Validation Labels')
FOLDER_TEST_OUTPUTS = os.path.join(FOLDER_ROOT, 'Test Labels')
FILENAME_SAMPLES_TRAIN = 'samples_train.txt'
FILENAME_SAMPLES_VALIDATION = 'samples_validation.txt'
FILENAME_SAMPLES_TEST = 'samples_test.txt'

# Number of digits used for numerical file names.
NUMBER_DIGITS = 6


def generate_samples(number_samples) -> dict:
    """Generate sample values for each parameter and return them as a dictionary."""

    # Generate sample values for each parameter.
    samples = {}
    samples[load.name] = generate_logspace_values(number_samples, load, skew_amount=1.5, skew_high=True)
    samples[angle.name] = generate_angles(number_samples, angle, std=30)
    samples[length.name] = generate_logspace_values(number_samples, length, skew_amount=1.5, skew_high=True)
    samples[height.name] = generate_logspace_values(number_samples, height, skew_amount=1.5, skew_high=False)
    samples[elastic_modulus.name] = generate_uniform_values(number_samples, elastic_modulus)
    
    # Calculate the image size corresponding to the geometry.
    image_lengths = np.round(OUTPUT_SIZE[2] * (samples[length.name] / length.high))
    image_heights = np.round(OUTPUT_SIZE[1] * (samples[height.name] / height.high))
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
    
    return samples

def generate_uniform_values(number_samples: int, parameter: Parameter) -> np.ndarray:
    """Generate uniformly distributed, evenly spaced samples for the specified parameter."""
    values = np.arange(parameter.low, parameter.high+parameter.step, parameter.step)
    values = np.array(random.choices(values, k=number_samples))
    values = np.round(values, parameter.precision)
    return values

def generate_logspace_values(number_samples: int, parameter: Parameter, skew_amount, skew_high) -> np.ndarray:
    """Generate samples that are more concentrated at one end of the range."""
    values = np.logspace(0, skew_amount, number_samples)
    values = values - np.min(values)
    values = values / np.max(values)
    if skew_high:
        values = 1 - values
    values = values * (parameter.high - parameter.low)
    values = values + parameter.low
    values = np.round(values, parameter.precision)
    np.random.shuffle(values)
    
    plot_histogram(values, title=parameter.name)

    return values

def generate_angles(number_samples: int, parameter: Parameter, std: int) -> np.ndarray:
    """Generate angle samples using a distribution with two peaks centered at 90 and 270 degrees."""
    values = np.append(
        np.random.normal(90, std, number_samples//2),
        np.random.normal(270, std, number_samples//2),
    )
    assert values.size == number_samples
    values = np.round(values, parameter.precision)
    np.random.shuffle(values)
    # Convert values outside [0, 360] to the equivalent value within that range.
    values = np.mod(values, 360)
    assert not np.any((values > parameter.high) | (values < parameter.low)), f"Angle values were generated outside the specified range: {parameter.low} to {parameter.high}."

    plot_histogram(values, title=parameter.name)

    return values

def plot_histogram(values: np.ndarray, title=None) -> None:
    plt.figure()
    plt.hist(values, bins=100, rwidth=0.75, color='#0095ff')
    if title:
        plt.title(title)
    plt.show()

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

def generate_input_images(samples) -> np.ndarray:
    """Return a 4D array of images for each of the specified sample values, with dimensions: [samples, channels, height, width]."""

    number_samples = get_sample_size(samples)
    inputs = np.full((number_samples, *INPUT_SIZE), 0, dtype=int)
    for i in range(number_samples):
        pixel_length, pixel_height = int(samples[key_image_length][i]), int(samples[key_image_height][i])
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

def generate_label_images(samples, folder, clip_high_stresses=False) -> np.ndarray:
    """
    Return a 4D array of images for each of the FEA text files, with dimensions: [samples, channels, height, width].

    `clip_high_stresses`: Reduce stresses above a threshold to the threshold.
    """
    number_samples = get_sample_size(samples)
    fea_filenames = glob.glob(os.path.join(folder, '*.txt'))
    fea_filenames = sorted(fea_filenames)
    assert len(fea_filenames) == number_samples, f'Found {len(fea_filenames)} .txt files in {folder}, but should be {number_samples}.'

    # Store all stress data in a single array, initialized with a specific background value.
    BACKGROUND_VALUE = 0
    labels = np.full((number_samples, *OUTPUT_SIZE), BACKGROUND_VALUE, dtype=float)
    for i, fea_filename in enumerate(fea_filenames):
        stress, displacement = parse_label_file(fea_filename)
        data = {'stress': stress, 'displacement': displacement}
        for channel, channel_name in enumerate(OUTPUT_CHANNEL_NAMES):
            values = data[channel_name]
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
            labels[i, channel, :array.shape[0], :array.shape[1]] = array
    
    # Reduce stresses above a threshold to the threshold value to prevent a large portion of the dataset having values near zero.
    if clip_high_stresses:
        stresses = labels[0, ...][labels[0, ...] != BACKGROUND_VALUE]
        threshold_stress = np.mean(stresses) + 5 * np.std(stresses)
        print(f'Clipping stresses to reduce maximum from {np.max(stresses)} to {threshold_stress}.')
        labels[0, ...] = np.clip(labels[0, ...], None, threshold_stress)
    
    return labels

def parse_label_file(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays of data stored in the specified label .txt file."""
    with open(filename, 'r') as file:
        stress, displacement_x, displacement_y = list(zip(
            *[[float(value) for value in line.split(',')] for line in file.readlines()]
            ))
        stress = np.array(stress)
        displacement = np.sqrt(
            np.power(np.array(displacement_x), 2) + np.power(np.array(displacement_y), 2)
            )
    return stress, displacement

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

def array_to_colormap(array, divide_by=None) -> np.ndarray:
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

def write_image(array, filename) -> None:
    with Image.fromarray(array.astype(np.uint8)) as image:
        image.save(filename)

def show_stress_histogram(train_dataset=True) -> None:
    """Read all stress values generated by FEA and show a histogram. Used to show uniformity of magnitudes of values."""
    fea_filenames = glob.glob(os.path.join(FOLDER_TRAIN_OUTPUTS if train_dataset else FOLDER_TEST_OUTPUTS, '*.txt'))
    fea_filenames = sorted(fea_filenames)

    stresses = np.zeros(0)
    for i, fea_filename in enumerate(fea_filenames):
        stress, displacement = parse_label_file(fea_filename)
        stresses = np.append(stresses, stress)
    
    plt.figure()
    plt.hist(stresses, bins=100, rwidth=0.75, color='#0095ff')
    plt.title(f"Maximum value: {np.max(stresses)}")
    plt.show()

if __name__ == "__main__":
    show_stress_histogram(train_dataset=True)