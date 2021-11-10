import colorsys
from dataclasses import dataclass
import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from main import generate_input_images, FOLDER_ROOT, FOLDER_TRAIN_INPUTS, FOLDER_TRAIN_OUTPUTS, OUTPUT_SIZE


# A dataclass that stores settings for each parameter.
@dataclass
class Parameter:
    # The minimum and maximum values between which samples are generated.
    range: tuple
    # Number of decimal places to which sample values are rounded.
    precision: int
    # Name of the parameter.
    name: str
    # Units for the parameter.
    units: str = ''
    # Generated sample values.
    samples: list = None

# Dataset size.
NUMBER_SAMPLES = 100
assert NUMBER_SAMPLES % 4 == 0, 'Sample size must be divisible by 4 for angles to be generated properly.'
# Define the settings for each parameter.
load = Parameter(range=(10000, 100000), precision=0, name='Load', units='N')
angle = Parameter(range=(0, 360), precision=2, name='Angle', units='Degrees')
length = Parameter(range=(0.1, 4), precision=3, name='Length', units='m')
height = Parameter(range=(0.1, 1), precision=3, name='Height', units='m')
# Store all parameters.
parameters = (load, angle, length, height)
# The two angle values at which there should be peaks in the probability distribution used to generate angle values.
ANGLE_PEAKS = (0, 180)

# Names of text files to be generated.
FILENAME_SAMPLES = 'cantilever_samples.txt'
FILENAME_ANSYS = 'ansys.txt'

# Generate and return sample values for each parameter.
def generate_samples():
    # Helper function for generating unevenly spaced samples within a defined range. Setting "increasing" to True makes spacings increase as values increase.
    generate_logspace_samples = lambda low, high, increasing: (((
        np.logspace(
            0, 1, round(NUMBER_SAMPLES/4)+1
            ) - 1) / 9) * (1 if increasing else -1) + (0 if increasing else 1)
        ) * (high - low) + low
    # Generate samples.
    load_samples = np.linspace(load.range[0], load.range[1], NUMBER_SAMPLES)
    angle_samples = np.concatenate((
        generate_logspace_samples(angle.range[0], (ANGLE_PEAKS[1]-ANGLE_PEAKS[0])/2, increasing=True)[:-1],
        generate_logspace_samples((ANGLE_PEAKS[1]-ANGLE_PEAKS[0])/2, ANGLE_PEAKS[1], increasing=False)[1:],
        generate_logspace_samples(ANGLE_PEAKS[1], (ANGLE_PEAKS[1]-ANGLE_PEAKS[0])/2 + ANGLE_PEAKS[1], increasing=True)[:-1],
        generate_logspace_samples((ANGLE_PEAKS[1]-ANGLE_PEAKS[0])/2 + ANGLE_PEAKS[1], angle.range[1], increasing=False)[1:],
        ))
    length_samples = np.linspace(length.range[0], length.range[1], NUMBER_SAMPLES)
    height_samples = np.linspace(height.range[0], height.range[1], NUMBER_SAMPLES)
    # Randomize ordering of samples.
    np.random.shuffle(load_samples)
    np.random.shuffle(angle_samples)
    np.random.shuffle(length_samples)
    np.random.shuffle(height_samples)
    # Round all samples to a fixed number of decimal places.
    load_samples = np.round(load_samples, load.precision)
    angle_samples = np.round(angle_samples, angle.precision)
    length_samples = np.round(length_samples, length.precision)
    height_samples = np.round(height_samples, height.precision)
    # # Store samples.
    # load.samples = load_samples
    # angle.samples = angle_samples
    # length.samples = length_samples
    # height.samples = height_samples
    
    # Plot histograms for generated samples.
    plt.figure()
    plt.hist(
        angle_samples,
        bins=round(angle.range[1]-angle.range[0]),
        rwidth=0.75,
        color='#0095ff',
        )
    plt.xticks(np.linspace(angle.range[0], angle.range[1], 5))
    plt.title(angle.name)
    # for i, samples in enumerate([load_samples, angle_samples, length_samples, height_samples]):
    #     is_angle = i == 1
    #     plt.subplot(1, len(parameters), i+1)
    #     plt.hist(
    #         samples,
    #         bins=round(parameters[i].range[1]-parameters[i].range[0]) if is_angle else 50,
    #         rwidth=0.75,
    #         color='#0095ff' if not is_angle else '#ff4040',
    #         )
    #     plt.xticks(np.linspace(parameters[i].range[0], parameters[i].range[1], 5 if is_angle else 2))
    #     plt.title(parameters[i].name)
    plt.show()
    return load_samples, angle_samples, length_samples, height_samples

# Write the specified sample values to a text file.
def write_samples(samples):
    load_samples, angle_samples, length_samples, height_samples = samples
    # Determine the x and y components of the load for easier entry in FEA.
    load_components = []
    for load, angle in zip(load_samples, angle_samples):
        angle *= (np.pi / 180)
        load_components.append((
            np.cos(angle) * load,
            np.sin(angle) * load,
            ))
    # Write samples to text file.
    text = [
        f'Load: {load:>10},  X load: {load_x:>10.2f},  Y load: {load_y:>10.2f},  Angle: {angle:>10},  Length: {length:>5},  Height: {height:>5}\n'
        for load, (load_x, load_y), angle, length, height in zip(load_samples, load_components, angle_samples, length_samples, height_samples)
        ]
    with open(os.path.join(FOLDER_ROOT, FILENAME_SAMPLES), 'w') as file:
        file.writelines(text)
    print(f'Wrote samples in {FILENAME_SAMPLES}.')

# Write a text file containing ANSYS commands used to perform FEA and generate stress contour images.
def write_ansys_script(samples):
    pass

# Return the sample values found in the text file previously generated.
def read_samples():
    angle_samples = []
    filename = os.path.join(FOLDER_ROOT, FILENAME_SAMPLES)
    try:
        with open(filename, 'r') as file:
            for string in file.readlines():
                *_, angle = [int(float(string.split(':')[1])) for string in string.split(',')]
                angle_samples.append(angle)
        print(f'Found samples in {filename}.')
    except FileNotFoundError:
        print(f'"{filename}" not found.')
    return angle_samples

# Crop and resize the stress contour images.
def crop_output_images():
    # LEFT, TOP = 209, 108
    # SIZE = (616, 155)
    filenames = glob.glob(os.path.join(FOLDER_TRAIN_OUTPUTS, '*.png'))
    for filename in filenames:
        with Image.open(filename) as image:
            # if image.size[0] > SIZE[0] and image.size[1] > SIZE[1]:
            #     image = image.crop((LEFT, TOP, LEFT+SIZE[0]-1, TOP+SIZE[1]-1))
            image_copy = image.convert('L')
            area = ImageOps.invert(image_copy).getbbox()
            image = image.crop(area)
            image = image.resize(OUTPUT_SIZE)
            image.save(filename)

# Try to read sample values from the text file if it already exists. If not, generate the samples.
# samples = read_samples()
if True: #not samples:
    samples = generate_samples()
    write_samples(samples)
# generate_input_images(samples, FOLDER_TRAIN_INPUTS)

# Crop and resize stress contour images generated by FEA. This only needs to run the first time the images are added to the folder.
# crop_output_images()