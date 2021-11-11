import colorsys
import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

from main import load, angle, length, height, ANGLE_PEAKS, generate_input_images, FOLDER_ROOT, FOLDER_TRAIN_INPUTS, FOLDER_TRAIN_OUTPUTS, OUTPUT_SIZE


# Dataset size.
NUMBER_SAMPLES = 100
assert NUMBER_SAMPLES % 4 == 0, 'Sample size must be divisible by 4 for angles to be generated properly.'

# Names of text files to be generated.
FILENAME_SAMPLES = 'cantilever_samples.txt'
FILENAME_ANSYS = 'ansys.txt'

# Generate and store sample values for each parameter.
def generate_samples(show_histogram=False):
    # Helper function for generating unevenly spaced samples within a defined range. Setting "increasing" to True makes spacings increase as values increase.
    generate_logspace_samples = lambda low, high, increasing: (((
        np.logspace(
            0, 1, round(NUMBER_SAMPLES/4)+1
            ) - 1) / 9) * (1 if increasing else -1) + (0 if increasing else 1)
        ) * (high - low) + low
    # Generate samples.
    load_samples = np.linspace(load.low, load.high, NUMBER_SAMPLES)
    angle_samples = np.concatenate((
        generate_logspace_samples(angle.low, (ANGLE_PEAKS[1]-ANGLE_PEAKS[0])/2, increasing=True)[:-1],
        generate_logspace_samples((ANGLE_PEAKS[1]-ANGLE_PEAKS[0])/2, ANGLE_PEAKS[1], increasing=False)[1:],
        generate_logspace_samples(ANGLE_PEAKS[1], (ANGLE_PEAKS[1]-ANGLE_PEAKS[0])/2 + ANGLE_PEAKS[1], increasing=True)[:-1],
        generate_logspace_samples((ANGLE_PEAKS[1]-ANGLE_PEAKS[0])/2 + ANGLE_PEAKS[1], angle.high, increasing=False)[1:],
        ))
    length_samples = np.linspace(length.low, length.high, NUMBER_SAMPLES)
    height_samples = np.linspace(height.low, height.high, NUMBER_SAMPLES)
    # Randomize ordering of samples.
    np.random.shuffle(load_samples)
    np.random.shuffle(angle_samples)
    np.random.shuffle(length_samples)
    np.random.shuffle(height_samples)
    # Round samples to a fixed number of decimal places.
    load_samples = np.round(load_samples, load.precision)
    angle_samples = np.round(angle_samples, angle.precision)
    length_samples = np.round(length_samples, length.precision)
    height_samples = np.round(height_samples, height.precision)
    
    # Plot histograms for angle samples.
    if show_histogram:
        plt.figure()
        plt.hist(
            angle_samples,
            bins=round(angle.high-angle.low),
            rwidth=0.75,
            color='#0095ff',
            )
        plt.xticks(np.linspace(angle.low, angle.high, 5))
        plt.title(angle.name)
        plt.show()
    return load_samples, angle_samples, length_samples, height_samples

# Write the specified sample values to a text file.
def write_samples(samples):
    # Determine the x and y components of the load per element node for easier entry in FEA.
    load_components = []
    NUMBER_ELEMENTS = 10  # Defined in ANSYS
    for load_sample, angle_sample in zip(*samples[0:2]):
        angle_sample *= (np.pi / 180)
        load_components.append((
            np.cos(angle_sample) * load_sample / NUMBER_ELEMENTS,
            np.sin(angle_sample) * load_sample / NUMBER_ELEMENTS,
            ))
    # Write samples to text file.
    text = [
        # Name: {'_'.join([load_sample, angle_sample, length_sample, height_sample])}, 
        f'Load: {load_sample:>10},  X load: {load_x:>10.2f},  Y load: {load_y:>10.2f},  X load (edge): {load_x/2:>10.2f},  Y load (edge): {load_y/2:>10.2f},  Angle: {angle_sample:>10},  Length: {length_sample:>5},  Height: {height_sample:>5}\n'
        for load_sample, angle_sample, length_sample, height_sample, (load_x, load_y) in zip(*samples, load_components)
        ]
    with open(os.path.join(FOLDER_ROOT, FILENAME_SAMPLES), 'w') as file:
        file.writelines(text)
    print(f'Wrote samples in {FILENAME_SAMPLES}.')

# Write a text file containing ANSYS commands used to perform FEA and generate stress contour images.
def write_ansys_script(samples):
    pass

# Return the sample values found in the text file previously generated.
def read_samples():
    load_samples, angle_samples, length_samples, height_samples = [], [], [], []
    filename = os.path.join(FOLDER_ROOT, FILENAME_SAMPLES)
    try:
        with open(filename, 'r') as file:
            for string in file.readlines():
                load, *_, angle, length, height = [float(string.split(':')[1]) for string in string.split(',')]
                load_samples.append(load)
                angle_samples.append(angle)
                length_samples.append(length)
                height_samples.append(height)
    except FileNotFoundError:
        print(f'"{filename}" not found.')
        return None
    else:
        print(f'Found samples in {filename}.')
        return load_samples, angle_samples, length_samples, height_samples

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
samples = None #read_samples()
if not samples:
    samples = generate_samples(show_histogram=False)
    write_samples(samples)
generate_input_images(samples, FOLDER_TRAIN_INPUTS)

# Crop and resize stress contour images generated by FEA. This only needs to run the first time the images are added to the folder.
# crop_output_images()