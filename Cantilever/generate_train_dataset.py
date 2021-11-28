import colorsys
import glob
import os
import random

import numpy as np
from PIL import Image

from main import NUMBER_SAMPLES, FOLDER_ROOT, FOLDER_TRAIN_INPUTS, FOLDER_TRAIN_OUTPUTS, NUMBER_DIGITS, generate_samples, generate_input_images, calculate_load_components, write_ansys_script, write_fea_spreadsheet


# Names of files to be generated.
FILENAME_SAMPLES = 'samples_train.txt'

# Write the specified sample values to a text file.
def write_samples(samples, load_components, filename) -> None:
    # Write samples to text file.
    text = [
        f'{str(i+1).zfill(NUMBER_DIGITS)},  Load: {load_sample:>10},  X load: {load_x:>15},  Y load: {load_y:>15},  Angle: {angle_sample:>10},  Length: {length_sample:>5},  Height: {height_sample:>5}\n'
        for i, (load_sample, angle_sample, length_sample, height_sample, (load_x, load_y)) in enumerate(zip(*samples, load_components))
        ]
    with open(os.path.join(FOLDER_ROOT, filename), 'w') as file:
        file.writelines(text)
    print(f'Wrote samples in {filename}.')

# Return the sample values found in the text file previously generated.
def read_samples(filename) -> tuple:
    load_samples, angle_samples, length_samples, height_samples = [], [], [], []
    filename = os.path.join(FOLDER_ROOT, filename)
    try:
        with open(filename, 'r') as file:
            for line in file.readlines():
                load_sample, *_, angle_sample, length_sample, height_sample = [float(line.split(':')[1]) for line in line.split(',')[1:]]
                load_samples.append(load_sample)
                angle_samples.append(angle_sample)
                length_samples.append(length_sample)
                height_samples.append(height_sample)
    except FileNotFoundError:
        print(f'"{filename}" not found.')
        return
    else:
        print(f'Found samples in {filename}.')
        return load_samples, angle_samples, length_samples, height_samples

if __name__ == '__main__':
    # Try to read sample values from the text file if it already exists. If not, generate the samples.
    samples = read_samples(FILENAME_SAMPLES)
    if not samples:
        samples = generate_samples(NUMBER_SAMPLES, show_histogram=False)
    load_components = calculate_load_components(*samples[0:2])
    write_samples(samples, load_components, FILENAME_SAMPLES)
    write_ansys_script(samples, load_components, 'ansys_script_train.lgw')
    generate_input_images(samples, FOLDER_TRAIN_INPUTS)
    write_fea_spreadsheet(samples, FOLDER_TRAIN_OUTPUTS, 'stress.csv')