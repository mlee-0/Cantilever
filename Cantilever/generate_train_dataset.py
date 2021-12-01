import colorsys
import glob
import os
import random

import numpy as np
from PIL import Image

from main import FOLDER_ROOT, FOLDER_TRAIN_INPUTS, FOLDER_TRAIN_OUTPUTS, FILENAME_SAMPLES_TRAIN, generate_samples, write_samples, read_samples, generate_input_images, write_ansys_script


NUMBER_SAMPLES = 1000

if __name__ == '__main__':
    # Try to read sample values from the text file if it already exists. If not, generate the samples.
    samples = read_samples(FILENAME_SAMPLES_TRAIN)
    if not samples:
        samples = generate_samples(NUMBER_SAMPLES, show_histogram=False)
    write_samples(samples, FILENAME_SAMPLES_TRAIN)
    write_ansys_script(samples, 'ansys_script_train.lgw')
    generate_input_images(samples, FOLDER_TRAIN_INPUTS)