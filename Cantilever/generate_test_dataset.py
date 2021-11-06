import colorsys
import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from main import generate_input_images, FOLDER_TEST_INPUTS


# Remove existing images in the folder.
filenames = glob.glob(os.path.join(FOLDER_TEST_INPUTS, '*.png'))
for filename in filenames:
    os.remove(filename)
print(f'Deleted {len(filenames)} existing images in {FOLDER_TEST_INPUTS}.')

# Generate new data not part of the training dataset.
test_angles = [1, 7, 13, 43, 82]
test_filenames = generate_input_images(test_angles, FOLDER_TEST_INPUTS)