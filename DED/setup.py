'''
Information about parameters and functions for reading and writing files.
'''

import glob
import os
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from PIL import Image, ImageFilter

try:
    from google.colab import drive  # type: ignore
except ModuleNotFoundError:
    GOOGLE_COLAB = False
else:
    GOOGLE_COLAB = True
    drive.mount("/content/drive")


# Size of output images (channel-height-width) produced by the network.
OUTPUT_CHANNELS = 1
OUTPUT_SIZE = (OUTPUT_CHANNELS, 64, 64)  # Make dimensions divisible by 4 for convenience in the se_2 layer
# Size of input images (channel-height-width).
INPUT_CHANNELS = 3
INPUT_SIZE = (INPUT_CHANNELS, *OUTPUT_SIZE[1:])

# Information about parameters.
LASER_POWER_RANGE = (250, 550)
POWDER_VALUE_RANGE = (2, 8)
FEED_RATE_RANGE = (700, 900)

# Folders and files.
experiment_number = 2

FOLDER_ROOT = 'DED' if not GOOGLE_COLAB else 'drive/My Drive/Colab Notebooks'
FOLDER_RESULTS = os.path.join(FOLDER_ROOT, "Results")
FILENAME_DATASET = 'Dataset_experiments_031722.xlsx'
if experiment_number == 1:
    FOLDER_LABELS = os.path.join(FOLDER_ROOT, 'Exp#1_(sheet#1)')
    SHEET_INDEX = 0
    TOTAL_SAMPLES = 81
    VALIDATE_SAMPLE_INDICES = range(0, TOTAL_SAMPLES, 3*3)
    TEST_SAMPLE_INDICES = range(1, TOTAL_SAMPLES, 3*3)

    # Number of unique sets of input parameters (product of numbers of unique values for each individual parameter found in the dataset).
    EMBEDDING_SIZE = 4  #4 * 3 * 4
elif experiment_number == 2:
    FOLDER_LABELS = os.path.join(FOLDER_ROOT, 'Exp#2_(sheet#2)')
    SHEET_INDEX = 1
    TOTAL_SAMPLES = 192
    VALIDATE_SAMPLE_INDICES = range(0, TOTAL_SAMPLES, 4)
    TEST_SAMPLE_INDICES = range(1, TOTAL_SAMPLES, 4)

    EMBEDDING_SIZE = 4 * 4 * 3
elif experiment_number == "faces":
    FOLDER_LABELS = os.path.join(FOLDER_ROOT, "Faces")
    TOTAL_SAMPLES = 47_011
    VALIDATE_SAMPLE_INDICES = range(0, TOTAL_SAMPLES, 10)
    TEST_SAMPLE_INDICES = range(1, TOTAL_SAMPLES, 10)
    
    EMBEDDING_SIZE = 2
else:
    print(f"Invalid experiment number: {experiment_number}")

# Colors for plots.
class Colors:
    BLUE = "#0095ff"
    BLUE_DARK = "#005c9e"
    BLUE_LIGHT = "#9ed7ff"
    
    RED = "#ff4040"
    RED_DARK = "#9e2828"
    RED_LIGHT = "#ffb6b6"

    ORANGE = "#FF8000"
    ORANGE_DARK = "#9E4F00"
    ORANGE_LIGHT = "#FFCE9E"
    
    GRAY = "#808080"
    GRAY_DARK = "#404040"
    GRAY_LIGHT = "#bfbfbf"


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

def _read_inputs(filename: str, sheetindex: int, sample_indices: List[int] = None) -> List[np.ndarray]:
    """Generate an image for each set of input parameters and return a list of images."""
    print("Reading inputs...")
    data = pd.read_excel(os.path.join(FOLDER_ROOT, filename), sheet_name=sheetindex, usecols=(1, 2, 3))
    value_ranges = (LASER_POWER_RANGE, POWDER_VALUE_RANGE, FEED_RATE_RANGE)

    number_samples = len(sample_indices) if sample_indices else len(data)
    if not sample_indices:
        sample_indices = range(len(data))

    # # Average the shape of the label images.
    # average_labels = np.array(read_labels(FOLDER_LABELS))
    # average_labels = np.round(np.mean(average_labels, axis=0) / 255) * 255

    images = []
    for index in sample_indices:
        image = np.empty(INPUT_SIZE)
        for channel in range(INPUT_CHANNELS):
            image[channel, ...] = data.loc[index][channel] / value_ranges[channel][-1] * 255

        # # First channel contains the average shape of label image with brightness representing powder value.
        # image[0, ...] = average_labels * (data.loc[index][1] / POWDER_VALUE_RANGE[-1])

        # # Second channel contains an ellipse whose height represents laser power and whose length represents feed rate.
        # max_radius_y = (INPUT_SIZE[1] // 2)
        # max_radius_x = (INPUT_SIZE[2] // 2)
        # y, x = np.indices(INPUT_SIZE[1:])
        # y = y - max_radius_y
        # x = x - max_radius_x
        # radius_y = (data.loc[index][0] / LASER_POWER_RANGE[-1]) * max_radius_y
        # radius_x = (data.loc[index][2] / FEED_RATE_RANGE[-1]) * max_radius_x
        # image[1, ...] = (x ** 2 / radius_x ** 2 + y ** 2 / radius_y ** 2) <= 1
        # image[1, ...] *= 255.0
        
        images.append(image.astype(np.uint8))
    return images

def read_inputs(filename: str, sheetindex: int, sample_indices: List[int] = None) -> np.ndarray:
    """Return a Series of class labels for each set of input parameters."""
    print("Reading inputs...")
    if experiment_number == "faces":
        return np.repeat([0, 1], TOTAL_SAMPLES//2)
    
    data = pd.read_excel(os.path.join(FOLDER_ROOT, filename), sheet_name=sheetindex, usecols=(1, 2, 3))
    value_ranges = (LASER_POWER_RANGE, POWDER_VALUE_RANGE, FEED_RATE_RANGE)

    unique_values = [sorted(pd.unique(data.iloc[:, column])) for column in range(data.shape[1])]
    # unique_labels = np.empty([len(_) for _ in unique_values], int)
    # unique_labels = np.arange(unique_labels.size).reshape(unique_labels.shape)

    if not sample_indices:
        sample_indices = range(len(data))
    number_samples = len(sample_indices)

    labels = np.empty((number_samples,), int)
    for i, index in enumerate(sample_indices):
        labels[i] = unique_values[0].index(data.iloc[index, 0])
        # TODO: Make generic (not just 3 columns)
        # labels[i] = unique_labels[
        #     unique_values[0].index(data.iloc[index, 0]),
        #     unique_values[1].index(data.iloc[index, 1]),
        #     unique_values[2].index(data.iloc[index, 2]),
        # ]
    
    labels = np.repeat([1, 2], number_samples//2)
    
    return labels

def read_labels(folder: str, sample_indices: List[int] = None) -> List[np.ndarray]:
    print("Reading labels...")
    """Return a list of arrays of data in the specified folder."""
    filenames = glob.glob(os.path.join(folder, '*.jpg'))
    filenames = sorted(filenames)
    # Only use the filenames that match the specified sample numbers. Assumes that filename numbers start from 1 and are contiguous.
    if sample_indices:
        assert len(sample_indices) <= len(filenames), f"The requested number of samples {len(sample_indices)} exceeds the number of available files {len(filenames)}."
        filenames = [filenames[index] for index in sample_indices]
    sample_size = len(filenames)

    labels = []
    for filename in filenames:
        with Image.open(filename) as image:
            # Resize image.
            image = image.resize((OUTPUT_SIZE[2], OUTPUT_SIZE[1]))

            # # Blur to reduce noise.
            # image = image.filter(ImageFilter.GaussianBlur(radius=15.0))

            # Convert to a numpy array.
            array = np.asarray(image, dtype=np.uint8).transpose((2, 0, 1))
            # Convert to grayscale.
            array = np.mean(array, axis=0)
            
            # # Perform k-means clustering.
            # k = 3
            # array = k_means(array, k=k, centroids=[0, 50, 200])

            # # Scale values to [0, 1].
            # array = array - np.min(array)
            # array = array / np.max(array)

            # # Perform closing and opening to remove noise and holes.
            # array = sp.ndimage.binary_opening(array)
            # array = sp.ndimage.binary_closing(array)

            # array = array * 255

            # Scale values to [-1, 1]. Required for use with GAN, where the generator outputs images in [-1, 1].
            array = array / 255 * 2 - 1

            labels.append(array.reshape(OUTPUT_SIZE))

    return labels

def k_means(array: np.ndarray, k: int, centroids: list) ->  np.ndarray:
    """Perform K-means clustering on the given 2D array and return an array of the same shape where each pixel is replaced with its cluster's centroid."""
    assert len(centroids) == k, f"The length of the initial centroids {len(centroids)} does not match k = {k}."
    
    MAX_ITERATIONS = 1000

    clusters = np.empty(array.shape)
    for i in range(MAX_ITERATIONS):
        previous_clusters = clusters.copy()

        # Assign each value in the array to a cluster.
        clusters = np.argmin(
            (np.dstack((array,)*k) - np.array(centroids)) ** 2,
            axis=2,
        )

        # Update the centroids based on the new clusters.
        centroids = [np.mean(array[clusters == cluster]) for cluster in range(k)]
        
        # Stop iterating if convergence is reached.
        if np.all(clusters == previous_clusters):
            for cluster in range(k):
                clusters[clusters == cluster] = centroids[cluster]
            break
    
    return clusters

def write_image(array: np.ndarray, filename: str) -> None:
    with Image.fromarray(array.astype(np.uint8)) as image:
        image.save(filename)

# with Image.open("DED/Exp#1_(sheet#1)/01.jpg") as image:
#     # Resize.
#     image = image.resize((OUTPUT_SIZE[2], OUTPUT_SIZE[1]))
#     array = np.asarray(image, dtype=np.uint8).transpose((2, 0, 1))
#     # Convert to grayscale.
#     array = np.mean(array, axis=0).reshape(OUTPUT_SIZE[1:])
# k = 3
# clusters = k_means(array, k, [0, 255])
# print(clusters)
# clusters = clusters / np.max(clusters) * 255
# clusters = np.dstack((clusters,)*3)
# print(np.min(clusters), np.max(clusters))
# write_image(clusters, 'test.png')