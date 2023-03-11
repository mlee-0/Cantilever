"""
Information about parameters and functions for reading and writing files. Run this script to generate and save label images.
"""


import colorsys
from dataclasses import dataclass
import glob
import os
import pickle
import re
import time
from typing import *

import matplotlib
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

# Folders and files.
FOLDER_ROOT = "." if not GOOGLE_COLAB else "drive/My Drive/Colab Notebooks"
FOLDER_CHECKPOINTS = os.path.join(FOLDER_ROOT, "Checkpoints")

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
length = Parameter(low=0.8, high=3.2, step=0.2, precision=1, name="Length", units="m")
height = Parameter(low=0.4, high=1.6, step=0.2, precision=1, name="Height", units="m")
width = Parameter(low=0.4, high=1.6, step=0.2, precision=1, name="Width", units="m")
load = Parameter(low=500, high=500, step=5, precision=0, name="Load", units="N")
position = Parameter(low=0.2, high=1.0, step=0.2, precision=1, name="Load Position", units="fraction")
angle_1 = Parameter(low=0, high=90, step=5, precision=0, name="Angle XY", units="Degrees")
angle_2 = Parameter(low=0, high=90, step=5, precision=0, name="Angle XZ", units="Degrees")

# Names of quantities that are derived from randomly generated values.
KEY_SAMPLE_NUMBER = "Sample Number"
KEY_X_LOAD_2D = "Load X (2D)"
KEY_Y_LOAD_2D = "Load Y (2D)"
KEY_X_LOAD_3D = "Load X (3D)"
KEY_Y_LOAD_3D = "Load Y (3D)"
KEY_Z_LOAD_3D = "Load Z (3D)"
KEY_NODES_LENGTH = "Nodes Length"
KEY_NODES_HEIGHT = "Nodes Height"
KEY_NODES_WIDTH = "Nodes Width"
KEY_LOAD_NODE_NUMBER = "Load Node Number"

# Size of input images (height, width). Must have the same aspect ratio as the largest possible cantilever geometry.
INPUT_SIZE = (16, 32)
INPUT_SIZE_3D = (16, 32, 16)
# Number of nodes to create in each direction in FEA.
NODES_X = 32
NODES_Y = 16
NODES_Z = 16
# Size of output images (height, width) produced by the network. Each pixel corresponds to a single node in the FEA mesh.
OUTPUT_SIZE = (NODES_Y, NODES_X)
OUTPUT_SIZE_3D = (NODES_Y, NODES_X, NODES_Z)


def split_dataset(dataset_size: int, splits: List[float]) -> List[int]:
    """Return the subset sizes according to the fractions defined in `splits`."""

    assert sum(splits) == 1.0, f"The fractions {splits} must sum to 1."

    # Define the last subset size as the remaining number of data to ensure that they all sum to dataset_size.
    subset_sizes = []
    for fraction in splits[:-1]:
        subset_sizes.append(int(fraction * dataset_size))
    subset_sizes.append(dataset_size - sum(subset_sizes))

    return subset_sizes

def plot_histogram(values: np.ndarray, title=None) -> None:
    plt.figure()
    plt.hist(values, bins=100, rwidth=0.75, color="#0095ff")
    if title:
        plt.title(title)
    plt.show()

def read_samples(filepath: str) -> pd.DataFrame:
    """Return the sample values found in the given file."""
    
    try:
        samples = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f'"{filepath}" not found.')
        return None
    else:
        print(f"Found {len(samples):,} samples in {filepath}.")
        return samples

def generate_input_images(samples: pd.DataFrame, is_3d: bool) -> np.ndarray:
    """Return a 4D array of images for each of the specified sample values, with dimensions: (samples, channels, height, width)."""
    
    time_start = time.time()

    DATA_TYPE = int

    number_samples = len(samples)
    h, w = INPUT_SIZE
    images = np.zeros((number_samples, 3 if not is_3d else 5, h, w), DATA_TYPE)

    for i in range(number_samples):
        pixel_length = int(samples[KEY_NODES_LENGTH][i])
        pixel_height = int(samples[KEY_NODES_HEIGHT][i])
        pixel_width = int(samples[KEY_NODES_WIDTH][i])

        # Create a channel with a rectangle representing the length and height of the cantilever.
        images[i, 0, :pixel_height, :pixel_length] = 1
        # Add a single pixel representing where the load is.
        position_x = int(np.round(samples[position.name][i] * samples[KEY_NODES_LENGTH][i]) - 1)
        position_y = int(samples[KEY_NODES_HEIGHT][i] - 1)
        images[i, 0, position_y, position_x] = 2

        # Create a channel with a rectangle representing the height and width of the cantilever.
        if is_3d:
            images[i, 1, :pixel_height, :pixel_width] = 1

        # Create a channel with a line representing the XY angle.
        r = np.arange(max(h, w))
        x = r * np.cos(samples[angle_1.name][i] * np.pi/180) + w/2
        y = r * np.sin(samples[angle_1.name][i] * np.pi/180) + h/2
        x = x.astype(int)
        y = y.astype(int)
        inside_image = (x >= 0) * (x < w) * (y >= 0) * (y < h)
        images[i, 1 if not is_3d else 2, y[inside_image], x[inside_image]] = 1
        images[i, 1 if not is_3d else 2, ...] = np.flipud(images[i, 1 if not is_3d else 2, ...])
        
        # Create a channel with a line representing the XZ angle.
        if is_3d:
            r = np.arange(max(h, w))
            x = r * np.cos(samples[angle_2.name][i] * np.pi/180) + w/2
            y = r * np.sin(samples[angle_2.name][i] * np.pi/180) + h/2
            x = x.astype(int)
            y = y.astype(int)
            inside_image = (x >= 0) * (x < w) * (y >= 0) * (y < h)
            images[i, 3, y[inside_image], x[inside_image]] = 1
            images[i, 3, ...] = np.flipud(images[i, 3, ...])

    time_end = time.time()
    print(f"Generated {images.shape[0]:,} input images in {time_end - time_start:.2f} seconds.")

    return images

def generate_input_images_3d(samples: pd.DataFrame) -> np.ndarray:
    """Return a 5D array of images for each of the specified sample values, with dimensions: (samples, channels, height, width, depth)."""

    time_start = time.time()

    DATA_TYPE = np.uint8

    number_samples = len(samples)
    images = [None] * number_samples

    for i in range(number_samples):
        pixel_length = int(samples[KEY_NODES_LENGTH][i])
        pixel_height = int(samples[KEY_NODES_HEIGHT][i])
        pixel_width = int(samples[KEY_NODES_WIDTH][i])
        
        channels = []
        h, w, d = INPUT_SIZE_3D

        # Create a channel with a white rectangular volume representing the length, height, and width of the cantilever.
        channel = np.zeros((h, w, d), dtype=DATA_TYPE)
        channel[:pixel_height, :pixel_length, :pixel_width] = 255
        channels.append(channel)

        # Create a channel with a gray line of pixels representing the load magnitude and both angles. The line is oriented by both angles and extends from the midpoint of the volume. The brightness of the line represents the load magnitude.
        channel = np.zeros((h, w, d), dtype=DATA_TYPE)
        r = np.arange(max(channel.shape))
        x = r * np.cos(samples[angle_1.name][i] * np.pi/180) * np.cos(samples[angle_2.name][i]) + w/2
        y = r * np.sin(samples[angle_1.name][i] * np.pi/180) + h/2
        z = r * np.cos(samples[angle_1.name][i] * np.pi/180) * np.sin(samples[angle_2.name][i]) + d/2
        x = x.astype(int)
        y = y.astype(int)
        z = z.astype(int)
        inside_image = (x >= 0) * (x < w) * (y >= 0) * (y < h) * (z >= 0) * (z < d)
        channel[y[inside_image], x[inside_image], z[inside_image]] = 255 * (samples[load.name][i] / load.high)
        channels.append(channel)
        
        # # Add two channels with vertical and horizontal indices.
        # indices = np.indices((h, w), dtype=DATA_TYPE)
        # channels.append(indices[0, ...])
        # channels.append(indices[1, ...])
        
        images[i] = channels
    
    images = np.array(images)

    time_end = time.time()
    print(f"Generated {images.shape[0]} input images in {time_end - time_start:.2f} seconds.")

    return images

def generate_label_images(samples: pd.DataFrame, folder: str, is_3d: bool) -> np.ndarray:
    """Return a 4D array of images for the FEA text files found in the specified folder that correspond to the given samples, with dimensions: (samples, channels, height, width)."""
    
    time_start = time.time()

    DATA_TYPE = np.float32

    number_samples = len(samples)
    
    # Get and sort all FEA filenames.
    filenames = glob.glob(os.path.join(folder, "*.txt"))
    filenames = sorted(filenames)

    # Only use the filenames that match the specified sample numbers (filename "...000123.txt" matches sample number 123).
    assert len(samples) <= len(filenames), f"{folder} only contains {len(filenames)} .txt files, which is less than the requested {number_samples}."
    filenames = [_ for _ in filenames if int(re.split("_|\.", os.path.basename(_))[1]) in samples[KEY_SAMPLE_NUMBER].values]

    # Store all data in a single array, initialized with a default value. The order of values in the text files is determined by ANSYS.
    DEFAULT_VALUE = 0
    labels = np.full(
        (number_samples, NODES_Z if is_3d else 1, *OUTPUT_SIZE),
        DEFAULT_VALUE, dtype=DATA_TYPE
    )
    for i, (index, filename) in enumerate(zip(samples.index, filenames)):
        with open(filename, "r") as file:
            lines = file.readlines()
        
        # Assume each line contains the result followed by the corresponding nodal coordinates, in the format: value, x, y, z. Round the coordinates to the specified number of digits to eliminate rounding errors from FEA.
        node_values = [
            [float(value) if j == 0 else round(float(value), 2) for j, value in enumerate(line.split(","))]
            for line in lines
        ]
        # Sort the values using the coordinates.
        node_values.sort(key=lambda _: (_[3], _[2], _[1]))

        image_channels = int(samples[KEY_NODES_WIDTH][index])
        image_height = int(samples[KEY_NODES_HEIGHT][index])
        image_length = int(samples[KEY_NODES_LENGTH][index])

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
            print(f"Reading label {i+1} / {number_samples}...", end="\r")
    print()

    time_end = time.time()
    print(f"Generated {labels.shape[0]:,} label images in {time_end - time_start:.2f} seconds.")

    return labels

def plot_loss(figure: matplotlib.figure.Figure, epochs: list, loss: List[list], labels: List[str], start_epoch: int = None) -> None:
    """
    Plot loss values over epochs on the given figure.

    Parameters:
    `figure`: A figure to plot on.
    `epochs`: A sequence of epoch numbers.
    `loss`: A list of lists of loss values, each of which are plotted as separate lines. Each nested list must have the same length as `epochs`.
    `labels`: A list of strings to display in the legend for each item in `loss`.
    `start_epoch`: The epoch number at which to display a horizontal line to indicate the start of the current training session.
    """
    figure.clear()
    axis = figure.add_subplot(1, 1, 1)  # Number of rows, number of columns, index
    
    # markers = (".:", ".-")
    colors = (Colors.BLUE, Colors.ORANGE)

    # Plot each set of loss values.
    for i, loss_i in enumerate(loss):
        if not len(loss_i):
            continue
        color = colors[i % len(colors)]
        axis.semilogy(epochs[:len(loss_i)], loss_i, ".-", color=color, label=labels[i])
        axis.annotate(f"{loss_i[-1]:,.2f}", (epochs[-1 - (len(epochs)-len(loss_i))], loss_i[-1]), color=color, fontsize=10)
    
    # Plot a vertical line indicating when the current training session began.
    if start_epoch:
        axis.vlines(start_epoch - 0.5, 0, max([max(_) for _ in loss]), colors=(Colors.GRAY,), label="Current session starts")
    
    axis.legend()
    axis.set_xlabel("Epochs")
    axis.set_ylabel("Loss")
    axis.grid(axis="y")

def plot_input_image_3d(array: np.ndarray) -> None:
    """Show a 3D voxel plot for each channel of the given 4D array with shape (channels, height, length, width)."""
    fig = plt.figure()
    channels = array.shape[0]

    for channel in range(channels):
        ax = fig.add_subplot(1, channels, channel+1, projection="3d")
        filled = array[channel, ...] != 0
        rgb = np.stack([array[channel, ...]]*3, axis=-1)
        # rgb = np.concatenate(
        #     (rgb, np.where(array[channel, ...] != 0, 255, 255/4)),
        #     axis=-1,
        # )
        rgb = rgb / 255
        ax.voxels(
            filled=filled,
            facecolors=rgb,
            linewidth=0.25,
            edgecolors=(0.5, 0.5, 0.5),
        )
        ax.set_title(f"Channel {channel+1}")
    
    plt.show()

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
    """Convert an array of values of any range to an array of RGB colors ranging from red to blue. The returned array has one more dimension than the input array: (..., 3)."""
    # Make copy of array so that the original array is not modified.
    array = np.copy(array)

    # Scale the values.
    if divide_by is not None:
        array /= divide_by
    else:
        array /= np.max(array)
    # Invert the values so that red represents high values.
    array = 1 - array
    # Scale the values to the range of hues from red to blue to match standard colors used in FEA.
    array = array * (240/360)
    # Create an array of HSV values, using the array values as hues.
    SATURATION, VALUE = 1, 2/3
    array = np.stack((array, np.full(array.shape, SATURATION), np.full(array.shape, VALUE)), axis=-1)
    # Convert the array to RGB values.
    array_flatten = array.reshape(-1, array.shape[-1])
    for i in range(array_flatten.shape[0]):
        array_flatten[i, :] = colorsys.hsv_to_rgb(*array_flatten[i, :])
    array = array_flatten.reshape(array.shape)
    array *= 255
    
    return array

def write_image(array: np.ndarray, filename: str) -> None:
    with Image.fromarray(array.astype(np.uint8)) as image:
        image.save(filename)
    print(f"Saved array with shape {array.shape} to {filename}.")

def read_pickle(filepath: str) -> Any:
    time_start = time.time()
    with open(filepath, "rb") as f:
        x = pickle.load(f)
    time_end = time.time()
    print(f"Loaded {type(x)} from {filepath} in {time_end - time_start:.2f} seconds.")

    return x

def write_pickle(x: object, filepath: str) -> None:
    assert filepath.endswith(".pickle")
    time_start = time.time()
    with open(filepath, "wb") as f:
        pickle.dump(x, f)
    time_end = time.time()
    print(f"Saved {type(x)} to {filepath} in {time_end - time_start:.2f} seconds.")

# Colors for plots.
class Colors:
    RED = "#ff4040"
    RED_DARK = "#9e2828"
    RED_LIGHT = "#ffb6b6"
    ORANGE = "#ff8000"
    ORANGE_DARK = "#9e4f00"
    ORANGE_LIGHT = "#ffce9e"
    YELLOW = "#ffbf00"
    YELLOW_DARK = "#9e7600"
    YELLOW_LIGHT = "#ffe79e"
    BLUE = "#0095ff"
    BLUE_DARK = "#005c9e"
    BLUE_LIGHT = "#9ed7ff"
    GRAY = "#808080"
    GRAY_DARK = "#404040"
    GRAY_LIGHT = "#bfbfbf"


if __name__ == "__main__":
    # Convert text files to an array and save them as .pickle files.
    samples = read_samples(os.path.join(FOLDER_ROOT, "samples.csv"))

    folder = os.path.join(FOLDER_ROOT, "Labels 2D")
    labels = generate_label_images(samples, folder, is_3d=not True)
    write_pickle(labels, os.path.join(folder, "labels.pickle"))

    # p = read_pickle("Cantilever/Labels 3D/labels.pickle")[:1000, ...]
    # max_value = np.max(p)
    # normalized = p / np.max(p, axis=(1, 2, 3), keepdims=True)

    # p = p.flatten()
    # normalized = normalized.flatten()
    
    # bins = 100

    # plt.figure()
    # plt.subplot(1, 2, 1)
    # plt.hist(p, bins=bins, label="Original")
    # plt.title("Original")
    # plt.subplot(1, 2, 2)
    # plt.hist(normalized, bins=bins)
    # plt.title("Averaged by sample")
    # plt.show()

    # plt.figure()
    # exponents = (1/2.2, 1/2.1, 0.5023404737562848, 1/1.9, 1/1.8, 1/1.7, 1/1.6, 1/1.5, 1)
    # titles = ('1/2.2', '1/2.1', '1/1.99', '1/1.9', '1/1.8', '1/1.7', '1/1.6', '1/1.5', '1')
    # for i, exponent in enumerate(exponents):
    #     plt.subplot(2, 5, i+1)
    #     transformed = p.flatten() ** exponent
    #     plt.hist(normalized, bins=bins, alpha=0.5, label="Target")
    #     plt.hist(transformed/transformed.max(), bins=bins, alpha=0.5, label="Transformed")
    #     plt.legend()
    #     plt.title(titles[i])
    # plt.show()