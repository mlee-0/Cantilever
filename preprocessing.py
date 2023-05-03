"""Read and cache simulation data. Run this file to read simulation data, which are stored as text files, and convert them to tensors and save them as .pickle files."""


import colorsys
import glob
import os
import pickle
import time
from typing import *

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# Size of input images (height, width). Must have the same aspect ratio as the largest possible cantilever geometry.
INPUT_SIZE = (16, 32)
INPUT_SIZE_3D = (16, 32, 16)
# Size of output images (height, width) produced by the network. Each pixel corresponds to a single node in the FEA mesh.
OUTPUT_SIZE = (16, 32)
OUTPUT_SIZE_3D = (16, 32, 16)


def generate_simulation_parameters() -> List[Tuple[float, float, float, float]]:
    """Return a list of tuples of simulation parameters for each simulation."""

    return [
        (angle, length, height, position)
        for angle in np.arange(0, 90+1, 5).round(0)
        for length in np.arange(0.8, 3.2+0.1, 0.2).round(1)
        for height in np.arange(0.4, 1.6+0.1, 0.2).round(1)
        for position in np.arange(0.2, 1.0+0.1, 0.2).round(1)
    ]

def get_parameters_from_filename(filename: str) -> Tuple[float, ...]:
    """Return a tuple of the simulation parameters defined in the filename, which is assumed to have the format: `prefix_<parameter 1>_<...>_<parameter n>.txt`."""
    return tuple(float(_) for _ in filename[:-4].split('_')[1:])

def make_inputs(parameters: List[Tuple[float, float, float, float]]) -> np.ndarray:
    """Create inputs for each set of parameters as a 4D array, with dimensions: (data, channels, height, width)."""

    time_start = time.time()

    h, w = INPUT_SIZE
    array = np.zeros((len(parameters), 2, h, w), int)

    for i, (angle, length, height, position) in enumerate(parameters):
        pixel_length = round(length / 0.1)
        pixel_height = round(height / 0.1)

        # Create a channel with a rectangle representing the length and height of the cantilever.
        array[i, 0, :pixel_height, :pixel_length] = 1
        # Add a single pixel representing where the load is.
        position_x = round(position * pixel_length) - 1
        position_y = pixel_height - 1
        array[i, 0, position_y, position_x] = 2

        # Create a channel with a line representing the XY angle.
        r = np.arange(max(h, w))
        x = r * np.cos(angle * np.pi/180) + w/2
        y = r * np.sin(angle * np.pi/180) + h/2
        x = x.astype(int)
        y = y.astype(int)
        inside_image = (x >= 0) * (x < w) * (y >= 0) * (y < h)
        array[i, 1, y[inside_image], x[inside_image]] = 1
        array[i, 1, ...] = np.flipud(array[i, 1, ...])

    time_end = time.time()
    print(f"Generated {array.shape[0]:,} input images in {time_end - time_start:.2f} seconds.")

    return array

def make_inputs_3d(parameters: List[Tuple[float, float, float, float, float, float]]) -> np.ndarray:
    """Create inputs for each set of parameters as a 5D array, with dimensions: (samples, channels, height, width, depth)."""

    time_start = time.time()

    DATA_TYPE = int

    h, w, d = INPUT_SIZE_3D
    array = np.zeros((len(parameters), 4, h, w, d), DATA_TYPE)

    for i, (length, height, width, position, angle_1, angle_2) in enumerate(parameters):
        pixel_length = round(length / 0.1)
        pixel_height = round(height / 0.1)
        pixel_width = round(width / 0.1)

        # Create a channel with a white rectangular volume representing the length, height, and width of the cantilever.
        array[i, 0, :pixel_height, :pixel_length, :pixel_width] = 1

        # Create a channel with a gray line of pixels representing the load magnitude and both angles. The line is oriented by both angles and extends from the midpoint of the volume. The brightness of the line represents the load magnitude.
        r = np.arange(max((h, w, d)))
        x = r * np.cos(angle_1 * np.pi/180) * np.cos(angle_2) + w/2
        y = r * np.sin(angle_1 * np.pi/180) + h/2
        z = r * np.cos(angle_1 * np.pi/180) * np.sin(angle_2) + d/2
        x = x.astype(int)
        y = y.astype(int)
        z = z.astype(int)
        inside_image = (x >= 0) * (x < w) * (y >= 0) * (y < h) * (z >= 0) * (z < d)
        array[i, 1, y[inside_image], x[inside_image], z[inside_image]] = 1

        # # Add two channels with vertical and horizontal indices.
        # indices = np.indices((h, w), dtype=DATA_TYPE)
        # channels.append(indices[0, ...])
        # channels.append(indices[1, ...])

    time_end = time.time()
    print(f"Generated {array.shape[0]} input images in {time_end - time_start:.2f} seconds.")

    return array

def make_labels(folder: str) -> np.ndarray:
    """Return a 4D array of images for the FEA text files found in the specified folder, with dimensions: (samples, channels, height, width)."""

    time_start = time.time()

    parameters = generate_simulation_parameters()

    # Get and sort all FEA filenames.
    filenames = glob.glob(os.path.join(folder, '*.txt'))
    filenames = sorted(filenames, key=lambda filename: parameters.index(get_parameters_from_filename(filename)))

    # Store all data in a single array, initialized with a default value. The order of values in the text files is determined by ANSYS.
    array = np.zeros((len(filenames), 1, *OUTPUT_SIZE), dtype=np.float32)
    for i, filename in enumerate(filenames):
        with open(filename, 'r') as file:
            lines = file.readlines()

        # Get the simulation parameters from the filename.
        angle, length, height, position = get_parameters_from_filename(filename)
        
        # Assume each line contains the result followed by the corresponding nodal coordinates, in the format: value, x, y, z. Round the coordinates to the specified number of digits to eliminate rounding errors from FEA.
        values = [
            [float(value) if j == 0 else round(float(value), 2) for j, value in enumerate(line.split(','))]
            for line in lines
        ]
        # Sort the values using the coordinates.
        values.sort(key=lambda _: (_[-1], _[-2], _[-3]))
        # Remove coordinates.
        values = [_[0] for _ in values]

        pixel_height = round(height / 0.1)
        pixel_length = round(length / 0.1)

        # Insert the values into the combined array, aligned top-left. Flip along height due to inverted y-axis in FEA.
        array[i, 0, :pixel_height, :pixel_length] = np.reshape(values, (pixel_height, pixel_length))[::-1, :]
        
        if (i+1) % 10 == 0:
            print(f"Reading label {i+1} / {len(filenames)}...", end='\r')
    print()

    time_end = time.time()
    print(f"Generated {array.shape[0]:,} label images in {time_end - time_start:.2f} seconds.")

    return array

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


if __name__ == '__main__':
    # Convert text files to an array and save them as .pickle files.
    folder = "Stress 2D 2023-05"
    labels = make_labels(folder)
    write_pickle(labels, os.path.join(folder, "labels.pickle"))