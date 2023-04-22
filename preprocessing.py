"""
Run this script to generate dataset files.
"""


import os
import random

import numpy as np
import pandas as pd

from helpers import *


def generate_samples(start: int = 1, is_3d: bool=False) -> pd.DataFrame:
    """Generate sample values for each parameter and return them as a DataFrame. Specify the starting sample number if generating samples to add to an existing dataset."""

    assert start != 0, f"The starting sample number {start} should be a positive integer."

    # Generate sample values for each parameter.
    samples = {}
    if not is_3d:
        data = np.meshgrid(*[np.arange(p.low, p.high + (p.step/2), p.step).round(p.precision) for p in [load, angle_1, length, height, position]])
        number_samples = data[0].size
        samples[KEY_SAMPLE_NUMBER] = range(start, start+number_samples)
        samples[load.name] = data[0].flatten()
        samples[angle_1.name] = data[1].flatten()
        samples[angle_2.name] = np.zeros(number_samples)
        samples[length.name] = data[2].flatten()
        samples[height.name] = data[3].flatten()
        samples[width.name] = np.zeros(number_samples)
        samples[position.name] = data[4].flatten()
    else:
        data = np.meshgrid(*[np.arange(p.low, p.high + (p.step/2), p.step).round(p.precision) for p in [load, angle_1, angle_2, length, height, width, position]])
        number_samples = data[0].size
        samples[KEY_SAMPLE_NUMBER] = range(start, start+number_samples)
        samples[load.name] = data[0].flatten()
        samples[angle_1.name] = data[1].flatten()
        samples[angle_2.name] = data[2].flatten()
        samples[length.name] = data[3].flatten()
        samples[height.name] = data[4].flatten()
        samples[width.name] = data[5].flatten()
        samples[position.name] = data[6].flatten()

    # Calculate the number of nodes in each direction.
    samples[KEY_NODES_LENGTH] = np.round(NODES_X * (samples[length.name] / length.high))
    samples[KEY_NODES_HEIGHT] = np.round(NODES_Y * (samples[height.name] / height.high))
    samples[KEY_NODES_WIDTH] = np.round(NODES_Z * (samples[width.name] / width.high))

    # Calculate the node number of the node the load acts on.
    samples[KEY_LOAD_NODE_NUMBER] = np.round(samples[position.name] * samples[KEY_NODES_LENGTH]).astype(int)
    samples[KEY_LOAD_NODE_NUMBER][samples[KEY_LOAD_NODE_NUMBER] >= 2] += 1
    samples[KEY_LOAD_NODE_NUMBER][samples[position.name] == 1.0] = 2
    
    # Calculate the x-, y-, z-components of the loads.
    x_loads_2d = np.round(np.cos(samples[angle_1.name] * (np.pi/180)) * samples[load.name], load.precision)
    y_loads_2d = np.round(np.sin(samples[angle_1.name] * (np.pi/180)) * samples[load.name], load.precision)
    x_loads_3d = np.round(
        np.cos(samples[angle_2.name] * (np.pi/180)) * np.cos(samples[angle_1.name] * (np.pi/180)) * samples[load.name],
        load.precision,
    )
    y_loads_3d = np.round(
        np.sin(samples[angle_1.name] * (np.pi/180)) * samples[load.name],
        load.precision,
    )
    z_loads_3d = np.round(
        np.sin(samples[angle_2.name] * (np.pi/180)) * np.cos(samples[angle_1.name] * (np.pi/180)) * samples[load.name],
        load.precision,
    )
    samples[KEY_X_LOAD_2D] = x_loads_2d
    samples[KEY_Y_LOAD_2D] = y_loads_2d
    samples[KEY_X_LOAD_3D] = x_loads_3d
    samples[KEY_Y_LOAD_3D] = y_loads_3d
    samples[KEY_Z_LOAD_3D] = z_loads_3d

    samples = pd.DataFrame.from_dict(samples)
    
    return samples

def generate_uniform_values(number_samples: int, bounds: Tuple[float, float], step, precision) -> np.ndarray:
    """Generate uniformly distributed, evenly spaced values."""

    values = np.arange(bounds[0], bounds[1]+step, step)
    values = random.choices(values, k=number_samples)
    values = np.array(values)
    values = np.round(values, precision)

    return values

def generate_logspace_values(number_samples: int, bounds: Tuple[float, float], step, precision, skew_amount: float, skew_high: bool) -> np.ndarray:
    """Generate values that are more concentrated at one end of a range."""

    population = np.arange(bounds[0], bounds[1]+step, step)
    weights = np.logspace(0, skew_amount, len(population))
    if not skew_high:
        weights = weights[::-1]
    values = random.choices(population, weights=weights, k=number_samples)
    values = np.array(values)
    values = np.round(values / step) * step
    values = np.round(values, precision)
    
    plot_histogram(values)

    return values

def generate_angles(number_samples: int, bounds: Tuple[float, float], step, precision, std: int) -> np.ndarray:
    """Generate angle samples using a distribution with two peaks centered at 90 and 270 degrees."""

    values = np.append(
        np.random.normal(90, std, number_samples//2),
        np.random.normal(270, std, number_samples//2),
    )
    np.random.shuffle(values)
    assert values.size == number_samples, f"The number of samples {number_samples} should be even."
    values = np.round(values / step) * step
    values = np.round(values, precision)
    # Convert values outside [0, 360] to the equivalent value within that range.
    values = np.mod(values, 360)
    assert not np.any((values > bounds[1]) | (values < bounds[0])), f"Angle values were generated outside the specified range: {bounds[0]} to {bounds[1]}."

    plot_histogram(values)

    return values

def find_uniform_bins(data, bins=10):
    """Return an array of bin edges for which each bin has the almost same number of data and an array of bin edges within the range [0, 1]."""

    # Order each value in the data.
    data_ = data[data > 0].flatten().numpy()
    data_.sort()

    # Find the bin edges for the data's range and the bin edges for the range [0, 1].
    bin_indices = np.linspace(0, data_.size-1, bins+1).round().astype(int)
    bin_edges = [0] + [data_[i] for i in bin_indices[1:]]
    normalized_bin_edges = np.linspace(0, 1, bins+1)

    return bin_edges, normalized_bin_edges

def transform_uniform_bins(data, bin_edges, normalized_bin_edges) -> np.ndarray:
    transformed = np.zeros(data.shape)
    for low, high, normalized_low, normalized_high in zip(bin_edges[:-1], bin_edges[1:], normalized_bin_edges[:-1], normalized_bin_edges[1:]):
        # For the last bin, include the upper bound.
        if high != bin_edges[-1]:
            mask = (data >= low) & (data < high)
        else:
            mask = (data >= low) & (data <= high)

        transformed[mask] = (data[mask] - low) / (high - low)
        transformed[mask] = transformed[mask] * (normalized_high - normalized_low) + normalized_low
    
    return transformed

def untransform_uniform_bins(data, bin_edges, normalized_bin_edges) -> np.ndarray:
    untransformed = np.zeros(data.shape)
    for low, high, normalized_low, normalized_high in zip(bin_edges[:-1], bin_edges[1:], normalized_bin_edges[:-1], normalized_bin_edges[1:]):
        # For the last bin, include the upper bound.
        if normalized_high != normalized_bin_edges[-1]:
            mask = (data >= normalized_low) & (data < normalized_high)
        else:
            mask = (data >= normalized_low) & (data <= normalized_high)

        untransformed[mask] = (data[mask] - normalized_low) / (normalized_high - normalized_low)
        untransformed[mask] = untransformed[mask] * (high - low) + low
    
    return untransformed

def write_samples(samples: pd.DataFrame, filename: str, mode: str = 'w') -> None:
    """
    Write the specified sample values to a file.

    `mode`: Write to a file ('w') or append to an existing file ('a').
    """

    ALLOWED_MODES = ['w', 'a']
    assert mode in ALLOWED_MODES, f"The specified mode '{mode}' should be one of {ALLOWED_MODES}."

    samples.to_csv(os.path.join(FOLDER_ROOT, filename), header=(mode == 'w'), index=False, mode=mode)
    print(f"{'Wrote' if mode == 'w' else 'Appended'} samples in {filename}.")

def write_ansys_script(samples: pd.DataFrame, filename: str) -> None:
    """Write a text file containing ANSYS commands used to automate FEA and generate stress contour images."""

    number_samples = len(samples)
    # Read the template script.
    with open(os.path.join(FOLDER_ROOT, 'ansys_template.lgw'), 'r') as file:
        lines = file.readlines()
    
    # Number of digits used for numbers in file names (for example, 6 results in "xxx_000123.txt").
    NUMBER_DIGITS = 6
    
    # Replace placeholder lines in the template script.
    with open(os.path.join(FOLDER_ROOT, filename), 'w') as file:
        # Initialize dictionary of placeholder strings (keys) and strings they should be replaced with (values).
        placeholder_substitutions = {}
        # Define names of variables.
        loop_variable = 'i'
        samples_variable = 'samples'
        # Add commands that define the array containing generated samples.
        commands_define_samples = [f'*DIM,{samples_variable},ARRAY,{13},{number_samples}\n']
        for i in range(number_samples):
            commands_define_samples.append(
                f'{samples_variable}(1,{samples[KEY_SAMPLE_NUMBER][i]}) = {samples[load.name][i]},{samples[KEY_X_LOAD_2D][i]},{samples[KEY_Y_LOAD_2D][i]},{samples[KEY_X_LOAD_3D][i]},{samples[KEY_Y_LOAD_3D][i]},{samples[KEY_Z_LOAD_3D][i]},{samples[KEY_LOAD_NODE_NUMBER][i]},{samples[length.name][i]},{samples[height.name][i]},{samples[width.name][i]},{samples[KEY_NODES_LENGTH][i]},{samples[KEY_NODES_HEIGHT][i]},{samples[KEY_NODES_WIDTH][i]}\n'
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


if __name__ == "__main__":
    # Must be 1 if creating a new dataset.
    START_SAMPLE_NUMBER = 1
    # Specify "w" (write) to create a new dataset, or specify "a" (append) to add new data to the existing dataset.
    WRITE_MODE = "w"
    # 2D (False) or 3D (True) dataset.
    is_3d = False

    filename_sample = "samples.csv"
    filename_ansys = "ansys_script.lgw"

    if START_SAMPLE_NUMBER > 1 and WRITE_MODE == "w":
        START_SAMPLE_NUMBER = 1
        print("Changed the starting sample number to 1 because generating a new dataset.")

    # Try to read sample values from the file if it already exists. If not, generate the samples.
    samples = read_samples(os.path.join(FOLDER_ROOT, filename_sample))
    if samples is not None:
        overwrite_existing = input(f"Overwrite {filename_sample}? [y/n] ") == "y"
    else:
        overwrite_existing = False
    if samples is None or overwrite_existing:
        samples_new = generate_samples(start=START_SAMPLE_NUMBER)
        if WRITE_MODE == "a" and samples is not None:
            samples = pd.concat((samples, samples_new), axis=0, ignore_index=True)
        else:
            samples = samples_new

    write_samples(samples, filename_sample, mode=WRITE_MODE)
    write_ansys_script(samples, filename_ansys)