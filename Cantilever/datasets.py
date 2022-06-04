"""
Run this script to generate dataset files.
"""


import math
import os
import pickle
import random

import numpy as np
import pandas as pd

from setup import *


def generate_samples(number_samples: int, start: int = 1) -> pd.DataFrame:
    """Generate sample values for each parameter and return them as a DataFrame. Specify the starting sample number if generating samples to add to an existing dataset."""

    assert start != 0, f"The starting sample number {start} should be a positive integer."

    # Generate sample values for each parameter.
    samples = {}
    samples[KEY_SAMPLE_NUMBER] = range(start, start+number_samples)
    samples[load.name] = generate_uniform_values(number_samples, (load.low, load.high), load.step, load.precision)  # generate_logspace_values(..., skew_amount=2.0, skew_high=True)
    samples[angle_1.name] = generate_uniform_values(number_samples, (angle_1.low, angle_1.high), angle_1.step, angle_1.precision)  # generate_angles(..., std=45)
    samples[angle_2.name] = generate_uniform_values(number_samples, (angle_2.low, angle_2.high), angle_2.step, angle_2.precision)  # generate_angles(..., std=45)
    samples[length.name] = generate_uniform_values(number_samples, (length.low, length.high), length.step, length.precision)  # generate_logspace_values(..., skew_amount=1.0, skew_high=True)
    samples[height.name] = generate_uniform_values(number_samples, (height.low, height.high), height.step, height.precision)  # generate_logspace_values(..., skew_amount=1.0, skew_high=False)
    samples[width.name] = generate_uniform_values(number_samples, (width.low, width.high), width.step, width.precision)  # generate_logspace_values(..., skew_amount=1.0, skew_high=False)
    samples[elastic_modulus.name] = generate_uniform_values(number_samples, (elastic_modulus.low, elastic_modulus.high), elastic_modulus.step, elastic_modulus.precision)
    
    # Calculate the number of nodes in each direction.
    nodes_length = np.round(NODES_X * (samples[length.name] / length.high))
    nodes_height = np.round(NODES_Y * (samples[height.name] / height.high))
    nodes_width = np.round(NODES_Z * (samples[width.name] / width.high))

    samples[KEY_NODES_LENGTH] = nodes_length
    samples[KEY_NODES_HEIGHT] = nodes_height
    samples[KEY_NODES_WIDTH] = nodes_width
    
    # Calculate the x-, y-, z-components of the loads.
    x_loads = np.round(
        np.cos(samples[angle_2.name] * (np.pi/180)) * np.cos(samples[angle_1.name] * (np.pi/180)) * samples[load.name],
        load.precision,
    )
    y_loads = np.round(
        np.sin(samples[angle_1.name] * (np.pi/180)) * samples[load.name],
        load.precision,
    )
    z_loads = np.round(
        np.sin(samples[angle_2.name] * (np.pi/180)) * np.cos(samples[angle_1.name] * (np.pi/180)) * samples[load.name],
        load.precision,
    )
    samples[KEY_X_LOAD] = x_loads
    samples[KEY_Y_LOAD] = y_loads
    samples[KEY_Z_LOAD] = z_loads

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
        commands_define_samples = [f'*DIM,{samples_variable},ARRAY,{11},{number_samples}\n']
        for i in range(number_samples):
            commands_define_samples.append(
                f'{samples_variable}(1,{samples[KEY_SAMPLE_NUMBER][i]}) = {samples[load.name][i]},{samples[KEY_X_LOAD][i]},{samples[KEY_Y_LOAD][i]},{samples[KEY_Z_LOAD][i]},{samples[length.name][i]},{samples[height.name][i]},{samples[width.name][i]},{samples[elastic_modulus.name][i]},{samples[KEY_NODES_LENGTH][i]},{samples[KEY_NODES_HEIGHT][i]},{samples[KEY_NODES_WIDTH][i]}\n'
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


# Specify either "train" or "test".
DATASET = "train"
NUMBER_SAMPLES = 10000
# Must be 1 if creating a new dataset.
START_SAMPLE_NUMBER = 1
# Specify "w" (write) to create a new dataset, or specify "a" (append) to add new data to the existing dataset.
WRITE_MODE = "w"

# Dataset file names.
FILENAME_SAMPLES_TRAIN = "samples_train.csv"
FILENAME_SAMPLES_TEST = "samples_test.csv"

if __name__ == "__main__":
    if DATASET == "train":
        filename_sample = FILENAME_SAMPLES_TRAIN
        filename_ansys = "ansys_script_train.lgw"
    elif DATASET == "test":
        filename_sample = FILENAME_SAMPLES_TEST
        filename_ansys = "ansys_script_test.lgw"
    else:
        print(f"Invalid dataset type: {DATASET}")

    if START_SAMPLE_NUMBER > 1 and WRITE_MODE == "w":
        START_SAMPLE_NUMBER = 1
        print("Changed the starting sample number to 1 because generating a new dataset.")

    # Try to read sample values from the file if it already exists. If not, generate the samples.
    samples_existing = read_samples(os.path.join(FOLDER_ROOT, filename_sample))
    if samples_existing is not None:
        overwrite_existing_samples = input(f"Overwrite {filename_sample}? [y/n] ") == "y"
    else:
        overwrite_existing_samples = False
    if samples_existing is None or overwrite_existing_samples:
        samples = generate_samples(NUMBER_SAMPLES, start=START_SAMPLE_NUMBER)
        if WRITE_MODE == "a" and samples_existing is not None:
            samples = pd.concat((samples_existing, samples), axis=0, ignore_index=True)

    write_samples(samples, filename_sample, mode=WRITE_MODE)
    write_ansys_script(samples, filename_ansys)