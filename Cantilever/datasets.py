'''
Run this script to generate datasets.
'''

import math
import os
import random

import numpy as np
import pandas as pd

from setup import *


def generate_samples(number_samples: int, start: int = 1) -> dict:
    """Generate sample values for each parameter and return them as a dictionary. Specify the starting sample number if generating samples to add to an existing dataset."""

    assert start != 0, f"The starting sample number {start} should be a positive integer."

    # Generate sample values for each parameter.
    samples = {}
    samples[KEY_SAMPLE_NUMBER] = range(start, start+number_samples)
    samples[load.name] = generate_logspace_values(number_samples, (load.low, load.high), load.step, load.precision, skew_amount=3.0, skew_high=True)
    samples[angle.name] = generate_angles(number_samples, (angle.low, angle.high), angle.step, angle.precision, std=45)
    samples[length.name] = generate_uniform_values(number_samples, (length.low, length.high), length.step, length.precision)
    samples[height.name] = generate_uniform_values(number_samples, (height.low, height.high), height.step, height.precision)
    samples[elastic_modulus.name] = generate_uniform_values(number_samples, (elastic_modulus.low, elastic_modulus.high), elastic_modulus.step, elastic_modulus.precision)
    
    # Calculate the image size corresponding to the geometry.
    image_lengths = np.round(OUTPUT_SIZE[2] * (samples[length.name] / length.high))
    image_heights = np.round(OUTPUT_SIZE[1] * (samples[height.name] / height.high))
    samples[KEY_IMAGE_LENGTH] = image_lengths
    samples[KEY_IMAGE_HEIGHT] = image_heights
    
    # Calculate the x- and y-components of the loads and corresponding angles.
    x_loads = np.round(
        np.cos(samples[angle.name] * (np.pi/180)) * samples[load.name] / (image_heights-1),
        load.precision
        )
    y_loads = np.round(
        np.sin(samples[angle.name] * (np.pi/180)) * samples[load.name] / (image_heights-1),
        load.precision
        )
    samples[KEY_X_LOAD] = x_loads
    samples[KEY_Y_LOAD] = y_loads
    
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
    assert values.size == number_samples, f"The number of samples {number_samples} should be even."
    values = np.round(values / step) * step
    values = np.round(values, precision)
    # Convert values outside [0, 360] to the equivalent value within that range.
    values = np.mod(values, 360)
    assert not np.any((values > bounds[1]) | (values < bounds[0])), f"Angle values were generated outside the specified range: {bounds[0]} to {bounds[1]}."

    plot_histogram(values)

    return values

def write_samples(samples: dict, filename: str, mode: str = 'w') -> None:
    """
    Write the specified sample values to a file.

    `mode`: Write to a file ('w') or append to an existing file ('a').
    """

    ALLOWED_MODES = ['w', 'a']
    assert mode in ALLOWED_MODES, f"The specified mode '{mode}' should be one of {ALLOWED_MODES}."
    data = pd.DataFrame.from_dict(samples)
    data.to_csv(os.path.join(FOLDER_ROOT, filename), header=(mode == 'w'), mode=mode)
    print(f"{'Wrote' if mode == 'w' else 'Appended'} samples in {filename}.")

def read_samples(filename: str) -> dict:
    """Return the sample values found in the file previously generated."""
    
    samples = {}
    filename = os.path.join(FOLDER_ROOT, filename)
    try:
        data = pd.read_csv(filename, index_col=0)
        samples = data.to_dict()
    except FileNotFoundError:
        print(f'"{filename}" not found.')
        return
    else:
        print(f'Found samples in {filename}.')
        return samples

def write_ansys_script(samples: dict, filename: str) -> None:
    """Write a text file containing ANSYS commands used to automate FEA and generate stress contour images."""

    number_samples = get_sample_size(samples)
    # Read the template script.
    with open(os.path.join(FOLDER_ROOT, 'ansys_template.lgw'), 'r') as file:
        lines = file.readlines()
    
    # Replace placeholder lines in the template script.
    with open(os.path.join(FOLDER_ROOT, filename), 'w') as file:
        # Initialize dictionary of placeholder strings (keys) and strings they should be replaced with (values).
        placeholder_substitutions = {}
        # Define names of variables.
        loop_variable = 'i'
        samples_variable = 'samples'
        # Add commands that define the array containing generated samples.
        commands_define_samples = [f'*DIM,{samples_variable},ARRAY,{9},{number_samples}\n']
        for i in range(number_samples):
            commands_define_samples.append(
                f'{samples_variable}(1,{samples[KEY_SAMPLE_NUMBER][i]}) = {samples[load.name][i]},{samples[KEY_X_LOAD][i]},{samples[KEY_Y_LOAD][i]},{samples[angle.name][i]},{samples[length.name][i]},{samples[height.name][i]},{samples[elastic_modulus.name][i]},{samples[KEY_IMAGE_LENGTH][i]},{samples[KEY_IMAGE_HEIGHT][i]}\n'
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

def get_stratified_samples(samples: dict, folder: str, bins: int, desired_subset_size: int) -> dict:
    """Return a subset of the given samples in which the same number of maximum stress values exists in each bin."""

    # Get the maximum stress values in each label.
    stresses, displacements = read_labels(folder)
    stresses = np.array([np.max(stress) for stress in stresses])
    actual_raw_size = len(stresses)

    # Calculate the histogram.
    maximum_stress = np.max(stresses)
    histogram_range = (0, maximum_stress)  # Set minimum to 0 prevent small stresses being excluded
    frequencies, bin_edges = np.histogram(stresses, bins=bins, range=histogram_range)
    minimum_frequency = np.min(frequencies)
    minimum_required_frequency = math.ceil(desired_subset_size / bins)
    actual_subset_size = minimum_frequency * bins
    recommended_raw_size = actual_raw_size * minimum_required_frequency / minimum_frequency
    
    if actual_subset_size < desired_subset_size:
        plt.figure()
        plt.hist(stresses, bins=bins, range=histogram_range, rwidth=0.95, color=BLUE)
        plt.plot((0, maximum_stress), (minimum_required_frequency,)*2, 'k--')
        plt.xticks(bin_edges, rotation=90, fontsize=8)
        index = np.nonzero(frequencies == minimum_frequency)[0][0]
        plt.annotate(f"{minimum_frequency}", (np.mean(bin_edges[index:index+2]), minimum_frequency), color=RED, fontweight='bold', horizontalalignment='center')
        plt.title(f"Subset contains {actual_subset_size} out of desired {desired_subset_size}, dataset of {actual_raw_size} should be around {recommended_raw_size:.0f}", fontsize=8)
        plt.legend([f"{minimum_required_frequency} samples required in each bin"])
        plt.show()

    # Verify that there are enough samples to create a dataset of the desired size.
    assert actual_subset_size >= desired_subset_size, f"The subset only contains {actual_subset_size} out of the desired {desired_subset_size}. The raw dataset of {actual_raw_size} samples should be around {recommended_raw_size:.0f}."

    sample_indices = np.empty(0, dtype=int)
    for i, bin_edge in enumerate(bin_edges[1:], 1):
        # Indices of values that fall inside current bin.
        indices = np.nonzero((bin_edges[i-1] < stresses) & (stresses <= bin_edge))[0]
        # Select the first n values only.
        indices = indices[:minimum_required_frequency]
        sample_indices = np.append(sample_indices, indices)
    np.random.shuffle(sample_indices)
    stratified_samples = {key: [value[i] for i in sample_indices] for key, value in samples.items()}

    return stratified_samples


DATASET = 'test'
NUMBER_SAMPLES = 10
START_SAMPLE_NUMBER = 1
WRITE_MODE = 'w'

# if __name__ == "__main__":
#     # a = sorted(generate_logspace_values(1000, (2, 4), 0.1, 2, 1, True))

#     samples = read_samples(FILENAME_SAMPLES_TEST)
#     stratified_samples = get_stratified_samples(samples, 'Cantilever/Train Labels', bins=10, desired_subset_size=1000)

if __name__ == '__main__':
    DATASET_TYPES = ['train', 'test']
    filename_sample = dict(zip(DATASET_TYPES, [FILENAME_SAMPLES_TRAIN, FILENAME_SAMPLES_TEST]))[DATASET]
    filename_ansys = dict(zip(DATASET_TYPES, ['ansys_script_train.lgw', 'ansys_script_test.lgw']))[DATASET]

    assert DATASET in DATASET_TYPES
    assert not (START_SAMPLE_NUMBER > 1 and WRITE_MODE == 'w'), f"The starting sample number {START_SAMPLE_NUMBER} must be 1 when generating a new dataset."

    # Try to read sample values from the text file if it already exists. If not, generate the samples.
    samples = read_samples(filename_sample)
    if samples:
        overwrite_existing_samples = input(f"Overwrite {filename_sample}? [y/n] ") == "y"
    else:
        overwrite_existing_samples = False
    if not samples or overwrite_existing_samples:
        samples = generate_samples(NUMBER_SAMPLES, start=START_SAMPLE_NUMBER)

    write_samples(samples, filename_sample, mode=WRITE_MODE)
    write_ansys_script(samples, filename_ansys)