"""
Run this script to perform stratified sampling and write text files containing subsets of a dataset.
"""

import math
import os
import pickle

import numpy as np
import pandas as pd

from datasets import FILENAME_SAMPLES, read_samples
from setup import *


def stratify_samples(folder: str, filename: str, subset_size: int, bins: int, nonuniformity: float = 1.0) -> None:
    """
    Write a text file containing indices of the given samples that form a subset in which the same number of maximum values exists in each bin. For a given dataset, the same samples will be included in the subset because the first n samples are selected from each histogram bin rather than being randomly selected. The order of the samples in the subset is randomized.

    `samples`: DataFrame of samples of entire dataset.
    `folder`: Folder in which labels are read.
    `subset_size`: The number of samples to put in the subset. The actual subset size may not exactly match this number.
    `bins`: The number of bins to use in the histogram of maximum values.
    `nonuniformity`: How much larger than the smallest bin the largest bin is. For example, a value of 1 results in a uniform distribution, in which the largest bin has as many samples as the smallest bin. A value of 2 results in the largest bin having twice as many samples as the smallest bin.
    """

    # Load the label images.
    files = glob.glob(os.path.join(folder, "*.pickle"))
    assert len(files), f"A .pickle file must exist in {folder}."
    file = files[0]
    labels = read_pickle(file)

    # Get the maximum values in each label.
    maxima = np.array([np.max(_) for _ in labels])
    actual_raw_size = len(maxima)

    # Calculate the histogram.
    histogram_range = (0, np.max(maxima))  # Set minimum to 0 prevent small stresses being excluded
    frequencies, bin_edges = np.histogram(maxima, bins=bins, range=histogram_range)
    minimum_frequency = np.min(frequencies)
    minimum_bin = np.argmin(frequencies)
    
    assert nonuniformity > 0, f"The nonuniformity value {nonuniformity} should be positive."
    if nonuniformity == 1.0:
        required_frequencies = np.full(bins, math.ceil(subset_size / bins))
    else:
        required_frequencies = frequencies / np.min(frequencies)
        required_frequencies = np.power(
            required_frequencies,
            np.log(nonuniformity) / np.log(np.max(required_frequencies))
        )
        required_frequencies *= subset_size / np.sum(required_frequencies)
        required_frequencies = np.round(required_frequencies).astype(int)
    actual_subset_size = np.sum(required_frequencies) if minimum_frequency >= required_frequencies[minimum_bin] else np.sum(required_frequencies) * (minimum_frequency / required_frequencies[minimum_bin])
    recommended_raw_size = actual_raw_size * required_frequencies[minimum_bin] / minimum_frequency
    
    if actual_subset_size < subset_size:
        plt.figure()
        plt.hist(maxima, bins=bins, range=histogram_range, rwidth=0.95, color=Colors.BLUE)
        plt.plot(
            [bin_edges[:-1], bin_edges[1:]],
            [required_frequencies, required_frequencies],
            'k--'
        )
        plt.annotate(f"{minimum_frequency}", (np.mean(bin_edges[minimum_bin:minimum_bin+2]), minimum_frequency), color=Colors.RED, fontweight='bold', horizontalalignment='center')
        plt.xticks(bin_edges, rotation=90, fontsize=6)
        plt.xlabel("Stress")
        plt.title(f"Subset contains {actual_subset_size} out of desired {subset_size}, dataset of {actual_raw_size} should be around {recommended_raw_size:.0f}", fontsize=10)
        plt.legend([f"Samples required in each bin"])
        plt.show()

    # Verify that there are enough samples to create a dataset of the desired size.
    print(f"The subset contains {actual_subset_size} out of the desired {subset_size}.")
    assert actual_subset_size >= subset_size, f"The raw dataset of {actual_raw_size} samples should be around {recommended_raw_size:.0f}."

    # Create the subset.
    sample_indices = np.empty(0, dtype=int)
    for i, f in enumerate(required_frequencies):
        # Indices of values that fall inside current bin.
        indices = np.nonzero((bin_edges[i] < maxima) & (maxima <= bin_edges[i+1]))[0]
        # Select the first f values only.
        indices = indices[:f]
        sample_indices = np.append(sample_indices, indices)
    np.random.shuffle(sample_indices)

    # Write the sample indices to a text file.
    with open(filename, "w") as f:
        f.writelines([str(_) for _ in sample_indices])


if __name__ == "__main__":
    samples = read_samples(os.path.join(FOLDER_ROOT, FILENAME_SAMPLES))
    filename = "subset.txt"

    stratified_samples = stratify_samples(samples, "Cantilever/Labels", filename, subset_size=1000, bins=15, nonuniformity=1)