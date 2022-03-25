from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np
from scipy import stats


def area_metric(network: np.ndarray, label: np.ndarray, max_value) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return the CDF of the inputs and the difference between their areas under the CDF."""
    NUMBER_BINS = 1000
    # Specify the range of possible values for the histogram to use for both network and label data.
    value_range = (0, max_value)

    histogram_network, bin_edges = np.histogram(network.flatten(), bins=NUMBER_BINS, range=value_range)
    histogram_label, _ = np.histogram(label.flatten(), bins=NUMBER_BINS, range=value_range)
    histogram_network = histogram_network / np.sum(histogram_network)
    histogram_label = histogram_label / np.sum(histogram_label)
    cdf_network = np.cumsum(histogram_network)
    cdf_label = np.cumsum(histogram_label)
    area_network = np.sum(cdf_network * np.diff(bin_edges))
    area_label = np.sum(cdf_label * np.diff(bin_edges))
    area_difference = area_network - area_label

    return cdf_network, cdf_label, bin_edges, area_difference

def maximum_value(network: np.ndarray, label: np.ndarray) -> Tuple[float, float]:
    """Return the maxima in both inputs."""
    return np.max(network), np.max(label)

def mean_error(network: np.ndarray, label: np.ndarray) -> float:
    """Return the mean error."""
    return np.mean(network - label)

def mean_absolute_error(network: np.ndarray, label: np.ndarray) -> float:
    """Return the mean absolute error."""
    return np.mean(np.abs(network - label))

def mean_squared_error(network: np.ndarray, label: np.ndarray) -> float:
    return np.mean(np.power(network - label, 2))

def mean_relative_error(network: np.ndarray, label: np.ndarray) -> float:
    """Return the mean relative error as a percentage."""
    # Smoothing term to avoid division by zero.
    EPSILON = 0.01
    return np.mean(np.abs(network - label) / (EPSILON + np.maximum(network, label))) * 100