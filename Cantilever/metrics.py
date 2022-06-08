from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np

from setup import Colors


def area_metric(network: np.ndarray, label: np.ndarray, max_value, plot=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Plot and return the CDFs of the inputs and the difference between their areas under the CDF."""
    
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

    if plot:
        plt.figure()
        plt.plot(bin_edges[1:], cdf_network, "-", color=Colors.BLUE)
        plt.plot(bin_edges[1:], cdf_label, ":", color=Colors.RED)
        plt.legend(["CNN", "FEA"])
        plt.grid(visible=True, axis="y")
        plt.xticks([*plt.xticks()[0], max_value])
        plt.title(f"{area_difference:0.2f}", fontsize=10, fontweight="bold")
        plt.show()

    return cdf_network, cdf_label, bin_edges, area_difference

def maximum_value(network: np.ndarray, label: np.ndarray, plot=False) -> Tuple[float, float]:
    """Plot and return the maxima along the first dimension in both inputs."""
    max_network = np.max(network, axis=0)
    max_label = np.max(network, axis=0)

    if plot:
        plt.figure()
        plt.plot(max_label, 'o', color=Colors.RED, label="True")
        plt.plot(max_network, '.', color=Colors.BLUE, label="Predicted")
        plt.legend()
    
    return max_network, max_label

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