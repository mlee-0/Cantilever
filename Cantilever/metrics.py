from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np

from setup import Colors


def area_metric(predicted: np.ndarray, true: np.ndarray, max_value, plot=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Plot and return the CDFs of the inputs and the difference between the areas under their CDFs."""
    
    NUMBER_BINS = 1000
    # Specify the range of possible values for the histogram to use for both predicted and true data.
    value_range = (0, max_value)

    histogram_predicted, bin_edges = np.histogram(predicted.flatten(), bins=NUMBER_BINS, range=value_range)
    histogram_true, _ = np.histogram(true.flatten(), bins=NUMBER_BINS, range=value_range)
    histogram_predicted = histogram_predicted / np.sum(histogram_predicted)
    histogram_true = histogram_true / np.sum(histogram_true)
    cdf_predicted = np.cumsum(histogram_predicted)
    cdf_true = np.cumsum(histogram_true)
    area_predicted = np.sum(cdf_predicted * np.diff(bin_edges))
    area_true = np.sum(cdf_true * np.diff(bin_edges))
    area_difference = area_predicted - area_true

    if plot:
        plt.figure()
        plt.plot(bin_edges[1:], cdf_predicted, "-", color=Colors.BLUE)
        plt.plot(bin_edges[1:], cdf_true, ":", color=Colors.RED)
        plt.legend(["CNN", "FEA"])
        plt.grid(visible=True, axis="y")
        plt.xticks([*plt.xticks()[0], max_value])
        plt.title(f"{area_difference:0.2f}", fontsize=10, fontweight="bold")
        plt.show()

    return cdf_predicted, cdf_true, bin_edges, area_difference

def maximum_value(predicted: np.ndarray, true: np.ndarray, plot=False) -> Tuple[float, float]:
    """Plot and return the maximum values along the first dimension in both inputs."""
    max_predicted = np.max(predicted, axis=(1, 2, 3))
    max_true = np.max(true, axis=(1, 2, 3))

    if plot:
        plt.figure()
        plt.plot(max_true, 'o', color=Colors.RED, label="True")
        plt.plot(max_predicted, '.', color=Colors.BLUE, label="Predicted")
        plt.legend()
    
    return max_predicted, max_true

def mean_error(predicted: np.ndarray, true: np.ndarray) -> float:
    return np.mean(predicted - true)

def mean_absolute_error(predicted: np.ndarray, true: np.ndarray) -> float:
    return np.mean(np.abs(predicted - true))

def mean_squared_error(predicted: np.ndarray, true: np.ndarray) -> float:
    return np.mean((predicted - true) ** 2)

def root_mean_squared_error(predicted: np.ndarray, true: np.ndarray) -> float:
    return np.sqrt(np.mean((predicted - true) ** 2))

def mean_relative_error(predicted: np.ndarray, true: np.ndarray) -> float:
    """Return the mean relative error as a percentage."""
    # Smoothing term to avoid division by zero.
    EPSILON = 0.01
    return np.mean(np.abs(predicted - true) / (EPSILON + np.maximum(predicted, true))) * 100