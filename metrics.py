"""Evaluation metrics."""


from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np


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
        plt.plot(bin_edges[1:], cdf_predicted, "-", label="Predicted")
        plt.plot(bin_edges[1:], cdf_true, ":", label="True")
        plt.legend()
        plt.grid(visible=True, axis="y")
        # plt.title(f"{area_difference:0.2f}", fontsize=10, fontweight="bold")
        plt.show()

    return cdf_predicted, cdf_true, bin_edges, area_difference

def maximum_value(predicted: np.ndarray, true: np.ndarray, plot=False) -> Tuple[float, float]:
    """Plot and return the maximum values for each subarray along the first dimension in both inputs."""
    max_predicted = np.max(predicted, axis=tuple(range(1, predicted.ndim)))
    max_true = np.max(true, axis=tuple(range(1, true.ndim)))
    # Sort both arrays together sorting by true values.
    max_predicted, max_true = zip(*sorted(zip(max_predicted, max_true), key=lambda _: _[1]))
    max_predicted, max_true = np.array(max_predicted), np.array(max_true)

    if plot:
        plt.figure()
        plt.bar(range(len(max_true)), max_true, color=[0.75]*3, alpha=1.0, width=1.0, label="True")
        plt.bar(range(len(max_predicted)), max_predicted, alpha=0.5, width=1.0, label="Predicted")
        plt.xticks([])
        plt.ylabel('Stress [Pa]')
        plt.legend()
        plt.title("Maxima")
        plt.show()
    
    return max_predicted, max_true

def mean_error(predicted: np.ndarray, true: np.ndarray) -> float:
    return np.mean(predicted - true)

def mean_absolute_error(predicted: np.ndarray, true: np.ndarray) -> float:
    return np.mean(np.abs(predicted - true))

def normalized_mean_absolute_error(predicted: np.ndarray, true: np.ndarray) -> float:
    return np.mean(np.abs(predicted - true)) / np.mean(true)

def mean_squared_error(predicted: np.ndarray, true: np.ndarray) -> float:
    return np.mean((predicted - true) ** 2)

def normalized_mean_squared_error(predicted: np.ndarray, true: np.ndarray) -> float:
    return np.mean((predicted - true) ** 2) / np.mean(true) ** 2

def root_mean_squared_error(predicted: np.ndarray, true: np.ndarray) -> float:
    return np.sqrt(np.mean((predicted - true) ** 2))

def normalized_root_mean_squared_error(predicted: np.ndarray, true: np.ndarray) -> float:
    return np.sqrt(np.mean((predicted - true) ** 2)) / np.mean(true)

def mean_relative_error(predicted: np.ndarray, true: np.ndarray) -> float:
    """Return the mean relative error as a percentage."""
    # Smoothing term to avoid division by zero.
    EPSILON = 0.01
    return np.mean(np.abs(predicted - true) / (EPSILON + true)) * 100

def get_maxima_indices(data: np.ndarray) -> np.ndarray:
    """Return a Boolean array containing the locations of the maxima for each data, assuming the data dimension is the first dimension."""
    maxima = data.max(axis=tuple(range(1, data.ndim)), keepdims=True)
    return data == maxima