"""Use optimization to find a transformation that transforms the dataset to the desired target distribution."""


from typing import *

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, stats

from preprocessing import read_pickle


def transform_exponentiation(data, x: np.ndarray):
    return data ** x[0]

def transform_logarithm(data: np.ndarray, x: np.ndarray):
    """Scale data to range [x, x+1], take the natural logarithm, and scale to [0, 1]."""
    data = data - data.min()
    data = data / data.max()
    data = data * (1) + x
    transformed = np.log(data)
    transformed -= transformed.min()
    transformed /= transformed.max()
    return transformed

def statistical_moments(data: np.ndarray) -> Tuple[float, float, float, float]:
    return np.array([data.mean(), data.std(), stats.skew(data, axis=None), stats.kurtosis(data, axis=None)])

def cost_statistical_moments(x: np.ndarray, data: np.ndarray, target_moments: Tuple[float, float, float, float]):
    transformed = transformation(data, x)
    moments = statistical_moments(transformed)
    return np.sum((moments - target_moments) ** 2)

def cost_histogram_mae(x: np.ndarray, transformation: Callable, data: np.ndarray, target_histogram: np.ndarray):
    """MAE of the corresponding bins."""
    transformed = transformation(data, x)
    histogram, _ = np.histogram(transformed, bins=target_histogram.size)
    mae = np.mean(np.abs(histogram - target_histogram))
    return mae

def cost_histogram_mse(x: np.ndarray, transformation: Callable, data: np.ndarray, target_histogram: np.ndarray):
    """MAE of the corresponding bins."""
    transformed = transformation(data, x)
    histogram, _ = np.histogram(transformed, bins=target_histogram.size)
    mae = np.mean((histogram - target_histogram) ** 2)
    return mae

def cost_kl_divergence(x: np.ndarray, transformation: Callable, data: np.ndarray, target_histogram: np.ndarray):
    """KL-diverence of PDFs."""
    transformed = transformation(data, x)
    histogram, _ = np.histogram(transformed, bins=target_histogram.size)
    divergence = stats.entropy(histogram, target_histogram)
    return divergence


if __name__ == '__main__':
    bins = 100
    initial_guess = 1e-5  #1/2
    transformation = transform_logarithm
    objective = cost_kl_divergence

    data = read_pickle('Labels 2D/labels.pickle')
    # Calculate the target distribution as the histogram of the given data in which each individual sample is normalized to [0, 1].
    data_normalized = data / data.max(axis=tuple(range(1, data.ndim)), keepdims=True)
    target_histogram, _ = np.histogram(data_normalized, bins=bins)
    target_moments = statistical_moments(data_normalized)

    if objective is cost_statistical_moments:
        args = (transformation, data, target_moments)
    elif objective in [cost_histogram_mae, cost_histogram_mse, cost_kl_divergence]:
        args = (transformation, data, target_histogram)

    optimum = optimize.fmin(objective, initial_guess, args=args)
    print(f"Optimum found: {optimum}")

    plt.figure(figsize=(4, 3))
    plt.hist(data_normalized[data_normalized > 0].flatten(), bins=bins, alpha=0.5, label='Target')
    transformed = transformation(data, optimum)
    plt.hist(transformed[transformed > 0].flatten() / transformed.max(), bins=bins, alpha=0.5, label='Transformed')
    plt.yticks([])
    plt.legend()
    plt.show()