"""
Use optimization to find a power that transforms the dataset to the desired target distribution.
"""


from typing import *

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, stats

from helpers import read_pickle


def statistical_moments(data: np.ndarray) -> Tuple[float, float, float, float]:
    return np.array([data.mean(), data.std(), stats.skew(data, axis=None), stats.kurtosis(data, axis=None)])

def cost_statistical_moments(exponent: float, data: np.ndarray, target_moments: Tuple[float, float, float, float]):
    transformed = data ** exponent
    moments = statistical_moments(transformed)
    return np.sum((moments - target_moments) ** 2)

def cost_histogram_mae(exponent: float, data: np.ndarray, target_histogram: np.ndarray):
    """MAE of the corresponding bins."""
    transformed = data ** exponent
    histogram, _ = np.histogram(transformed, bins=bins)
    mae = np.mean(np.abs(histogram - target_histogram))
    return mae

def cost_histogram_mse(exponent: float, data: np.ndarray, target_histogram: np.ndarray):
    """MAE of the corresponding bins."""
    transformed = data ** exponent
    histogram, _ = np.histogram(transformed, bins=bins)
    mae = np.mean((histogram - target_histogram) ** 2)
    return mae


if __name__ == '__main__':
    bins = 100
    initial_guess = 1/2
    objective = cost_histogram_mse

    data = read_pickle('Labels 2D/labels.pickle')
    # Calculate the target distribution as the histogram of the given data in which each individual sample is normalized to [0, 1].
    data_normalized = data / data.max(axis=tuple(range(1, data.ndim)), keepdims=True)
    target_histogram, _ = np.histogram(data_normalized, bins=bins)
    target_moments = statistical_moments(data_normalized)

    if objective is cost_statistical_moments:
        args = (data, target_moments)
    elif objective in [cost_histogram_mae, cost_histogram_mse]:
        args = (data, target_histogram)

    result = optimize.minimize_scalar(objective, bounds=(0, 1), method="bounded", args=args)
    exponent = result.x
    print(f"Optimum exponent: {exponent:.2f} = 1/{1/exponent:.2f}")

    plt.figure()
    plt.hist(data_normalized[data_normalized > 0].flatten(), bins=bins, alpha=0.5, label='Target')
    transformed = (data/data.max()) ** exponent
    plt.hist(transformed[transformed > 0].flatten() / transformed.max(), bins=bins, alpha=0.5, label=f'{exponent}')
    plt.legend()
    plt.show()