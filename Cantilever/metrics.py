from matplotlib import pyplot as plt
import numpy as np
from scipy import stats


def evaluate(network, fea):
    """Return a dictionary containing the names (keys) and results (values) of multiple evaluation metrics."""
    
    results = {}

    # Area metric.
    NUMBER_BINS = 100
    histogram_network = np.histogram(network.flatten(), bins=NUMBER_BINS)[0]
    histogram_fea = np.histogram(fea.flatten(), bins=NUMBER_BINS)[0]
    histogram_network = histogram_network / np.sum(histogram_network)
    histogram_fea = histogram_fea / np.sum(histogram_fea)
    cdf_network = np.cumsum(histogram_network)
    cdf_fea = np.cumsum(histogram_fea)
    statistic = np.sum(cdf_network) - np.sum(cdf_fea)
    # statistic = np.max(np.abs(cdf_network - cdf_fea))
    results['Area Metric'] = statistic
    
    # K-S test.
    statistic, pvalue = stats.kstest(network.flatten(), fea.flatten())
    results['K-S Test'] = statistic

    # Maximum stress value.
    network_max, fea_max = np.max(network.flatten()), np.max(fea.flatten())
    results['Max. Value'] = (network_max, fea_max)

    return results

def area_metric(network: np.ndarray, label: np.ndarray, max_value):
    """Return the CDF of the inputs and the difference between their areas under the CDF."""
    NUMBER_BINS = 1000
    # Specify the range of possible values for the histogram to use for both network and label data.
    value_range = (0, max_value)

    histogram_network, bins = np.histogram(network.flatten(), bins=NUMBER_BINS, range=value_range)
    histogram_label, _ = np.histogram(label.flatten(), bins=NUMBER_BINS, range=value_range)
    histogram_network = histogram_network / np.sum(histogram_network)
    histogram_label = histogram_label / np.sum(histogram_label)
    cdf_network = np.cumsum(histogram_network)
    cdf_label = np.cumsum(histogram_label)
    area_network = np.sum(cdf_network * np.diff(bins))
    area_label = np.sum(cdf_label * np.diff(bins))
    area_difference = area_network - area_label

    return cdf_network, cdf_label, bins, area_difference

def mean_error(network: np.ndarray, label: np.ndarray):
    """Return the mean difference between the inputs. Different from mean squared error."""
    me = np.mean(network - label)
    return me