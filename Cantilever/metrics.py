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
    statistic = np.max(np.abs(cdf_network - cdf_fea))
    results['Area Metric'] = statistic
    
    # K-S test.
    statistic, pvalue = stats.kstest(network.flatten(), fea.flatten())
    results['K-S Test'] = statistic

    # Maximum stress value.
    network_max, fea_max = np.max(network.flatten()), np.max(fea.flatten())
    results['Max. Stress'] = (network_max, fea_max)

    return results