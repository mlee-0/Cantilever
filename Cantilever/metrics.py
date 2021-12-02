import numpy as np
from scipy import stats


def area_metric(network, fea):
    NUMBER_BINS = 100
    histogram_network = np.histogram(network.flatten(), bins=NUMBER_BINS)[0]
    histogram_fea = np.histogram(fea.flatten(), bins=NUMBER_BINS)[0]
    histogram_network = histogram_network / np.sum(histogram_network)
    histogram_fea = histogram_fea / np.sum(histogram_fea)
    cdf_network = np.cumsum(histogram_network)
    cdf_fea = np.cumsum(histogram_fea)
    statistic = np.max(np.abs(cdf_network - cdf_fea))
    return statistic

def ks_test(network, fea):
    statistic, pvalue = stats.kstest(network.flatten(), fea.flatten())
    return statistic, pvalue

def max_value(network, fea):
    network_max, fea_max = np.max(network.flatten()), np.max(fea.flatten())
    return network_max, fea_max