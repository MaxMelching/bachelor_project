# -----------------------------------------------------------------------------
#
# This script contains criteria for the analysis of probability distributions
#
# Author: Max Melching
# Source: https://github.com/MaxMelching/bachelor_project
# 
# -----------------------------------------------------------------------------



import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy



def jsd(data1: list, data2: list, binnumber: any = None) -> float:
    """
    Computes the Jensen-Shannon divergence of two samples.

    Parameters:
        - data1 (numpy-array): first sample.
        - data2 (numpy-array): second sample.
        - binnumber (any, optional, default = 'default'): can be any
          argument accepted as binnumber by the numpy or matplotlib
          functions to compute histograms. For 'default', a number
          like int((4 * samplesize) ** (1 / 3)) is used.
    
    Returns:
        - div (float): Jensen-Shannon divergence of data1 and data2.
    """

    # Store lengths
    len1, len2 = data1.size, data2.size
    
    # Find intervals of data1, data2 and add them together in limits
    limits1 = np.array([data1.min(), data1.max()])
    limits2 = np.array([data2.min(), data2.max()])

    limits = (min(limits1[0], limits2[0]), max(limits1[1], limits2[1]))
    
    # Determine binnumber for each case, choose minimum
    if binnumber is None:
        binnumber = int(min(
                        (limits[1] - limits[0]) / (limits1[1] - limits1[0])
                        * (4 * len1) ** (1 / 3), 
                        (limits[1] - limits[0]) / (limits2[1] - limits2[0])
                        * (4 * len2) ** (1 / 3)
                        ))
        
    # Compute histograms as estimates of probability density function
    # Note: no normalization here (would be "density = True") because
    # jensenshannon does this automatically.
    hist1, bins1 = np.histogram(data1, bins=binnumber, range=limits)
    hist2 = np.histogram(data2, bins=bins1)[0]
    
    # Use scipy function for Jensen-Shannon distance, which is square
    # root of Jensen-Shannon divergence
    # Note: it makes the input sum to 1, which is not necessarily the
    # case for output of hist (normalizes such that integral is 1).
    div = jensenshannon(hist1, hist2, base=2) ** 2
    
    return div


def kld(data1: list, data2: list, binnumber: any = None) -> float:
    """
    Computes the Kullback-Leibler divergence of two samples.

    Parameters:
        - data1 (numpy-array): first sample.
        - data2 (numpy-array): second sample.
        - binnumber (any, optional, default = 'default'): can be any
          argument accepted as binnumber by the numpy or matplotlib
          functions to compute histograms. For 'default', a number
          like int((4 * samplesize) ** (1 / 3)) is used.
    
    Returns:
        - div (float): Kullback-Leibler divergence of data1 and data2.
    """

    # Store lengths
    len1, len2 = data1.size, data2.size
    
    # Find intervals of data1, data2 and add them together in limits
    limits1 = np.array([data1.min(), data1.max()])
    limits2 = np.array([data2.min(), data2.max()])

    limits = (min(limits1[0], limits2[0]), max(limits1[1], limits2[1]))
    
    # Determine binnumber for each case, choose minimum
    if binnumber is None:
        binnumber = int(min(
                        (limits[1] - limits[0]) / (limits1[1] - limits1[0])
                        * (4 * len1) ** (1 / 3), 
                        (limits[1] - limits[0]) / (limits2[1] - limits2[0])
                        * (4 * len2) ** (1 / 3)
                        ))
        
    # Compute histograms as estimates of probability density function
    # Note: no normalization here (would be "density = True") because
    # jensenshannon does this automatically.
    hist1, bins1 = np.histogram(data1, bins=binnumber, range=limits)
    hist2 = np.histogram(data2, bins=bins1)[0]
    
    # Use scipy function for entropy, which computes relative entropy
    # (= KLD) for two inputs (no need to use rel_entr, kl_div from
    # scipy.special)
    # Note: it makes the input sum to 1, which is not necessarily the
    # case for output of hist (normalizes such that integral is 1).
    div = entropy(hist1, hist2, base=2)
    
    return div


def mean_criterion(data1: any, data2: any) -> np.array:
    """
    Computes the mean difference in units of the average standard
    deviation of two samples.

    Parameters:
        - data1 (array-like): first sample.
        - data2 (array-like): second sample.
    
    Returns:
        - critval (numpy-array): value of mean criterion for columns
          of data1 and data2.
    """

    critvals = np.abs((np.mean(data1, axis = 0) - np.mean(data2, axis = 0))
                     / (np.std(data1, axis = 0) + np.std(data2, axis = 0)) * 2)

    return critvals


def median_criterion(data1: any, data2: any, cred: float) -> np.array:
    """
    Computes the median difference in units of the average credible
    interval of two samples.

    Parameters:
        - data1 (array-like): first sample.
        - data2 (array-like): second sample.
        - cred (float): percentage of data left/ right of median used
          for the calculation of the credible interval.
    
    Returns:
        - critvals (numpy-array): values of median criterion for
          columns of data1 and data2.
    """

    # Calculate left credible boundary, median, right credible boundary
    credleft1, medians1, credright1 = np.percentile(
        data1,
        [50 - cred, 50, 50 + cred],
        axis = 0
        )
    credleft2, medians2, credright2 = np.percentile(
        data2,
        [50 - cred, 50, 50 + cred],
        axis = 0
        )

    # Calculate credible interval as distance between left/ right boundary and
    # median (subtracting both is ok due to use of abs in next step)
    credleft1 -= medians1
    credright1 -= medians1
    credleft2 -= medians2
    credright2 -= medians2

    # Left/ right credible interval has to be chosen based on relative
    # position of medians for the normalization
    critvals = np.where(medians1 <= medians2,
        np.abs((medians2 - medians1))
        / (np.abs(credleft2) + np.abs(credright1)) * 2,
        np.abs((medians1 - medians2))
        / (np.abs(credleft1) + np.abs(credright2)) * 2)

    return critvals