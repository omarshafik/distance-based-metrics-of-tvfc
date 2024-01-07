"""
function definitions to apply statistical tests on computed estimates
"""
import numpy as np
from scipy import stats
import statsmodels.api as sm
from joblib import Parallel, delayed
from tools.analyze import swd
import tools.common as common

def test_stationary(
    timeseries,
    sample_size: int,
    test = sm.tsa.adfuller,
    alpha: float = 0.05,
    **kwargs) -> float:
    """ compute percentage of samples that satistify the stationarity \
        (identical distribution) assumption of central limit theorem 

    Args:
        timeseries (any): array of pointwise TVC estimates
        sample_size (int): sample size to apply the test on
        test (callable, optional): test function to use. Defaults to sm.tsa.adfuller.
        alpha (float, optional): alpha (critical p) value to use. Defaults to 0.05.

    Returns:
        float: percentage of samples that satisfy the identical distribution assumption 
    """
    pvalues = np.squeeze(Parallel()(
        delayed(test)(
            timeseries[idx:idx+sample_size], **kwargs
        ) for idx in range(len(timeseries) - sample_size - 1)))[:,1]
    return np.sum(pvalues <= alpha) / len(pvalues)

def test_independence(
    timeseries,
    sample_size: int,
    test = sm.stats.acorr_ljungbox,
    alpha: float = 0.05,
    **kwargs) -> float:
    """ compute percentage of samples that satistify the autocorrelation \
        (independence) assumption of central limit theorem 

    Args:
        timeseries (any): array of pointwise TVC estimates
        sample_size (int): sample size to apply the test on
        test (callable, optional): test function to use. Defaults to sm.tsa.adfuller.
        alpha (float, optional): alpha (critical p) value to use. Defaults to 0.05.

    Returns:
        float: percentage of samples that satisfy the independence assumption 
    """
    lb_pvalues = np.squeeze(Parallel()(
        delayed(test)(
                timeseries[idx:idx+sample_size], **kwargs
        ) for idx in range(len(timeseries) - sample_size - 1)))[:,1]
    return np.sum(lb_pvalues > alpha) / len(lb_pvalues)

def test_distribution(data) -> tuple:
    """ compute skewness and kurtosis of the data distribution

    Args:
        data (any): data to compute statistics for

    Returns:
        tuple: tuple of skewness and kurtosis
    """
    return (stats.skew(data.flatten()), stats.kurtosis(data.flatten()))

def test_identical_distribution(timeseries: np.ndarray, nrefs: int = 30) -> int:
    """ Apply Kolo

    Args:
        timeseries (ndarray): time series array of estimates
        nrefs (int): number of reference edges (pairs estimates). \
            These are selected randomly from the time series array. 

    Returns:
        int: percentage of edges that have identical distributions
    """
    ks_pvalues = []
    sample_ref_edges = timeseries[np.random.choice(timeseries.shape[0], size=nrefs, replace=False)]
    for refnode in sample_ref_edges:
        ks_pvalues.extend(np.squeeze(Parallel()(
            delayed(stats.kstest)(
                    node,
                    refnode
            ) for node in timeseries))[:, 1])
    ks_pvalues = np.array(ks_pvalues)
    return np.sum(ks_pvalues < 0.05) / len(ks_pvalues)

def get_edges_of_interest(
    empirical_timeseries: np.ndarray,
    surrogate_timeseries: np.ndarray,
    pairs: np.ndarray,
    window_size: int = None,
    alpha: float = 0.05,
    h1: bool = False,
    h2: bool = False
) -> int:
    """get edges that exhibit properties beyond given surrogate.

    Args:
        empirical_timeseries (np.ndarray): time series of empirical data
        surrogate_timeseries (np.ndarray): time series of surrogate data
        pairs (np.ndarray): order of pairing for empirical time series. \
            An array of region indices used to compute and return statistics.
        window_size (int, optional): Window/sample size to use. Defaults to None.\
            This must be given, when variance test is needed
        alpha (float, optional): significance level to use. Defaults to 0.05.
        h1 (bool, optional): Test the time-average estimate null hypothesis (H1)
        h2 (bool, optional): Test the edge variance null hypothesis (H2)
    """
    num_nodes = empirical_timeseries.shape[0]
    edges_of_interest = np.zeros(int((num_nodes * (num_nodes - 1)) / 2), dtype=int)
    if h1:
        estimates_empirical = swd(
            empirical_timeseries, window_size=empirical_timeseries.shape[-1], pairs=pairs)
        time_avg_estimates_surrogate = swd(
            surrogate_timeseries, window_size=empirical_timeseries.shape[-1])

        time_avg_lower_bound, time_avg_higher_bound = stats.norm.interval(
            (1 - alpha),
            loc=np.mean(time_avg_estimates_surrogate),
            scale=np.std(time_avg_estimates_surrogate))
        edges_of_interest = np.where(estimates_empirical < time_avg_lower_bound, 1, 0)
        edges_of_interest += np.where(estimates_empirical > time_avg_higher_bound, 1, 0)

    if h2 and window_size is not None:
        estimates_empirical = swd(
            empirical_timeseries, window_size=window_size, pairs=pairs)
        estimates_surrogate = swd(
            surrogate_timeseries, window_size=window_size)

        edge_variance_empirical = np.var(estimates_empirical, -1)
        surrogate_variance = np.var(estimates_surrogate, -1)
        sorted_surrogate_variance = np.sort(surrogate_variance)
        percentile_index = int((1 - alpha) * len(sorted_surrogate_variance))
        variance_higher_bound = sorted_surrogate_variance[percentile_index]
        edges_of_interest += np.where(
            edge_variance_empirical > variance_higher_bound, 1, 0)

    edges_of_interest[edges_of_interest != 0] = 1
    return edges_of_interest

def significant_estimates(
    estimates: np.ndarray,
    alpha: float = 0.05) -> np.ndarray:
    """ mark estimates that have values greater/lower than a given percentile bound
    (default is 0.95).

    Args:
        estimates (np.ndarray): array of computed estimates. 
            assumes that estimates are normally distributed
        alpha (float, optional): Significance level. Defaults to 0.05.
    Returns:
        list: significance array with equal size to given estimates array
    """
    lower_bound, upper_bound = stats.norm.interval(
        (1 - alpha),
        loc=np.mean(estimates),
        scale=np.std(estimates)
    )
    estimates_significance = np.where(
        estimates > upper_bound, 1, 0)
    estimates_significance += np.where(
        estimates < lower_bound, -1, 0)
    return estimates_significance

def significant_time_points(
    significance_array: np.ndarray,
    window_size: int) -> np.ndarray:
    """ get time points of the original timeseries \
        that belongs to samples with significant SWD values

    Args:
        uncertainty (np.ndarray): array of uncertainty values
        window_size (int): window size used for computing the estimates
        alpha (float, optional): p-value to use. Defaults to 0.05.

    Returns:
        np.array: array that marks time points belonging to (in)significant samples \
        -1 for time points belonging to significantly low SWD samples. \
         1 for time points belonging to significantly high SWD samples. \
         0 for time points belonging to insignificant samples
    """
    pos_significance_array = np.where(significance_array > 0, 1, 0)
    neg_significance_array = np.where(significance_array < 0, -1, 0)
    half_window = (window_size + 1) // 2
    pos_significance_array = common.sliding_average(pos_significance_array, half_window)
    pos_significance_array[pos_significance_array > 0] = 1
    neg_significance_array = common.sliding_average(neg_significance_array, half_window)
    neg_significance_array[neg_significance_array < 0] = -1
    result_arr = pos_significance_array + neg_significance_array
    pad_size = (window_size - 1) // 2
    extra_pad = (window_size - 1) % 2
    pad_width = [(0, 0)] * result_arr.ndim
    pad_width[-1] = (pad_size + extra_pad, pad_size)
    result_arr = np.pad(result_arr,
                        pad_width,
                        mode='constant',
                        constant_values=0)
    return result_arr
