"""
function definitions to apply statistical tests on computed estimates
"""
import warnings
import numpy as np
from scipy import stats
import statsmodels.api as sm
from joblib import Parallel, delayed
from statsmodels.tools.sm_exceptions import InterpolationWarning
from tools.analyze import swd
import tools.common as common
warnings.simplefilter('ignore', InterpolationWarning)

def test_stationary(
    timeseries,
    test = sm.tsa.kpss,
    alpha: float = 0.05,
    **kwargs) -> float:
    """ compute percentage of stationary edges

    Args:
        timeseries (any): array of tvFC estimates
        test (callable, optional): test function to use. Defaults to sm.tsa.kpss.
        alpha (float, optional): alpha (critical p) value to use. Defaults to 0.05.

    Returns:
        float: percentage of stationary edges
    """
    pvalues = np.squeeze(Parallel()(
        delayed(test)(
            edge, **kwargs
        ) for edge in timeseries))[:,1]
    return np.sum(pvalues > alpha) / len(pvalues)

def test_independence(
    timeseries,
    sample_size: int,
    test = sm.stats.acorr_ljungbox,
    alpha: float = 0.05,
    **kwargs) -> float:
    """ compute percentage of samples that satistify the autocorrelation \
        (independence) assumption of central limit theorem 

    Args:
        timeseries (any): array of tvFC estimates
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
    for refparcel in sample_ref_edges:
        ks_pvalues.extend(np.squeeze(Parallel()(
            delayed(stats.kstest)(
                    parcel,
                    refparcel
            ) for parcel in timeseries))[:, 1])
    ks_pvalues = np.array(ks_pvalues)
    return np.sum(ks_pvalues < 0.05) / len(ks_pvalues)

def get_edges_of_interest(
    empirical_data: np.ndarray,
    null_data: np.ndarray,
    pairs: np.ndarray,
    window_size: int = None,
    alpha: float = 0.0001,
    h1: bool = False,
    h2: bool = False,
    metric: callable = swd
) -> int:
    """get edges that exhibit properties beyond given surrogate.

    Args:
        empirical_data (np.ndarray): empirical data
        null_data (np.ndarray): surrogate data
        pairs (np.ndarray): order of pairing for empirical time series. \
            An array of region indices used to compute and return statistics.
        window_size (int, optional): Window/sample size to use. Defaults to None.\
            This must be given, when variance test is needed
        alpha (float, optional): significance level to use. Defaults to 0.05.
        h1 (bool, optional): Test the time-average estimate null hypothesis (H1)
        h2 (bool, optional): Test the edge variance null hypothesis (H2)
    """
    num_parcels = empirical_data.shape[0]
    edges_of_interest = np.zeros(int((num_parcels * (num_parcels - 1)) / 2), dtype=int)
    if h1:
        time_avg_estimates_empirical = metric(
            empirical_data, window_size=empirical_data.shape[-1], pairs=pairs)
        time_avg_estimates_surrogate = metric(
            null_data, window_size=empirical_data.shape[-1])

        time_avg_lower_bound, time_avg_higher_bound = stats.norm.interval(
            (1 - alpha),
            loc=np.mean(time_avg_estimates_surrogate),
            scale=np.std(time_avg_estimates_surrogate))
        edges_of_interest = np.where(time_avg_estimates_empirical < time_avg_lower_bound, 1, 0)
        edges_of_interest += np.where(time_avg_estimates_empirical > time_avg_higher_bound, 1, 0)

    if h2 and window_size is not None:
        estimates_empirical = metric(
            empirical_data, window_size=window_size, pairs=pairs)
        estimates_surrogate = metric(
            null_data, window_size=window_size)

        edge_variance_empirical = np.var(estimates_empirical, -1)
        surrogate_variance = np.var(estimates_surrogate, -1)
        variance_higher_bound = np.percentile(surrogate_variance, 100 * (1 - alpha))
        edges_of_interest += np.where(
            edge_variance_empirical > variance_higher_bound, 1, 0)

    edges_of_interest[edges_of_interest != 0] = 1
    return edges_of_interest

def significant_estimates(
    estimates: np.ndarray,
    alpha: float = 0.05,
    mean: float = None,
    std: float = None) -> np.ndarray:
    """ mark estimates that have values greater/lower than a given percentile bound
    (default is 0.95).

    Args:
        estimates (np.ndarray): array of computed estimates. 
            assumes that estimates are normally distributed
        alpha (float, optional): Significance level. Defaults to 0.05.
        mean (float, optional): Estimates' mean. Defaults to the mean of given estimates array.
        std (float, optional): Estimates' std. Defaults to the std of given estimates array.
    Returns:
        list: significance array with equal size to given estimates array
    """
    if mean is None:
        mean = np.mean(estimates)
    if std is None:
        std = np.std(estimates)
    lower_bound, upper_bound = stats.norm.interval(
        (1 - alpha),
        loc=mean,
        scale=std
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
