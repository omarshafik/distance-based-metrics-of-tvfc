"""
function definitions to apply statistical tests on computed estimates
"""
import warnings
import numpy as np
from scipy import stats
import statsmodels.api as sm
from joblib import Parallel, delayed
from statsmodels.tools.sm_exceptions import InterpolationWarning
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


def test_distribution(data) -> tuple:
    """ compute skewness and kurtosis of the data distribution

    Args:
        data (any): data to compute statistics for

    Returns:
        tuple: tuple of skewness and kurtosis
    """
    return (stats.skew(data.flatten()), stats.kurtosis(data.flatten()))

def get_edges_of_interest(
    empirical_measures: np.ndarray,
    surrogate_measures: np.ndarray,
    alpha: float = 0.001,
    bonferroni: bool = False,
    one_side: bool = False
) -> int:
    """get edges that exhibit properties beyond given surrogate.

    Args:
        empirical_measures (np.ndarray): empirical measures.
        surrogate_measures (np.ndarray): surrogate measures.
        alpha (float, optional): significance level to use. Defaults to 0.05.
        one_side (bool, optional): carry out one-side hypothesis testing. Defaults to False.
    """
    n_edges = empirical_measures.shape[0]
    edges_of_interest = np.zeros(n_edges, dtype=int)
    if bonferroni:
        alpha = alpha / 4  # divide by the number of sessions
    if not one_side:
        alpha = alpha / 2
        lower_bound = np.percentile(surrogate_measures, 100 * alpha)
        edges_of_interest += np.where(
            empirical_measures < lower_bound, 1, 0)
    upper_bound = np.percentile(surrogate_measures, 100 * (1 - alpha))
    edges_of_interest += np.where(
        empirical_measures > upper_bound, 1, 0)
    return edges_of_interest

def significance(
    estimates: np.ndarray,
    test = stats.norm.sf,
    mean: float = None,
    std: float = None,
    **kwargs) -> np.ndarray:
    """ get estimates significance. Spans from -1 to 1.

    Args:
        estimates (np.ndarray): array of computed estimates. 
            assumes that estimates are normally distributed
        mean (float, optional): Estimates' mean. Defaults to the mean of given estimates array.
        std (float, optional): Estimates' std. Defaults to the std of given estimates array.
    Returns:
        list: significance array with equal size to given estimates array
    """
    return (1 - (2 * test(estimates, loc=mean, scale=std, **kwargs)))

def significant_estimates(
    estimates: np.ndarray,
    null: np.ndarray = None,
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
    if null is not None:
        alpha = alpha / 2
        lower_bound = np.percentile(null, 100 * alpha)
        upper_bound = np.percentile(null, 100 * (1 - alpha))
    else:
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

def scaled_significance_rate(
    significance_array: np.ndarray):
    """get the significance rate of given indices normalized by the chance rate

    Args:
        significance_array (np.ndarray): array of significance, with values \
            either 0 (null) or 1 (significant)

    Returns:
        float: normalized significance rate
    """
    chance_significance_count_per_edge = np.sum(significance_array) / significance_array.shape[0]
    rate = np.sum(
        significance_array, axis=-1
    ) / (
        chance_significance_count_per_edge
    )
    return rate

def sdv(
    sr: np.ndarray,
    null_sr: np.ndarray):
    """get the significance discriminability variance (SDV)

    Args:
        significance_rate_of_interet (float): significance rate of interest edges
        false_significance_rate (float): significance rate of null edges

    Returns:
        float: the significance discriminability variance (SDV)
    """
    statistic = np.sum(
        (sr - np.mean(null_sr)) ** 2
    ) / (np.size(sr) * np.var(null_sr))
    return statistic

def sdr(
    significance_rate_of_interest: float,
    false_significance_rate: float):
    """get the significance discriminability rate (SDR)

    Args:
        significance_rate_of_interet (float): significance rate of interest edges
        false_significance_rate (float): significance rate of null edges

    Returns:
        float: the significance discriminability rate (SDR)
    """
    null_rate_upper_bound = 1
    null_rate_lower_bound = 1

    statistic = (
        np.sum(
            significance_rate_of_interest[significance_rate_of_interest > null_rate_upper_bound]
        ) + np.sum(
            false_significance_rate[false_significance_rate < null_rate_lower_bound]
        )
    ) / (
        np.sum(significance_rate_of_interest) + np.sum(false_significance_rate)
    )
    return statistic

def edr(
    significance_rate_of_interest: float,
    false_significance_rate: float):
    """get the Edge discriminability rate (EDR)

    Args:
        significance_rate_of_interet (float): significance rate of interest edges
        false_significance_rate (float): significance rate of null edges

    Returns:
        float: the edge discriminability rate (EDR)
    """
    null_rate_upper_bound = 1
    null_rate_lower_bound = 1

    statistic = (
        (
            significance_rate_of_interest[significance_rate_of_interest > null_rate_upper_bound].shape[0]
        ) + (
            false_significance_rate[false_significance_rate < null_rate_lower_bound].shape[0]
        )
    ) / (
        significance_rate_of_interest.shape[0] + false_significance_rate.shape[0]
    )
    return statistic
