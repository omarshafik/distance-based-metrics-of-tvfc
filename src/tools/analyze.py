"""
function definitions to compute time-varying connectvity estimates from fMRI data
"""
from itertools import combinations
from joblib import Parallel, delayed
import numpy as np
import numba as nb
import tools.common as common


@nb.njit(fastmath=True, cache=True)
def expanded_distance(
    x: np.ndarray,
    y: np.ndarray,
    x_derivative: np.ndarray,
    y_derivative: np.ndarray
) -> np.ndarray:
    """get the expanded distance between x and y. \
    Expanded Distance = Euclidean(Amplitude Distance, Derivative Distance)
    """
    amplitude_distance = np.abs(x - y)
    derivative_distance = np.abs(x_derivative - y_derivative)
    return np.sqrt(np.power(amplitude_distance, 2) + np.power(derivative_distance, 2))

@nb.njit(fastmath=True, cache=True)
def pwd(
    timeseries: np.ndarray,
    derivative: np.ndarray,
    pairs: np.ndarray) -> np.ndarray:
    """ compute TVC estimate from fMRI signals using distance-based approach

    Args:
        timeseries (np.ndarray): numpy array of time series of fMRI signals
        window_size (int, optional): window size to use. Defaults to 15.
        derivative (np.ndarray, optional): \
            numpy array of derivative of original fMRI timeseries. Defaults to None.
        use_derivative (bool, optional): \
            use the first order derivative in distance calculations. Defaults to True.
        use_actual (bool, optional): use actual fMRI timeseries. Defaults to True.
        transform (callable, optional): \
            the transform function to apply on computed distances. Defaults to np.log10.
        return_distance (bool, optional): \
            return distance values instead of inverted distance. Defaults to False.
        safe_guard (int, optional): a guard value to prevent division by zero. Defaults to 0.
        kaiser_beta (int, optional): \
            parameter to use for applying the sliding average on pointwise distance-based values. \
            if the value is not zero, a tapered window is applied. See np.kaiser for more info \
            Defaults to 0.
        pairs (np.ndarray, optional): numpy array of indices of region pairs. Defaults to None.
    Returns:
        np.ndarray: numpy array of TVC estimates
    """
    pwd_ts = np.empty((pairs.shape[0], timeseries.shape[-1]))
    for pair_idx, pair in enumerate(pairs):
    # Get distances between amplitudes
        expanded_distance_ts = np.sqrt(
            np.power(
                np.abs(timeseries[pair[0]] - timeseries[pair[1]]),
                2
            ) + np.power(
                np.abs(derivative[pair[0]] - derivative[pair[1]]),
                2
            )
        )
        # log transform to:
        #  - stabilize mean and variance
        #  - allow distance-based values to change on a continous scale (-inf,+inf)
        #  - decrease distribution skewness
        expanded_distance_ts = -1 * np.log2(expanded_distance_ts)
        pwd_ts[pair_idx] = expanded_distance_ts

    return pwd_ts

def swd(
    timeseries: np.ndarray,
    window_size: int = 15,
    derivative: np.ndarray = None,
    pairs: np.ndarray = None,
    kaiser_beta: int = 5,
    pad: bool = False,
) -> np.ndarray:
    """ compute SWD-based tvFC estimates in parallel

    Args:
        timeseries (np.ndarray): numpy array of time series of fMRI signals
        window_size (int, optional): window size to use. Defaults to 15.
        derivative (np.ndarray, optional): \
            numpy array of derivative of original fMRI timeseries. Defaults to None.
        use_derivative (bool, optional): \
            use the first order derivative in distance-based calculations. Defaults to True.
        use_actual (bool, optional): use actual fMRI timeseries. Defaults to True.
        transform (bool, optional): \
            Transform estimates using inverse log. Defaults to True.
        safe_guard (int, optional): a guard value to prevent division by zero. Defaults to 0.
        kaiser_beta (int, optional): \
            parameter to use for applying the sliding average on pointwise distance-based values. \
            if the value is not zero, a tapered window is applied. See np.kaiser for more info \
            Defaults to 0.
        pairs (np.ndarray, optional): numpy array of indices of region pairs. Defaults to None.
    Returns:
        np.ndarray: numpy array of TVC estimates
    """
    if pairs is None:
        # get an array of unique pair identifier, which will find out unique combinations of parcels
        pairs = np.array(list(combinations(range(timeseries.shape[0]), 2)))

    if derivative is None:
        derivative = common.derivative(timeseries, normalize=True, axis=-1)

    pwd_ts = pwd(timeseries=timeseries, pairs=pairs, derivative=derivative)

    # apply sliding-window average
    swd_ts = common.sliding_average(
        timeseries_array=pwd_ts,
        window_size=window_size,
        kaiser_beta=kaiser_beta,
        pad=pad
    )

    return np.squeeze(swd_ts)

def swc(
    timeseries: np.ndarray,
    window_size: int = 5,
    window: callable = np.hamming,
    axis: int = -1,
    transform: bool = True,
    pairs: np.ndarray = None,
    **kwargs) -> np.ndarray:
    """ compute TVC estimate from fMRI signals using sliding window correlation

    Args:
        timeseries (np.ndarray): numpy array of time series of fMRI signals
        window_size (int, optional): window size to use. Defaults to 15.
        axis (int, optional): the time axis/dimension . Defaults to -1.
        pairs (np.ndarray, optional): numpy array of indices of region pairs. Defaults to None.

    Returns:
        np.ndarray: numpy array of TVC estimates
    """
    if window_size == 1:
        return timeseries

    if pairs is None:
        pairs = np.array(list(combinations(range(timeseries.shape[0]), 2)))

    corr_values = []
    if window_size >= timeseries.shape[-1]:
        # the case of static FC
        corr_values = np.corrcoef(timeseries)
        corr_values = np.array([corr_values[i, j] for i, j in pairs])
    else:
        corr_values = np.squeeze(Parallel()(
            delayed(np.corrcoef)(
                timeseries[:, i:i+window_size] * window(window_size)
            ) for i in range(0, timeseries.shape[axis] - window_size + 1)))
        corr_values = np.array([corr_values[:, i, j] for i, j in pairs])

    if transform:
        return np.arctanh(corr_values)
    return corr_values

@nb.njit(fastmath=True, cache=True)
def pwcov(
    timeseries: np.ndarray,
    pairs: np.ndarray) -> np.ndarray:
    """ compute TVC estimate from fMRI signals using MTD approach

    Args:
        timeseries (np.ndarray): numpy array of time series of fMRI signals
        window_size (int, optional): window size to use. Defaults to 15.
        kaiser_beta (int, optional): \
            parameter to use for applying the sliding average on pointwise MTD values. \
            if the value is not zero, a tapered window is applied. See np.kaiser for more info \
            Defaults to 0.
        pairs (np.ndarray, optional): numpy array of indices of region pairs. Defaults to None.
    Returns:
        np.ndarray: numpy array of TVC estimates
    """
    # Get distances between amplitudes
    pwcov_ts = timeseries[pairs[:, 0]] * timeseries[pairs[:, 1]]
    return pwcov_ts

def mtd(
    timeseries: np.ndarray,
    window_size: int = 15,
    derivative: np.ndarray = None,
    kaiser_beta: int = 5,
    pairs: np.ndarray = None,
    pad: bool = False
) -> np.ndarray:
    """ compute MTD-based tvFC estimates in parallel

    Args:
        timeseries (np.ndarray): numpy array of time series of fMRI signals
        window_size (int, optional): window size to use. Defaults to 15.
        kaiser_beta (int, optional): \
            parameter to use for applying the sliding average on pointwise MTD values. \
            if the value is not zero, a tapered window is applied. See np.kaiser for more info \
            Defaults to 0.
        pairs (np.ndarray, optional): numpy array of indices of region pairs. Defaults to None.
    Returns:
        np.ndarray: numpy array of TVC estimates
    """
    if pairs is None:
        # get an array of unique pair identifier, which will find out unique combinations of parcels
        pairs = np.array(list(combinations(range(timeseries.shape[0]), 2)))

    if derivative is None:
        derivative = common.derivative(timeseries, normalize=True, axis=-1)
    pwcov_ts = pwcov(timeseries=derivative, pairs=pairs)
    mtd_ts = common.sliding_average(
        timeseries_array=pwcov_ts,
        window_size=window_size,
        kaiser_beta=kaiser_beta,
        pad=pad)
    return np.squeeze(mtd_ts)
