"""
function definitions to compute time-varying connectvity estimates from fMRI data
"""
from itertools import combinations, product
from joblib import Parallel, delayed
import numpy as np
from scipy import stats
import tools.common as common

def swd_no_parallel(
    timeseries: np.ndarray,
    window_size: int = 15,
    derivative: np.ndarray = None,
    use_derivative: bool = True,
    use_actual: bool = True,
    transform: bool = True,
    distance: str = "euclidean",
    scale: float = 1,
    safe_guard: int = 0,
    kaiser_beta: int = 0,
    pad: bool = False,
    pairs: np.ndarray = None) -> np.ndarray:
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
    if pairs is None:
        # get an array of unique pair identifier, which will find out unique combinations of parcels
        pairs = np.array(list(combinations(range(timeseries.shape[0]), 2)))

    if derivative is None and use_derivative is True:
        derivative = common.derivative(timeseries, normalize=True)

    # Get distances between amplitudes
    distance_ts = np.abs(timeseries[pairs[:, 0]] - timeseries[pairs[:, 1]])

    # use differenced data to get distance between rates of change of parcel pairs (if specified)
    if use_derivative is True:
        distance_diff_ts = np.abs(
            derivative[pairs[:, 0]] - derivative[pairs[:, 1]])
        if use_actual is False:
            distance_ts = distance_diff_ts
        else:
            if distance == "euclidean":
                distance_ts = np.sqrt(distance_ts ** 2 + distance_diff_ts ** 2)
            else:
                distance_ts = (distance_ts + distance_diff_ts) / 2

    distance_ts = distance_ts / scale

    # log transform to:
    #  - stabilize mean and variance
    #  - allow distance-based values to change on a continous scale (-inf,+inf)
    #  - decrease distribution skewness
    if transform:
        distance_ts = -1 * np.log2(distance_ts + safe_guard)

    # get sliding window average (sample mean) values over the given window (sample) size
    sampled_distance_ts = common.sliding_average(
        distance_ts, window_size=window_size, kaiser_beta=kaiser_beta, pad=pad)

    return sampled_distance_ts


def swd(
    timeseries: np.ndarray,
    window_size: int = 15,
    derivative: np.ndarray = None,
    use_derivative: bool = True,
    use_actual: bool = True,
    transform: bool = True,
    safe_guard: int = 0,
    kaiser_beta: int = 5,
    pairs: np.ndarray = None,
    **kwargs
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

    timeseries = common.normalized(timeseries, axis=-1)

    if use_derivative:
        derivative = common.derivative(timeseries, normalize=True, axis=-1)
    else:
        derivative = None

    return np.squeeze(Parallel()(
        delayed(swd_no_parallel)(
            timeseries,
            derivative=derivative,
            use_derivative=use_derivative,
            window_size=window_size,
            use_actual=use_actual,
            transform=transform,
            safe_guard=safe_guard,
            kaiser_beta=kaiser_beta,
            pairs=np.array([pair]),
            **kwargs) for pair in pairs))

def swc(
    timeseries: np.ndarray,
    window_size: int = 5,
    window: callable = np.hamming,
    axis: int = -1,
    transform: bool = True,
    pairs: np.ndarray = None) -> np.ndarray:
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

    corr_values = []
    if window_size >= timeseries.shape[-1]:
        # the case of static FC
        corr_values = np.corrcoef(timeseries)
        if pairs is None:
            pairs = np.array(list(combinations(range(timeseries.shape[0]), 2)))
        corr_values = np.array([corr_values[i, j] for i, j in pairs])
    else:
        corr_values = np.squeeze(Parallel()(
            delayed(np.corrcoef)(
                timeseries[:, i:i+window_size] * window(window_size)
            ) for i in range(0, timeseries.shape[axis] - window_size + 1)))
        if pairs is None:
            pairs = np.array(list(combinations(range(timeseries.shape[0]), 2)))
        corr_values = np.array([corr_values[:, i, j] for i, j in pairs])

    if transform:
        return np.arctanh(corr_values)
    return corr_values

def mtd_no_parallel(
    timeseries: np.ndarray,
    window_size: int = 15,
    kaiser_beta: int = 0,
    pairs: np.ndarray = None) -> np.ndarray:
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
    if pairs is None:
        # get an array of unique pair identifier, which will find out unique combinations of parcels
        pairs = np.array(list(combinations(range(timeseries.shape[0]), 2)))

    # Get distances between amplitudes
    estimate = timeseries[pairs[:, 0]] * timeseries[pairs[:, 1]]

    # get sliding window average (sample mean) values over the given window (sample) size
    sampled_estimate = common.sliding_average(
        estimate, window_size=window_size, kaiser_beta=kaiser_beta, pad=False)

    return sampled_estimate


def mtd(
    timeseries: np.ndarray,
    window_size: int = 15,
    kaiser_beta: int = 5,
    pairs: np.ndarray = None,
    **kwargs
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

    timeseries = common.normalized(timeseries, axis=-1)
    timeseries = common.derivative(timeseries, normalize=True, axis=-1)
    return np.squeeze(Parallel()(
        delayed(mtd_no_parallel)(
            timeseries,
            window_size=window_size,
            kaiser_beta=kaiser_beta,
            pairs=np.array([pair]),
            **kwargs) for pair in pairs))
