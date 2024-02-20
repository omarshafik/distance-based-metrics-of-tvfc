"""
common utility functions
"""
import os
import numpy as np

PRINT = 1

def normalized(timeseries_array: np.ndarray, axis: int = -1) -> np.ndarray:
    """ normalize given timeseries array

    Args:
        timeseries_array (np.ndarray): input timeseries array
        axis (int, optional): Defaults to -1

    Returns:
        np.ndarray: normalized timeseries array
    """
# normalize array along axis (proovided mean, standard deviation, and axis)
    mean = np.mean(timeseries_array, axis=axis, keepdims=True)
    std = np.std(timeseries_array, axis=axis, keepdims=True)
    demeaned_data = timeseries_array - mean
    return demeaned_data / std

def differenced(
    timeseries_array: np.ndarray,
    order: int = 1,
    axis: int = -1,
    centered: bool = False,
    normalize: bool = False) -> np.ndarray:
    """ Compute the time derivative for the specified order

    Args:
        timeseries_array (np.ndarray): input timeseries array
        order (int, optional): differentiation order to apply. Defaults to 1.
        axis (int, optional): Defaults to -1.
        normalize (bool, optional): \
            if True, the array is normalized on the differtiated axis after differentiation. \
            Defaults to False.

    Returns:
        np.ndarray: derivative array
    """
    pad_width = [(0, 0)] * timeseries_array.ndim
    if centered:
        pad_width[axis] = (1, 1)
    else:
        pad_width[axis] = (1, 0)
    derivative = np.pad(timeseries_array, pad_width, mode='edge')
    for _ in range(order):
        derivative = np.diff(derivative, axis=axis)
        if centered:
            derivative = sliding_average(derivative, 2, pad=False)
    if normalize:
        return normalized(derivative)
    return derivative

def sliding_average(
    timeseries_array: np.ndarray,
    window_size: int = 5,
    axis: int = -1,
    kaiser_beta: int = 0,
    integrate: bool = False,
    pad: bool = True) -> np.ndarray:
    """ get sliding average (sequential samples) of given timeseries array

    Args:
        timeseries_array (np.ndarray): input timeseries array
        window_size (int, optional): window size to use. Defaults to 5.
        axis (int, optional): Defaults to -1.
        kaiser_beta (int, optional): beta parameter for np.kaiser function. Defaults to 0.
        pad (bool, optional): if True, the given array is padded before applying the convolution. \
            Boundary effects apply. Defaults to True.

    Returns:
        np.ndarray: averaged array of timeseries
    """
    if window_size == 1:
        return timeseries_array
    if kaiser_beta:
        window = np.kaiser(window_size, beta=kaiser_beta)
    else:
        window = np.ones(window_size)

    if not integrate:
        window = window / np.sum(window)

    if pad:
        # Create padding based on the selected axis
        pad_size = (window_size - 1) // 2
        extra_pad = (window_size - 1) % 2
        pad_width = [(0, 0)] * timeseries_array.ndim
        pad_width[axis] = (pad_size + extra_pad, pad_size)
        padded_data = np.pad(timeseries_array, pad_width, mode='mean')
    else:
        padded_data = timeseries_array

    # Apply the sliding window and average along the specified axis
    averaged_data = np.apply_along_axis(
        lambda x: np.convolve(x, window, mode='valid'),
        axis=axis,
        arr=padded_data)
    return averaged_data

def pad_timeseries(timeseries: np.ndarray, size: int) -> np.ndarray:
    """ pad given timeseries in the last dimension

    Args:
        timeseries (np.ndarray): input timeseries array
        size (int): total size of padding

    Returns:
        np.ndarray: padded array
    """
    pad_size = (size - 1) // 2
    extra_pad = (size - 1) % 2
    pad_width = [(0, 0)] * timeseries.ndim
    pad_width[-1] = (pad_size + extra_pad, pad_size)
    return np.pad(timeseries, pad_width, mode='constant', constant_values=np.nan)

def find_segments(arr: np.ndarray) -> list:
    """ find sequential segments with values not equal to zero

    Args:
        arr (np.ndarray): input timeseries array

    Returns:
        list: array of tuples of indices. [(start, end), ...]
    """
    change_indices = np.where(np.diff(arr) != 0)[0]
    if len(change_indices) == 0:
        return []

    if arr[0] != 0:
        change_indices = np.insert(change_indices, 0, 0)

    if len(change_indices) % 2 != 0:
        change_indices = np.append(change_indices, len(arr) - 1)

    # Create an array of segment start and end indices
    segment_indices = change_indices.reshape(-1, 2)

    # Convert segment_indices into a list of tuples
    segments = [tuple(seg) for seg in segment_indices]

    return segments

def prep_emp_data(emp_data, num_sessions = 4, smooth = 10):
    """ prepare empirical data for tvFC processing

    Args:
        emp_data (any): empirical data
        num_sessions (int, optional): number of sessions. Defaults to 4.

    Returns:
        any: empirical data after normalizing and smoothing
    """
    session_length = int(emp_data.shape[-1] / num_sessions)
    emp_data_prepped = np.array([[]])
    for session_idx in range(num_sessions):
        session_start = session_idx * session_length
        session_end = session_start + session_length
        emp_session_data = emp_data[:, session_start:session_end]
        emp_session_data = normalized(emp_session_data, axis=-1)
        if smooth:
            emp_session_data = sliding_average(emp_session_data, kaiser_beta=5, window_size=smooth, pad=False)
            emp_session_data = normalized(emp_session_data, axis=-1)
        if session_idx > 0:
            emp_data_prepped = np.append(emp_data_prepped, emp_session_data, axis=-1)
        else:
            emp_data_prepped = emp_session_data

    return emp_data_prepped

def print_info(info_str: str, outdir: str = None):
    """print_info to stdout and to a file

    Args:
        info_str (str): string for output
    """
    if not PRINT:
        return
    print(info_str)
    if outdir is not None and os.path.exists(outdir):
        log_filename = os.path.join(outdir, "info.log")
        with open(log_filename, "a", encoding="utf-8") as logfile:
            logfile.write(info_str)
            logfile.write("\n")
