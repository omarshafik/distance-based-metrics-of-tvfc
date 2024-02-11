import numpy as np
import tools

def sc(
    empirical_data,
    average_spectrum=False,
    random: np.random.Generator = None):
    """
    Generate spectrally-constrained surrogate data from empirical data.

    Args:
        empirical_data (np.ndarray): The empirical time series data for \
                which the surrogate is to be generated.
            The data should be structured as a numpy array, \
                potentially multi-dimensional, where the last axis \
            is considered the time or sequence axis.
        random (np.random.Generator, optional): An instance of numpy's random number generator class.

    Returns:
        np.ndarray: The generated spectrally-constrained surrogate data.

    """
    if random is None:
        random = np.random
    empirical_fft = np.fft.fft(empirical_data, axis=-1)
    empirical_fft_amplitude = np.abs(empirical_fft)
    if average_spectrum:
        empirical_fft_amplitude = np.mean(empirical_fft_amplitude, axis=0)
    noise = random.randn(*empirical_data.shape)
    random_phases = np.angle(np.fft.fft(noise))
    simulated_spectrum = empirical_fft_amplitude \
        * np.exp(1j * random_phases)
    sc_data = np.fft.ifft(simulated_spectrum, axis=-1).real
    sc_data = tools.normalized(sc_data, axis=-1)
    return sc_data

def pr(
    empirical_data,
    average_spectrum=False,
    random: np.random.Generator = None):
    """
    Generate phase-randomized surrogate data based on the empirical data
    by randomizing the phase of the Fourier transform while preserving the amplitude spectrum.
    Ensures symmetrical phase distribution for real-valued time series data.

    Args:
        empirical_data (np.ndarray): The empirical time series data as a numpy array.

    Returns:
        any: MVPR surrogate data
    """
    if random is None:
        random = np.random
    empirical_fft = np.fft.fft(empirical_data, axis=-1)
    empirical_fft_amplitude = np.abs(empirical_fft)
    if average_spectrum:
        empirical_fft_amplitude = np.mean(empirical_fft_amplitude, axis=0)
    noise = random.randn(empirical_data.shape[-1])
    random_phases = np.angle(np.fft.fft(noise))
    # random_phases = random.uniform(low=-np.pi, high=np.pi, size=empirical_data.shape[1])
    empirical_fft_phases = np.angle(empirical_fft)
    simulated_spectrum = empirical_fft_amplitude \
        * np.exp(1j * (empirical_fft_phases + random_phases))
    pr_data = np.fft.ifft(simulated_spectrum, axis=-1).real
    pr_data = tools.normalized(pr_data, axis=-1)
    return pr_data

def laumann(
    empirical_data,
    average_spectrum=True,
    random: np.random.Generator = None):
    """_summary_

    Args:
        empirical_data (any): empirical data

    Returns:
        any: Laumann's surrogate data
    """
    if random is None:
        random = np.random
    sc_data = sc(empirical_data, average_spectrum=average_spectrum, random=random)
    emp_cov = np.cov(empirical_data)
    chol_decomposition = np.linalg.cholesky(emp_cov)
    laumann_data = np.dot(chol_decomposition, sc_data)
    laumann_data = tools.normalized(laumann_data, axis=-1)
    return laumann_data

def pr_new(
    empirical_data,
    random: np.random.Generator = None):
    """
    Generate phase-randomized surrogate data based on the empirical data
    by randomizing the phase of the Fourier transform while preserving the amplitude spectrum.
    Ensures symmetrical phase distribution for real-valued time series data.

    Args:
        empirical_data (np.ndarray): The empirical time series data as a numpy array.

    Returns:
        any: MVPR surrogate data
    """
    if random is None:
        random = np.random
    n_nodes = empirical_data.shape[0]
    fft = np.fft.fft(empirical_data, axis=-1)
    fft_amplitude = np.abs(fft)
    mean_fft_amplitude = np.mean(fft_amplitude, axis=0)
    cov_fft_amplitude = np.cov(fft_amplitude.T)
    fft_amplitude_resampled = np.abs(random.multivariate_normal(mean_fft_amplitude, cov_fft_amplitude, n_nodes))
    noise = random.randn(empirical_data.shape[-1])
    random_phases = np.angle(np.fft.fft(noise))
    # random_phases = random.uniform(low=-np.pi, high=np.pi, size=empirical_data.shape[1])
    fft_phases = np.angle(fft)
    simulated_spectrum = fft_amplitude_resampled \
        * np.exp(1j * (fft_phases + random_phases))
    pr_data = np.fft.ifft(simulated_spectrum, axis=-1).real
    pr_data = tools.normalized(pr_data, axis=-1)
    return pr_data
