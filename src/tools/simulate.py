import numpy as np
import tools

def sc(
    empirical_data,
    average_spectrum=False,
    sessions: int = 4,
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
    sc_data = None
    session_length = empirical_data.shape[-1] // sessions
    for session_idx in range(sessions):
        start = session_idx * session_length
        end = start + session_length
        session_data = empirical_data[:, start:end]
        session_fft = np.fft.rfft(session_data, axis=-1)
        session_fft_amplitude = np.abs(session_fft)
        if average_spectrum:
            session_fft_amplitude = np.mean(session_fft_amplitude, axis=0)
        noise = random.randn(*session_data.shape)
        random_phases = np.angle(np.fft.rfft(noise))
        simulated_spectrum = session_fft_amplitude \
            * np.exp(1j * random_phases)
        session_sc_data = np.fft.irfft(simulated_spectrum, axis=-1)
        session_sc_data = tools.normalized(session_sc_data, axis=-1)
        if sc_data is not None:
            sc_data = np.append(sc_data, session_sc_data, -1)
        else:
            sc_data = session_sc_data
    return sc_data

def sc_resample(
    empirical_data,
    sessions: int = 4,
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
    sc_data = None
    session_length = empirical_data.shape[-1] // sessions
    for session_idx in range(sessions):
        start = session_idx * session_length
        end = start + session_length
        session_data = empirical_data[:, start:end]
        session_fft = np.fft.rfft(session_data, axis=-1)
        session_fft_amplitude = np.abs(session_fft)
        auto_spectra = np.cov(session_fft_amplitude.T)
        average_spectra = np.mean(session_fft_amplitude, axis=0)
        resampled_fft_amplitude = np.abs(
            np.random.multivariate_normal(average_spectra, auto_spectra, empirical_data.shape[0]))
        noise = random.randn(*session_data.shape)
        random_phases = np.angle(np.fft.rfft(noise))
        simulated_spectrum = resampled_fft_amplitude \
            * np.exp(1j * random_phases)
        session_sc_data = np.fft.irfft(simulated_spectrum, axis=-1)
        session_sc_data = tools.normalized(session_sc_data, axis=-1)
        if sc_data is not None:
            sc_data = np.append(sc_data, session_sc_data, -1)
        else:
            sc_data = session_sc_data
    return sc_data

def pr(
    empirical_data,
    average_spectrum=False,
    sessions: int = 4,
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
    pr_data = None
    session_length = empirical_data.shape[-1] // sessions
    for session_idx in range(sessions):
        start = session_idx * session_length
        end = start + session_length
        session_data = empirical_data[:, start:end]
        session_fft = np.fft.rfft(session_data, axis=-1)
        session_fft_amplitude = np.abs(session_fft)
        if average_spectrum:
            session_fft_amplitude = np.mean(session_fft_amplitude, axis=0)
        noise = random.randn(session_data.shape[-1])
        random_phases = np.angle(np.fft.rfft(noise))
        session_fft_phases = np.angle(np.fft.rfft(session_data, axis=-1))
        simulated_spectrum = session_fft_amplitude \
            * np.exp(1j * (session_fft_phases + random_phases))
        session_pr_data = np.fft.irfft(simulated_spectrum, axis=-1)
        session_pr_data = tools.normalized(session_pr_data, axis=-1)
        if pr_data is not None:
            pr_data = np.append(pr_data, session_pr_data, -1)
        else:
            pr_data = session_pr_data
    return pr_data

def laumann(
    empirical_data,
    average_spectrum=True,
    sessions: int = 4,
    random: np.random.Generator = None):
    """_summary_

    Args:
        empirical_data (any): empirical data

    Returns:
        any: Laumann's surrogate data
    """
    if random is None:
        random = np.random
    sc_data = sc(empirical_data, sessions=sessions, average_spectrum=average_spectrum, random=random)
    laumann_data = None
    session_length = empirical_data.shape[-1] // sessions
    for session_idx in range(sessions):
        start = session_idx * session_length
        end = start + session_length
        session_data = empirical_data[:, start:end]
        session_sc_data = sc_data[:, start:end]
        session_cov = np.cov(session_data)
        chol_decomposition = np.linalg.cholesky(session_cov)
        session_laumann_data = np.dot(chol_decomposition, session_sc_data)
        session_laumann_data = tools.normalized(session_laumann_data, axis=-1)
        if laumann_data is not None:
            laumann_data = np.append(laumann_data, session_laumann_data, -1)
        else:
            laumann_data = session_laumann_data
    return laumann_data

def spectrally_constrained_pair(emp_pair, phase_lag = 0, noise_level = 0.5, length: int = -1, random: np.random.Generator = None):
    """
    Generates two simulated signals with power spectra of given signals
    and a controllable phase lag for all frequencies.
    """
    if random is None:
        random = np.random
    # Compute the average power spectrum
    power_spectrum1 = np.abs(np.fft.rfft(emp_pair[0], axis=-1))
    power_spectrum2 = np.abs(np.fft.rfft(emp_pair[1], axis=-1))
    # create a time series of phases
    noise = random.randn(emp_pair[0].shape[-1])
    phases = np.angle(np.fft.rfft(noise))
    # Construct the first simulated signal
    complex_spectrum1 = power_spectrum1 * np.exp(1j * (phases))
    signal1 = tools.normalized(np.fft.irfft(complex_spectrum1, axis=-1)[500:-500])[0:length]
    complex_spectrum2 = power_spectrum2 * np.exp(1j * (phases + phase_lag))
    signal2 = tools.normalized(np.fft.irfft(complex_spectrum2, axis=-1)[500:-500])[0:length]
    if noise_level > 0:
        signal1 = signal1 + np.random.normal(scale=noise_level, size=signal1.shape[-1])
        signal2 = signal2 + np.random.normal(scale=noise_level, size=signal2.shape[-1])
    return tools.normalized(signal1), tools.normalized(signal2)

def sinusoid(length=900, freq=0.01, phase_shift=0, repition_rate=1.2):
    """
    Generate a simulated time-series signal with specified parameters.
    """
    t = np.arange(0, length*repition_rate, repition_rate)
    return np.sin(2 * np.pi * freq * t + phase_shift)
