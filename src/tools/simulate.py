import numpy as np
import tools

def sc(empirical_data):
    """_summary_

    Args:
        empirical_data (any): empirical data

    Returns:
        any: spectrally-constrained surrogate data
    """
    empirical_fft = np.fft.fft(empirical_data, axis=-1)
    empirical_fft_amplitude = np.abs(empirical_fft)
    phase_noise = np.random.uniform(low=-np.pi, high=np.pi, size=empirical_data.shape)
    simulated_spectrum = empirical_fft_amplitude \
        * np.exp(1j * phase_noise)
    sc_data = np.fft.ifft(simulated_spectrum, axis=-1).real
    sc_data = tools.normalized(sc_data, axis=-1)
    return sc_data

def pr(empirical_data):
    """_summary_

    Args:
        empirical_data (any): empirical data

    Returns:
        any: MVPR surrogate data
    """
    empirical_fft = np.fft.fft(empirical_data, axis=-1)
    empirical_fft_amplitude = np.abs(empirical_fft)
    phase_noise = np.random.uniform(low=-np.pi, high=np.pi, size=empirical_data.shape[1])
    empirical_fft_phases = np.angle(empirical_fft)
    simulated_spectrum = empirical_fft_amplitude \
        * np.exp(1j * (empirical_fft_phases + phase_noise))
    pr_data = np.fft.ifft(simulated_spectrum, axis=-1).real
    pr_data = tools.normalized(pr_data, axis=-1)
    return pr_data

def laumann(empirical_data):
    """_summary_

    Args:
        empirical_data (any): empirical data

    Returns:
        any: Laumann's surrogate data
    """
    noise = np.random.uniform(low=-np.pi, high=np.pi, size=empirical_data.shape)
    power_spectrum = np.mean(np.abs(np.fft.fft(empirical_data, axis=-1)), axis=0)
    simulated_spectrum = power_spectrum \
        * np.exp(1j * noise)
    sc_data = np.fft.ifft(simulated_spectrum, axis=-1).real
    emp_cov = np.cov(empirical_data)
    chol_decomposition = np.linalg.cholesky(emp_cov)
    laumann_data = np.dot(chol_decomposition, sc_data)
    laumann_data = tools.normalized(laumann_data, axis=-1)
    return laumann_data
