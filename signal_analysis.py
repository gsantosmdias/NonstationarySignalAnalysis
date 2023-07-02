import numpy as np
from scipy import signal


def inst_spectral_entropy_stft(x, fs, window_size=256, hope_size=64, window='hann'):
    """
    Calculate the instantaneous spectral entropy of a signal using Short-Time Fourier Transform (STFT).

    Parameters:
    x (array_like): Input signal.
    fs (float): Sampling frequency of the input signal.
    window_size (int, optional): Size of the analysis window for STFT. Defaults to 256.
    hop_size (int, optional): Number of samples to shift the window for each STFT computation. Defaults to 64.
    window (str or tuple or array_like, optional): Desired window to use. Defaults to 'hann'.

    Returns:
    t (ndarray): Time vector corresponding to the computed Shannon spectral entropy values.
    shannon_entropy_norm (ndarray): Normalized Shannon spectral entropy values computed for each window.

    The function computes the instantaneous spectral entropy of the input signal x by performing the Short-Time
    Fourier Transform (STFT) using the scipy.signal.stft function. It then calculates the power spectrum of the STFT
    and proceeds to determine the Shannon spectral entropy in each window of the STFT. The Shannon spectral entropy
    measures the complexity or information content of the signal distribution at each frequency bin. The computed
    entropy values are then normalized by dividing them by the maximum possible entropy for the given number of
    frequency bins.

    The function returns the time vector t corresponding to the computed Shannon spectral entropy values, as well as
    the normalized Shannon spectral entropy values shannon_entropy_norm.
    """
    noverlap = window_size - hope_size
    # Compute the STFT using the scipy.signal.stft function
    f, t, zxx = signal.stft(x, fs, window=window, nperseg=window_size, noverlap=noverlap)

    # Compute the power spectrum of the STFT
    sxx = np.abs(zxx) ** 2

    # Initialize the Shannon spectral entropy array
    shannon_entropy = np.zeros(sxx.shape[1])

    # Compute the Shannon spectral entropy in each window
    for i in range(sxx.shape[1]):
        p = sxx[:, i] / np.sum(sxx[:, i])  # Computing the probability distribution
        shannon_entropy[i] = -np.sum(p * np.log2(p))  # Compute the entropy

    # Normalize the Shannon spectral entropy
    max_entropy = np.log2(sxx.shape[0])  # Maximum possible entropy
    shannon_entropy_norm = shannon_entropy / max_entropy  # Normalized entropy

    return t, shannon_entropy_norm


def inst_spectral_entropy_spectrogram(zxx):
    """
    Compute the instantaneous spectral Shannon entropy of a time series signal based on its power spectrogram.

    Parameters: zxx (ndarray): 2D numpy array with the STFT coefficients. Each column represents the values of each
    time window in the spectrogram.

    Returns:
    shannon_entropy_norm (ndarray): Instantaneous Shannon entropy values.

    The function calculates the power spectrum of the given STFT coefficients zxx by taking the absolute value
    squared. It then initializes an array to store the Shannon spectral entropy values. For each window in the
    spectrogram, the function computes the probability distribution by dividing the power spectrum values by the sum
    of the power spectrum values in that window. The Shannon spectral entropy is then computed by taking the negative
    sum of the probability distribution multiplied by the logarithm base 2 of the probability distribution. The
    computed entropy values are normalized by dividing them by the maximum possible entropy for the given number of
    frequency bins.

    The function returns the array of normalized Shannon spectral entropy values, shannon_entropy_norm.
    """

    # Compute the power spectrum of the STFT
    sxx = np.abs(zxx) ** 2

    # Initialize the Shannon spectral entropy array
    shannon_entropy = np.zeros(sxx.shape[1])

    # Compute the Shannon spectral entropy in each window
    for i in range(sxx.shape[1]):
        p = sxx[:, i] / np.sum(sxx[:, i])  # Computing the probability distribution
        shannon_entropy[i] = -np.sum(p * np.log2(p))  # Compute the entropy

    # Normalize the Shannon spectral entropy
    max_entropy = np.log2(sxx.shape[0])  # Maximum possible entropy
    shannon_entropy_norm = shannon_entropy / max_entropy  # Normalized entropy

    return shannon_entropy_norm


def spectral_entropy_welch_sv(x, fs):
    """
    Compute the Shannon Spectral Entropy of a signal.

    This function calculates the Shannon Spectral Entropy of a given signal, which provides a single number
    characterizing the spectral entropy and information content of the signal. The resulting value can be used to
    efficiently compare this signal with other signals.

    Parameters:
    x (array-like): Input signal.
    fs (float): Sampling frequency of the signal (in Hz).

    Returns:
    float: Normalized Shannon Spectral Entropy of the signal.

    Notes: - The input signal is assumed to be one-dimensional. - The signal is first transformed into the frequency
    domain using Welch's method to estimate the power spectrum. - The power spectrum is normalized to compute the
    probability distribution. - The Shannon entropy is then calculated as the negative sum of the probability
    distribution multiplied by the logarithm (base 2) of the probability distribution. - The resulting entropy value
    is normalized by dividing it by the maximum possible entropy, which is determined by the length of the power
    spectrum.
    """
    f, sx = signal.welch(x, fs)

    # Computing the probability distribution
    psd = sx / np.sum(sx)

    # Compute the Shannon entropy
    entropy = -np.sum(psd * np.log2(psd))

    # Normalize the spectral entropy
    max_entropy = np.log2(sx.shape[0])  # Maximum possible entropy
    spectral_entropy_norm = entropy / max_entropy

    return spectral_entropy_norm
