import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import plotly.graph_objs as go


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


def inst_freq(x, fs, window_size=256, hope_size=64, window='hann'):
    """
       Computes the instantaneous frequency of a non-stationary signal.

       The instantaneous frequency of a non-stationary signal is a time-varying parameter that relates to the average
       of the frequencies present in the signal as it evolves. This function estimates the instantaneous frequency
       as the first conditional spectral moment of the time-frequency distribution of the input signal.

       Args:
           x (array_like): The input signal.
           fs (float): The sampling frequency of the input signal.
           window_size (int, optional): The size of the analysis window for the Short-Time Fourier Transform (STFT).
               Defaults to 256.
           hope_size (int, optional): The number of samples the analysis window is shifted for each frame in the STFT.
               Defaults to 64.
           window (str, optional): The type of window function to use for the STFT. Defaults to 'hann'.

       Returns:
           tuple: A tuple containing:
               - t (array_like): The time values corresponding to the frames in the STFT.
               - f_inst (array_like): The estimated instantaneous frequency values for each frame.

       """
    noverlap = window_size - hope_size
    # Compute the STFT using the scipy.signal.stft function
    f, t, zxx = signal.stft(x, fs, window=window, nperseg=window_size, noverlap=noverlap)
    # Computes the spectrogram power spectrum
    sxx = np.abs(zxx) ** 2
    # Estimates the instantaneous frequency
    f = f.reshape((-1, 1))
    f_inst = np.dot(sxx.T, f) / np.sum(sxx, axis=0).reshape((-1, 1))

    return t, f_inst


def plot_spectrogram(x, fs, window_size=256, hope_size=64, window='hann', display_inst_freq=False):
    """
    Plot the spectrogram of an audio signal.

    Parameters:
        x (array-like): The input audio signal.
        fs (int): The sampling rate of the audio signal.
        window_size (int, optional): The size of the analysis window in samples. Default is 256.
        hope_size (int, optional): The size of the hop between successive windows in samples. Default is 64.
        window (str or tuple, optional): The window function to apply. Default is 'hann'.
        display_inst_freq (bool, optional): Whether to display the instantaneous frequency on the plot. Default is False.

    Returns:
        matplotlib.figure.Figure: The generated spectrogram plot.

    """
    # Compute the number of samples that overlap between windows
    noverlap = window_size - hope_size

    # Compute the Short-Time Fourier Transform (STFT) using the scipy.signal.stft function
    f, t, zxx = signal.stft(x, fs, window=window, nperseg=window_size, noverlap=noverlap)
    # Compute the spectrogram power spectrum P(t,f)
    sxx = np.abs(zxx) ** 2
    # Convert the power spectrum to decibels (dB)
    sxx_db = 20 * np.log(sxx / np.amax(sxx))
    # Create a new figure with a specified size
    plt.figure(figsize=(10, 6))
    # Create a pseudocolor plot of the spectrogram
    plt.pcolormesh(t, f, sxx_db, shading='gouraud')
    # Set the title and axis labels
    plt.title('Spectrogram [dB]')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    # Add a colorbar to the plot
    plt.colorbar()
    # Set the color map
    plt.set_cmap("magma")
    # Optionally, display the instantaneous frequency
    if display_inst_freq:
        _, f_inst = inst_freq(x, fs, window_size=window_size, hope_size=hope_size, window=window)
        plt.plot(t, f_inst, color='r', label='Instantaneous Frequency')
        # Add a legend to the plot
        plt.legend()
    # Return the generated figure
    return plt.gcf()


def plotly_spectrogram(x, fs, window_size=256, hope_size=64, window='hann', display_inst_freq=False):
    """
    Plot the spectrogram of an audio signal using Plotly.

    Parameters:
        x (array-like): The input audio signal.
        fs (int): The sampling rate of the audio signal.
        window_size (int, optional): The size of the analysis window in samples. Default is 256.
        hope_size (int, optional): The size of the hop between successive windows in samples. Default is 64.
        window (str or tuple, optional): The window function to apply. Default is 'hann'.
        display_inst_freq (bool, optional): Whether to display the instantaneous frequency on the plot. Default is False.

    Returns:
        plotly.graph_objects.Figure: The generated spectrogram plot.

    """
    # Compute the number of samples that overlap between windows
    noverlap = window_size - hope_size
    # Compute the Short-Time Fourier Transform (STFT) using the scipy.signal.stft function
    f, t, zxx = signal.stft(x, fs, window=window, nperseg=window_size, noverlap=noverlap)
    # Compute the spectrogram power spectrum P(t,f)
    sxx = np.abs(zxx) ** 2
    # Convert the power spectrum to decibels (dB)
    sxx_db = 20 * np.log(sxx / np.amax(sxx))
    # Create the heatmap trace for the spectrogram
    trace = [go.Heatmap(
        x=t,
        y=f,
        z=sxx_db,
        colorscale="Magma",
    )]
    # Create the layout for the plot
    layout = go.Layout(
        title='Spectrogram with Plotly [dB]',
        yaxis=dict(title='Frequency [Hz]'),  # y-axis label
        xaxis=dict(title='Time [s]'),  # x-axis label
    )
    # Create the figure using the trace and layout
    fig = go.Figure(data=trace, layout=layout)
    # Optionally, compute and add the instantaneous frequency trace
    if display_inst_freq:
        f_aux = f.reshape((-1, 1))
        # Compute the instantaneous frequency using the inst_freq function
        _, f_inst = inst_freq(x, fs, window_size=window_size, hope_size=hope_size, window=window)
        # Add the line trace of the instantaneous frequency function
        fig.add_trace(go.Scatter(x=t, y=np.squeeze(f_inst),
                                 mode='lines+markers',
                                 name='InstFreq',
                                 line=dict(color='firebrick')))
    # Return the generated figure
    return fig
