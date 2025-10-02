"""
filters.py — signal filtering utilities for BP estimation

Applies bandpass filtering to physiological signals (PPG, ECG).
Uses zero-phase filtering (filtfilt) to avoid phase distortion.
"""

from scipy.signal import butter, filtfilt


def bandpass_filter(signal, fs, lowcut=0.5, highcut=15, order=4):
    """
    Bandpass filter using Butterworth filter and zero-phase filtering.

    Args:
        signal (np.ndarray): input signal (1D)
        fs (float): sampling frequency (Hz)
        lowcut (float): low cutoff frequency (Hz)
        highcut (float): high cutoff frequency (Hz)
        order (int): filter order (default = 4)

    Returns:
        filtered_signal (np.ndarray): bandpass-filtered signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Clip for numerical stability
    low = max(low, 1e-5)
    high = min(high, 0.9999)

    b, a = butter(order, [low, high], btype="band")
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal
