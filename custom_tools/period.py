import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf


def find_period(
    signal: NDArray[np.int64], threshold: float = 0.5, prominence: float = 0.1
) -> int:
    """
    Find the period of a signal using autocorrelation and peak detection.

    Parameters:
    - signal: 1D numpy array representing the signal.
    - threshold: Minimum height of peaks to be considered significant.
    - prominence: Minimum prominence of peaks to be considered significant.

    Returns:
    - The estimated period of the signal, or -1 if no significant period is found.
    """
    acf_values = acf(signal, fft=True, nlags=len(signal) - 1)

    peaks, properties = find_peaks(acf_values, height=threshold, prominence=prominence)

    if len(peaks) > 0:
        return peaks[0]

    return 0
