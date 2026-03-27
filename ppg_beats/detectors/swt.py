"""SWT - Stationary Wavelet Transform beat detector.

Reference (algorithm):
    S. Vadrevu and M. Sabarimalai Manikandan, 'A robust pulse onset and peak
    detection method for automated PPG signal analysis system,' IEEE Trans
    Instrum Meas, vol. 68, no. 3, pp. 807-817, 2019.
    https://doi.org/10.1109/TIM.2018.2857878

Reference (implementation):
    D. Han et al., 'A Real-Time PPG Peak Detection Method for Accurate
    Determination of Heart Rate during Sinus Rhythm and Cardiac Arrhythmia,'
    Biosensors, vol. 12, no. 2, p. 82, 2022.
    https://doi.org/10.3390/bios12020082

Ported from swt_beat_detector.m (MIT License, Dong Han / Peter H. Charlton).
"""

import numpy as np
from scipy.signal import filtfilt, convolve
from scipy.signal.windows import gaussian

from .._utils import tidy_beats

try:
    import pywt

    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


def swt_beat_detector(sig: np.ndarray, fs: float) -> tuple:
    """Detect PPG beats using the Stationary Wavelet Transform method.

    Parameters
    ----------
    sig : array_like
        PPG signal values.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    peaks : np.ndarray
        Indices of detected pulse peaks.
    onsets : np.ndarray
        Indices of detected pulse onsets.

    Raises
    ------
    ImportError
        If PyWavelets is not installed.
    """
    if not HAS_PYWT:
        raise ImportError("SWT detector requires PyWavelets: pip install pywavelets")

    sig = np.asarray(sig, dtype=float).ravel()
    X = sig.copy()

    # Pad signal to required length for SWT
    wname = "bior1.5"
    max_level = pywt.swt_max_level(len(X))
    length_must_divide = 2 ** max_level
    req_length = len(X) - (len(X) % length_must_divide) + length_must_divide
    if req_length < len(X):
        req_length += length_must_divide
    padding = req_length - len(X)
    deb_len = int(np.ceil(padding / 2))
    fin_len = int(np.floor(padding / 2))
    padded = np.concatenate([np.zeros(deb_len), X, np.zeros(fin_len)])

    # Stationary Wavelet Transform
    L = pywt.swt_max_level(len(padded))
    coeffs = pywt.swt(padded, wname, level=L)
    # coeffs is list of (cA, cD) from level L down to level 1
    # Reorder to match MATLAB indexing: level 1 is index 0
    coeffs = list(reversed(coeffs))
    swd = np.array([cd for _, cd in coeffs])

    # Multiscale sum: s1 = d3 + d4, s2 = d5 + d6 + d7
    if L >= 7:
        s1 = swd[2] + swd[3]  # d3 + d4
        s2 = swd[4] + swd[5] + swd[6]  # d5 + d6 + d7
    elif L >= 4:
        s1 = swd[2] + swd[3]
        s2 = np.zeros_like(s1)
        for i in range(4, min(L, 7)):
            s2 += swd[i]
    else:
        # Signal too short for meaningful wavelet analysis
        return np.array([], dtype=int), np.array([], dtype=int)

    # Multiscale product
    p = s1 * s2

    # Shannon Entropy Envelope
    eta = 0.01 + np.std(p)
    p_tilda = np.abs(p)
    p_tilda[p_tilda < eta] = 0

    # Normalize
    p_range = np.max(p_tilda) - np.min(p_tilda)
    if p_range > 0:
        norm_p = (p_tilda - np.min(p_tilda)) / p_range
    else:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Shannon entropy
    se = np.zeros_like(norm_p)
    nonzero = norm_p > 0
    se[nonzero] = -norm_p[nonzero] * np.log(norm_p[nonzero])

    if np.any(np.isnan(se)):
        se = np.nan_to_num(se, nan=0.0)

    # Smoothing with filtfilt
    filt_len = int(np.floor(0.2 * fs))
    if filt_len < 1:
        filt_len = 1
    b_filt = np.ones(filt_len)
    a_filt = np.array([-1.0])
    s_smooth = filtfilt(b_filt, a_filt, se)

    # Gaussian derivative kernel
    sigma_1 = int(np.floor(0.05 * fs))
    if sigma_1 < 1:
        sigma_1 = 1
    M = int(np.floor(2 * fs))
    if M < 2:
        M = 2
    g = gaussian(M, sigma_1)
    h_d = np.diff(g)
    z = convolve(s_smooth, h_d, mode="same")

    # Negative zero crossings
    zx = np.where((z[:-1] >= 0) & (z[1:] < 0))[0]

    # Peak correction for onsets
    search_intv = int(np.floor(0.1 * fs / 2))
    onset_zx = np.empty(len(zx), dtype=int)
    for i in range(len(zx)):
        lo = max(0, zx[i] - search_intv)
        hi = min(len(padded), zx[i] + search_intv + 1)
        segment = padded[lo:hi]
        onset_zx[i] = lo + int(np.argmin(segment))

    # Find peaks between consecutive onsets
    peak_zx = np.empty(max(0, len(onset_zx) - 1), dtype=int)
    for i in range(1, len(onset_zx)):
        segment = padded[onset_zx[i - 1] : onset_zx[i] + 1]
        peak_zx[i - 1] = onset_zx[i - 1] + int(np.argmax(segment))

    # Remove padding and adjust indices
    onset_zx = onset_zx[(onset_zx >= deb_len) & (onset_zx < len(padded) - fin_len)]
    onset_zx = onset_zx - deb_len
    peak_zx = peak_zx[(peak_zx >= deb_len) & (peak_zx < len(padded) - fin_len)]
    peak_zx = peak_zx - deb_len

    onsets = tidy_beats(onset_zx)
    peaks = tidy_beats(peak_zx)
    return peaks, onsets
