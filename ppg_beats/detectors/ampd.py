"""AMPD - Automatic Multiscale-based Peak Detection.

Reference:
    F. Scholkmann et al., 'An efficient algorithm for automatic peak detection
    in noisy periodic and quasi-periodic signals,' Algorithms, vol. 5, no. 4,
    pp. 588-603, 2012. https://doi.org/10.3390/a5040588

Ported from ampd_beat_detector.m (MIT License, Peter H. Charlton).
"""

import numpy as np
from scipy.signal import detrend

from .._utils import tidy_beats, pulse_onsets_from_peaks


def _detect_peaks_using_ampd(x: np.ndarray) -> np.ndarray:
    """Core AMPD algorithm on a single window."""
    N = len(x)
    L = N // 2 - 1
    if L < 1:
        return np.array([], dtype=int)

    x = detrend(x)

    # Initialize LMS matrix with random values > 1
    m = 1.0 + np.random.rand(L, N)

    # Populate LMS matrix
    for k in range(1, L + 1):
        for i in range(k + 1, N - k + 1):
            if x[i - 1] > x[i - k - 1] and x[i - 1] > x[i + k - 1]:
                m[k - 1, i] = 0

    # Find scale with most local maxima (lowest gamma)
    gamma = np.sum(m, axis=1)
    lam = int(np.argmin(gamma))

    # Truncate to optimal scale
    m = m[: lam + 1, :]

    # Find peaks where std == 0 across all scales
    sigma = np.std(m, axis=0)
    p = np.where(sigma == 0)[0]
    return p


def ampd_beat_detector(sig: np.ndarray, fs: float) -> tuple:
    """Detect PPG beats using AMPD.

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
    """
    sig = np.asarray(sig, dtype=float).ravel()

    # Window signal into overlapping 6s windows
    win_len = 6  # seconds
    overlap = 0.2
    n_samps_win = int(win_len * fs)

    if len(sig) <= n_samps_win:
        win_starts = [0]
        win_ends = [len(sig)]
    else:
        win_offset = round(n_samps_win * (1 - overlap))
        win_starts = list(range(0, len(sig) - n_samps_win, win_offset))
        win_ends = [s + n_samps_win for s in win_starts]
        if win_ends[-1] < len(sig):
            win_starts.append(len(sig) - n_samps_win)
            win_ends.append(len(sig))

    all_peaks = []
    for ws, we in zip(win_starts, win_ends):
        win_sig = sig[ws:we]
        p = _detect_peaks_using_ampd(win_sig)
        # Correct: AMPD peaks are 1 index too high
        p = p - 1
        p = p[p >= 0]
        all_peaks.extend(p + ws)

    peaks = tidy_beats(np.array(all_peaks, dtype=int))
    onsets = pulse_onsets_from_peaks(sig, peaks)
    return peaks, onsets
