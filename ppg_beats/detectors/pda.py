"""PDA - Peak Detection Algorithm (upslope-based).

Reference:
    E. J. Arguello Prada and R. D. Serna Maldonado, 'A novel and low-complexity
    peak detection algorithm for heart rate estimation from low-amplitude
    photoplethysmographic (PPG) signals,' J Med Eng Technol, vol. 42, no. 8,
    pp. 569-577, 2018. https://doi.org/10.1080/03091902.2019.1572237

Ported from pda_beat_detector.m (MIT License, Elisa Mejia Mejia & Peter H. Charlton).
"""

import numpy as np

from .._utils import tidy_beats, pulse_onsets_from_peaks


def pda_beat_detector(sig: np.ndarray, fs: float) -> tuple:
    """Detect PPG beats using the upslope-based PDA method.

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
    ppg = sig

    th = 6  # Initial upslope threshold
    pks = []
    pos_peak = []
    pos_peak_b = False
    n_up = 0
    n_up_pre = 0

    for i in range(1, len(ppg)):
        if ppg[i] > ppg[i - 1]:
            n_up += 1
        else:
            if n_up >= th:
                pos_peak.append(i)
                pos_peak_b = True
                n_up_pre = n_up
            else:
                if pos_peak_b:
                    if ppg[i - 1] > ppg[pos_peak[-1]]:
                        pos_peak[-1] = i - 1
                    else:
                        pks.append(pos_peak[-1])
                    th = 0.6 * n_up_pre
                    pos_peak_b = False
            n_up = 0

    peaks = tidy_beats(np.array(pks, dtype=int))
    onsets = pulse_onsets_from_peaks(sig, peaks)
    return peaks, onsets
