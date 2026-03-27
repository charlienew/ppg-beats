"""ERMA - Event-Related Moving Averages beat detector.

Reference:
    M. Elgendi et al., 'Systolic peak detection in acceleration
    photoplethysmograms measured from emergency responders in tropical
    conditions,' PLoS ONE, vol. 8, no. 10, 2013.
    https://doi.org/10.1371/journal.pone.0076585

Ported from erma_beat_detector.m (MIT License, Elisa Mejia Mejia & Peter H. Charlton).
"""

import numpy as np
from scipy.signal import butter, filtfilt

from .._utils import tidy_beats, pulse_onsets_from_peaks


def erma_beat_detector(sig: np.ndarray, fs: float) -> tuple:
    """Detect PPG beats using ERMA.

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
    ppg = sig.copy()

    # Bandpass filter 0.5-8 Hz
    b, a = butter(2, np.array([0.5, 8]) / (fs / 2), btype="bandpass")
    s = filtfilt(b, a, ppg)

    # Clip negative values
    z = s.copy()
    z[z < 0] = 0

    # Square
    y = z ** 2

    # First moving average (111 ms window)
    w1 = int(2 * np.floor((0.111 * fs) / 2) + 1)
    b1 = np.ones(w1) / w1
    ma_peak = filtfilt(b1, 1, y)

    # Second moving average (667 ms window)
    w2 = int(2 * np.floor((0.667 * fs) / 2) + 1)
    b2 = np.ones(w2) / w2
    ma_beat = filtfilt(b2, 1, y)

    # Thresholding
    alpha = 0.02 * np.mean(y)
    th1 = ma_beat + alpha
    boi = (ma_peak > th1).astype(float)

    # Find blocks of interest
    th2 = w1
    dboi = np.diff(boi)
    pos_blocks_init = np.where(dboi > 0)[0] + 1
    pos_blocks_end = np.where(dboi < 0)[0] + 1

    if len(pos_blocks_init) == 0 or len(pos_blocks_end) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    if pos_blocks_init[0] > pos_blocks_end[0]:
        pos_blocks_init = np.concatenate([[0], pos_blocks_init])
    if pos_blocks_init[-1] > pos_blocks_end[-1]:
        pos_blocks_end = np.concatenate([pos_blocks_end, [len(y)]])

    # Detect peaks within qualifying blocks
    peaks_list = []
    for i in range(len(pos_blocks_init)):
        # Find first end after this start
        ends_after = pos_blocks_end[pos_blocks_end > pos_blocks_init[i]]
        if len(ends_after) == 0:
            continue
        end_idx = ends_after[0]
        block_len = end_idx - pos_blocks_init[i]
        if block_len >= th2:
            block = ppg[pos_blocks_init[i] : end_idx]
            max_idx = int(np.argmax(block)) + pos_blocks_init[i]
            peaks_list.append(max_idx)

    peaks = tidy_beats(np.array(peaks_list, dtype=int))
    onsets = pulse_onsets_from_peaks(sig, peaks)
    return peaks, onsets
