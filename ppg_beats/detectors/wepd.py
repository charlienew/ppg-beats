"""WEPD - Waveform Envelope Peak Detection.

Reference:
    D. Han et al., 'A Real-Time PPG Peak Detection Method for Accurate
    Determination of Heart Rate during Sinus Rhythm and Cardiac Arrhythmia,'
    Biosensors, vol. 12, no. 2, p. 82, 2022.
    https://doi.org/10.3390/bios12020082

Ported from wepd_beat_detector.m (MIT License, Peter H. Charlton).
"""

import numpy as np
from scipy.signal import ellip, filtfilt
from scipy.interpolate import interp1d

from .._utils import tidy_beats, pulse_onsets_from_peaks, pulse_peaks_from_onsets


def _mov_avg(a: np.ndarray, M: int) -> np.ndarray:
    """Moving average filter."""
    N = len(a)
    b = np.full(N, np.nan)
    for i in range(M, N - M - 1):
        if i - M < 0 or i + M >= N:
            continue
        b[i - M + 1] = np.sum(a[i - M : i + M + 1]) / (2 * M + 1)
    return b


def _find_minima(sig: np.ndarray) -> np.ndarray:
    """Find indices of local minima."""
    return np.where((sig[1:-1] < sig[:-2]) & (sig[1:-1] < sig[2:]))[0] + 1


def _find_hr(extrema_inds: np.ndarray, fs: float) -> float:
    """Calculate heart rate from extrema indices."""
    if len(extrema_inds) < 2:
        return 0.0
    rng = (extrema_inds[-1] - extrema_inds[0]) / fs
    if rng == 0:
        return 0.0
    return 60 * len(extrema_inds) / rng


def wepd_beat_detector(sig: np.ndarray, fs: float) -> tuple:
    """Detect PPG beats using Waveform Envelope Peak Detection.

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
    plausible_hrs = [30, 200]

    # Bandpass filter 0.5-5 Hz (elliptic)
    n_order = 3
    Rp = 10
    Rs = 50
    Wp = np.array([0.5, 5]) / (fs / 2)
    Wp = np.clip(Wp, 0.001, 0.999)
    b, a = ellip(n_order, Rp, Rs, Wp, btype="bandpass")
    bpf = filtfilt(b, a, sig)

    # Moving average filtering and differentiation
    M = round(fs / 10)
    b_ma = _mov_avg(bpf, M)
    M2 = round(fs / 9)
    b_ma = _mov_avg(b_ma, M2)

    # First-order difference
    c = np.full(len(sig), np.nan)
    c[:-1] = b_ma[1:] - b_ma[:-1]

    # Third moving average
    b_ma = _mov_avg(b_ma, M2)

    # Normalize
    non_nans = ~np.isnan(c)
    if not np.any(non_nans):
        return np.array([], dtype=int), np.array([], dtype=int)
    d = (c - np.nanmean(c)) / np.nanstd(c)

    # Find minima and maxima
    valid = ~np.isnan(d)
    d_filled = d.copy()
    d_filled[~valid] = 0

    d_min = _find_minima(d_filled)
    d_max = _find_minima(-d_filled)

    if len(d_min) < 2 and len(d_max) < 2:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Calculate HRs
    d_min_hr = _find_hr(d_min, fs)
    d_max_hr = _find_hr(d_max, fs)

    # Decide whether to use minima or maxima
    can_use_min = plausible_hrs[0] <= d_min_hr <= plausible_hrs[1]
    can_use_max = plausible_hrs[0] <= d_max_hr <= plausible_hrs[1]

    diff_hr = d_min_hr - d_max_hr
    if abs(diff_hr) > 10:
        if d_min_hr < d_max_hr:
            can_use_max = False
        else:
            can_use_min = False

    if can_use_max and can_use_min:
        # Choose sharper peaks
        if len(d_min) > 1 and d_min[0] > 0 and d_min[-1] < len(d_filled) - 1:
            min_sharp = np.mean([
                np.mean(d_filled[d_min - 1] - d_filled[d_min]),
                np.mean(d_filled[d_min + 1] - d_filled[d_min])
            ])
        else:
            min_sharp = 0
        if len(d_max) > 1 and d_max[0] > 0 and d_max[-1] < len(d_filled) - 1:
            max_sharp = np.mean([
                np.mean(d_filled[d_max] - d_filled[d_max - 1]),
                np.mean(d_filled[d_max] - d_filled[d_max + 1])
            ])
        else:
            max_sharp = 0
        if min_sharp > max_sharp:
            can_use_max = False
        else:
            can_use_min = False

    if can_use_max:
        min_els = d_max
        inv_log = True
    elif can_use_min:
        min_els = d_min
        inv_log = False
    else:
        # Fallback
        min_els = d_min if len(d_min) >= len(d_max) else d_max
        inv_log = len(d_max) > len(d_min)

    if inv_log:
        d_filled = -d_filled

    if len(min_els) < 2:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Use envelope to find beats - simplified approach
    # Find beats at the minima locations
    beat_indices = min_els.copy()

    # Eliminate overlapping beats
    tol_samps = round(0.3 * fs)
    rel_beats = []
    i = 0
    while i < len(beat_indices):
        cluster = [i]
        j = i + 1
        while j < len(beat_indices) and abs(beat_indices[j] - beat_indices[i]) < tol_samps:
            cluster.append(j)
            j += 1
        # Keep the one with minimum value
        best = cluster[int(np.argmin(d_filled[beat_indices[cluster]]))]
        rel_beats.append(beat_indices[best])
        i = j

    beat_indices = np.array(rel_beats, dtype=int)

    if inv_log:
        # These are peak-like detections
        peaks_raw = beat_indices
        onsets = np.empty(len(peaks_raw) - 1, dtype=int)
        for i in range(len(peaks_raw) - 1):
            segment = sig[peaks_raw[i] : peaks_raw[i + 1] + 1]
            onsets[i] = peaks_raw[i] + int(np.argmin(segment))
        onsets = tidy_beats(onsets)
        peaks = pulse_peaks_from_onsets(sig, onsets)
    else:
        onsets = beat_indices
        onsets = tidy_beats(onsets)
        peaks = pulse_peaks_from_onsets(sig, onsets)

    return peaks, onsets
