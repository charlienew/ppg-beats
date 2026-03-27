"""ATmax/ATmin - Adaptive Threshold Method for PPG peak/trough detection.

Reference (algorithm):
    H. S. Shin et al., 'Adaptive threshold method for the peak detection of
    photoplethysmographic waveform,' Comput Biol Med, vol. 39, no. 12,
    pp. 1145-52, 2009. https://doi.org/10.1016/j.compbiomed.2009.10.006

Reference (implementation):
    D. Han et al., 'A Real-Time PPG Peak Detection Method for Accurate
    Determination of Heart Rate during Sinus Rhythm and Cardiac Arrhythmia,'
    Biosensors, vol. 12, no. 2, p. 82, 2022.
    https://doi.org/10.3390/bios12020082

Ported from atmax_beat_detector.m / atmin_beat_detector.m
(MIT License, Dong Han / Peter H. Charlton).
"""

import numpy as np
from scipy.signal import butter, filtfilt

from .._utils import tidy_beats, pulse_onsets_from_peaks


def _adaptive_threshold(raw_ppg, fs, v_max_flag=True):
    """Core adaptive threshold peak detection (Shin 2009)."""
    # Bandpass filter 0.5-20 Hz
    b, a = butter(6, np.array([0.5, 20]) / (fs / 2), btype="bandpass")
    filtered = filtfilt(b, a, raw_ppg)
    # Normalize and zero-mean
    std_val = np.std(filtered)
    if std_val > 0:
        filtered = filtered / std_val
    filtered = filtered - np.mean(filtered)

    Fs = fs
    slope_k = np.full(len(filtered), np.nan)
    peak_loc = np.full(len(filtered), np.nan)
    pk_idx = 0

    refractory_period = 0.6 * Fs
    temp_win_left = round(0.15 * Fs)
    temp_win_right = round(0.15 * Fs)

    s_r = -0.6 if v_max_flag else 0.6
    std_PPG = np.std(filtered) if v_max_flag else -np.std(filtered)

    slope_meet_flag = False
    slope_lower_flag = False
    prev_slope = np.nan

    # Initialize
    if v_max_flag:
        slope_k[0] = 0.2 * np.max(filtered)
    else:
        slope_k[0] = 0.2 * np.min(filtered)
    V_n_1 = slope_k[0]

    for kk in range(1, len(filtered)):
        if slope_meet_flag:
            slope_k[kk] = filtered[kk]

            # Check for turning point
            if kk >= 2:
                if v_max_flag:
                    turn = (slope_k[kk] < slope_k[kk - 1]) and (slope_k[kk - 1] > slope_k[kk - 2])
                else:
                    turn = (slope_k[kk] > slope_k[kk - 1]) and (slope_k[kk - 1] < slope_k[kk - 2])
            else:
                turn = False

            if turn:
                temp_left = max(0, kk - temp_win_left)
                temp_right = min(len(filtered), kk + temp_win_right + 1)
                local_check = filtered[temp_left:temp_right]

                if v_max_flag:
                    has_higher = np.any(local_check > slope_k[kk - 1])
                else:
                    has_higher = np.any(local_check < slope_k[kk - 1])

                if not has_higher:
                    if pk_idx > 0:
                        last_peak = int(peak_loc[pk_idx - 1])
                        if kk - last_peak > refractory_period:
                            peak_loc[pk_idx] = kk - 1
                            V_n_1 = filtered[last_peak]
                            refractory_period = 0.6 * (kk - last_peak)
                            pk_idx += 1
                            slope_meet_flag = False

                            new_slope = slope_k[kk - 1] + s_r * ((V_n_1 + std_PPG) / Fs)
                            # Ensure correct slope direction
                            slope_change = s_r * ((V_n_1 + std_PPG) / Fs)
                            if v_max_flag and slope_change > 0:
                                slope_change = -slope_change
                            elif not v_max_flag and slope_change < 0:
                                slope_change = -slope_change
                            slope_k[kk] = slope_k[kk - 1] + slope_change

                            if v_max_flag:
                                below = slope_k[kk] < filtered[kk]
                            else:
                                below = slope_k[kk] > filtered[kk]
                            if below:
                                slope_lower_flag = True
                                prev_slope = slope_k[kk]
                                slope_k[kk] = filtered[kk]
                    else:
                        if not has_higher:
                            peak_loc[pk_idx] = kk - 1
                            V_n_1 = slope_k[kk - 1]
                            pk_idx += 1
                            slope_meet_flag = False

                            slope_change = s_r * ((V_n_1 + std_PPG) / Fs)
                            if v_max_flag and slope_change > 0:
                                slope_change = -slope_change
                            elif not v_max_flag and slope_change < 0:
                                slope_change = -slope_change
                            slope_k[kk] = slope_k[kk - 1] + slope_change

                            if v_max_flag:
                                below = slope_k[kk] < filtered[kk]
                            else:
                                below = slope_k[kk] > filtered[kk]
                            if below:
                                slope_k[kk] = filtered[kk]
        else:
            slope_change = s_r * ((V_n_1 + std_PPG) / Fs)
            if v_max_flag and slope_change > 0:
                slope_change = -slope_change
            elif not v_max_flag and slope_change < 0:
                slope_change = -slope_change
            slope_k[kk] = slope_k[kk - 1] + slope_change

            # Check if slope meets signal
            if v_max_flag:
                slope_meet_flag = (slope_k[kk] < filtered[kk] and slope_k[kk - 1] > filtered[kk - 1])
            else:
                slope_meet_flag = (slope_k[kk] > filtered[kk] and slope_k[kk - 1] < filtered[kk - 1])

            if slope_meet_flag:
                slope_k[kk] = filtered[kk]
            else:
                if not slope_lower_flag:
                    if v_max_flag:
                        slope_lower_flag = (slope_k[kk] < filtered[kk] and slope_k[kk - 1] == filtered[kk - 1])
                    else:
                        slope_lower_flag = (slope_k[kk] > filtered[kk] and slope_k[kk - 1] == filtered[kk - 1])
                    if slope_lower_flag:
                        prev_slope = slope_k[kk]
                        slope_k[kk] = filtered[kk]
                else:
                    if v_max_flag:
                        ppg_below = filtered[kk] < prev_slope if not np.isnan(prev_slope) else False
                    else:
                        ppg_below = filtered[kk] > prev_slope if not np.isnan(prev_slope) else False
                    if ppg_below:
                        slope_k[kk] = prev_slope
                        slope_lower_flag = False
                        prev_slope = np.nan
                    else:
                        slope_k[kk] = filtered[kk]

    # Clean up
    peak_loc = peak_loc[~np.isnan(peak_loc)].astype(int)
    return peak_loc


def atmax_beat_detector(sig: np.ndarray, fs: float) -> tuple:
    """Detect PPG beats using adaptive threshold (peak detection).

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
    peaks = _adaptive_threshold(sig, fs, v_max_flag=True)
    peaks = tidy_beats(peaks)
    onsets = pulse_onsets_from_peaks(sig, peaks)
    return peaks, onsets


def atmin_beat_detector(sig: np.ndarray, fs: float) -> tuple:
    """Detect PPG beats using adaptive threshold (trough/onset detection).

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
    from .._utils import pulse_peaks_from_onsets
    onsets = _adaptive_threshold(sig, fs, v_max_flag=False)
    onsets = tidy_beats(onsets)
    peaks = pulse_peaks_from_onsets(sig, onsets)
    return peaks, onsets
