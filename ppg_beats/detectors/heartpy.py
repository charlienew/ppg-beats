"""HeartPy - Heart rate algorithm with iterative peak fitting.

Reference:
    P. van Gent et al., 'HeartPy: A novel heart rate algorithm for the analysis
    of noisy signals,' Transportation Research Part F, vol. 66, pp. 368-378, 2019.
    https://doi.org/10.1016/j.trf.2019.09.015

Ported from heartpy_beat_detector.m (MIT License, Paul van Gent / Peter H. Charlton).
"""

import numpy as np

from .._utils import tidy_beats, pulse_onsets_from_peaks


def _scale_data(data: np.ndarray, lower: float = 0, upper: float = 1024) -> np.ndarray:
    """Scale data to [lower, upper] range."""
    rng = np.max(data) - np.min(data)
    if rng == 0:
        return np.full_like(data, (lower + upper) / 2)
    return (upper - lower) * ((data - np.min(data)) / rng) + lower


def _enhance_peaks(hrdata: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Enhance peaks by squaring and rescaling."""
    hrdata = _scale_data(hrdata)
    for _ in range(iterations):
        hrdata = hrdata ** 2
        hrdata = _scale_data(hrdata)
    return hrdata


def _rolling_mean(hrdata: np.ndarray, windowsize: float, sample_rate: float) -> np.ndarray:
    """Calculate rolling mean with edge padding."""
    win_samps = round(windowsize * sample_rate)
    if win_samps < 1:
        win_samps = 1
    if win_samps >= len(hrdata):
        return np.full(len(hrdata), np.mean(hrdata))

    # Compute rolling mean using cumsum for efficiency
    cumsum = np.cumsum(np.insert(hrdata, 0, 0))
    rol_mean = (cumsum[win_samps:] - cumsum[:-win_samps]) / win_samps

    # Pad edges
    n_miss = (len(hrdata) - len(rol_mean)) // 2
    pad_start = np.full(n_miss, rol_mean[0])
    pad_end = np.full(len(hrdata) - len(rol_mean) - n_miss, rol_mean[-1])
    return np.concatenate([pad_start, rol_mean, pad_end])


def _detect_peaks(hrdata, rol_mean, ma_perc, sample_rate):
    """Detect peaks above rolling mean + ma_perc threshold."""
    mn = np.mean(rol_mean / 100) * ma_perc
    threshold = rol_mean + mn

    peaksx = np.where(hrdata > threshold)[0]
    if len(peaksx) == 0:
        return [], np.inf

    peaksy = hrdata[peaksx]

    # Find edges of peak groups
    edges = np.where(np.diff(peaksx) > 1)[0]
    edges = np.concatenate([[0], edges, [len(peaksx) - 1]])

    peaklist = []
    for i in range(len(edges) - 1):
        start = edges[i] if i == 0 else edges[i] + 1
        end = edges[i + 1] + 1
        if start >= end:
            continue
        group_vals = peaksy[start:end]
        max_idx = int(np.argmax(group_vals))
        peaklist.append(peaksx[start + max_idx])

    if len(peaklist) == 0:
        return [], np.inf

    peaklist = np.array(peaklist, dtype=int)

    # Remove first peak if within first 150ms
    if peaklist[0] <= (sample_rate / 1000.0) * 150:
        peaklist = peaklist[1:]

    if len(peaklist) < 2:
        return peaklist.tolist(), np.inf

    # Calculate RR intervals and their std
    rr_list = np.diff(peaklist) / sample_rate * 1000.0
    rrsd = float(np.std(rr_list))

    return peaklist.tolist(), rrsd


def heartpy_beat_detector(sig: np.ndarray, fs: float) -> tuple:
    """Detect PPG beats using the HeartPy algorithm.

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

    # Pre-processing: peak enhancement
    hrdata = _enhance_peaks(sig.copy())

    # Calculate rolling mean
    rol_mean = _rolling_mean(hrdata, 0.75, fs)

    # Test multiple MA percentages
    ma_perc_list = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 200, 300]
    bpmmin, bpmmax = 40, 180

    results = []
    for ma_perc in ma_perc_list:
        peaklist, rrsd = _detect_peaks(hrdata, rol_mean, ma_perc, fs)
        bpm = (len(peaklist) / (len(hrdata) / fs)) * 60 if len(hrdata) > 0 else 0
        results.append((ma_perc, peaklist, rrsd, bpm))

    # Find best MA percentage
    valid = [(r[0], r[1], r[2]) for r in results
             if r[2] > 0.1 and bpmmin <= r[3] <= bpmmax]

    if valid:
        best = min(valid, key=lambda x: x[2])
        best_peaks = best[1]
    else:
        # Fallback: use the setting with most peaks in range
        best_peaks = []
        for r in results:
            if len(r[1]) > len(best_peaks):
                best_peaks = r[1]

    if len(best_peaks) < 2:
        peaks = tidy_beats(np.array(best_peaks, dtype=int))
        return peaks, np.array([], dtype=int)

    best_peaks = np.array(best_peaks, dtype=int)

    # Check peaks: exclude RR interval outliers
    rr_list = np.diff(best_peaks) / fs * 1000.0
    mean_rr = np.mean(rr_list)
    thirty_perc = 0.3 * mean_rr
    if thirty_perc <= 300:
        upper_th = mean_rr + 300
        lower_th = mean_rr - 300
    else:
        upper_th = mean_rr + thirty_perc
        lower_th = mean_rr - thirty_perc

    bad_rr = (rr_list <= lower_th) | (rr_list >= upper_th)
    binary = np.ones(len(best_peaks), dtype=bool)
    bad_indices = np.where(bad_rr)[0]
    binary[bad_indices] = False

    peaks = tidy_beats(best_peaks[binary])
    onsets = pulse_onsets_from_peaks(sig, peaks)
    return peaks, onsets
