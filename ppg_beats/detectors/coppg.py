"""COPPG - Percentile Peak Detector.

Reference:
    C. Orphanidou et al., 'Signal-quality indices for the electrocardiogram and
    photoplethysmogram: derivation and applications to wireless monitoring,'
    IEEE J Biomed Health Inform, vol. 19, no. 3, pp. 832-8, 2015.
    https://doi.org/10.1109/JBHI.2014.2338351

Ported from coppg_beat_detector.m (GPL-3.0, Peter H. Charlton / Christina Orphanidou).
"""

import numpy as np

from .._utils import tidy_beats


def coppg_beat_detector(sig: np.ndarray, fs: float) -> tuple:
    """Detect PPG beats using percentile-based thresholding.

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
    t = np.arange(len(sig)) / fs

    # Segmentation into 10s windows with 3s overlap
    win_length = 10
    overlap_sec = 3
    win_starts = np.arange(t[0], t[-1], win_length)
    win_ends = win_starts + win_length + overlap_sec
    win_ends[-1] = win_ends[-1] - overlap_sec

    overall_peaks = []
    overall_onsets = []

    for wno in range(len(win_starts)):
        mask = (t >= win_starts[wno]) & (t <= win_ends[wno])
        rel_i = np.where(mask)[0]
        if len(rel_i) == 0:
            continue
        rel_v = sig[rel_i]

        # Thresholds
        thresh1 = np.quantile(rel_v, 0.9)
        thresh2 = np.quantile(rel_v, 0.1)
        thresh3 = thresh2 + 0.3 * (thresh1 - thresh2)
        thresh4 = thresh2 + 0.7 * (thresh1 - thresh2)

        # Find all local peaks
        if len(rel_v) < 3:
            continue

        diffs_left = np.diff(rel_v)[:-1]  # diff at i, looking left
        diffs_right = np.diff(rel_v)[1:]  # diff at i, looking right

        left_pos = diffs_left > 0
        right_neg = diffs_right < 0
        right_zero = diffs_right == 0

        # Second right
        if len(rel_v) >= 4:
            diffs_second_right = np.diff(rel_v[1:])[1:]
            second_right_neg = np.zeros(len(left_pos), dtype=bool)
            second_right_neg[: len(diffs_second_right)] = diffs_second_right < 0
        else:
            second_right_neg = np.zeros(len(left_pos), dtype=bool)

        peak_mask = (left_pos & right_neg) | (left_pos & right_zero & second_right_neg)
        peak_indices = np.where(peak_mask)[0] + 1  # +1 for offset

        if len(peak_indices) == 0:
            peak_indices = np.array([int(np.argmax(rel_v))])

        peak_vals = rel_v[peak_indices]

        # Classify peaks by proximity to amplitude thresholds
        upper_diff = np.abs(peak_vals - thresh1)
        mid_high_diff = np.abs(peak_vals - thresh4)
        mid_low_diff = np.abs(peak_vals - thresh3)
        lower_diff = np.abs(peak_vals - thresh2)

        upper_pks = np.where(
            (upper_diff < mid_high_diff) & (upper_diff < mid_low_diff) & (upper_diff < lower_diff)
        )[0]

        if len(upper_pks) == 0:
            continue

        ppg_pks_i = peak_indices[upper_pks]
        # Eliminate peaks too close together (< fs/3 samples)
        if len(ppg_pks_i) > 1:
            good = np.where(np.diff(ppg_pks_i) >= fs / 3)[0] + 1
            ppg_pks_i = ppg_pks_i[good]

        if len(ppg_pks_i) < 2:
            overall_peaks.extend(rel_i[ppg_pks_i].tolist())
            continue

        # Find troughs between consecutive peaks
        ppg_trs_i = np.empty(len(ppg_pks_i) - 1, dtype=int)
        for s in range(len(ppg_pks_i) - 1):
            segment = rel_v[ppg_pks_i[s] : ppg_pks_i[s + 1] + 1]
            ppg_trs_i[s] = ppg_pks_i[s] + int(np.argmin(segment))

        overall_peaks.extend(rel_i[ppg_pks_i].tolist())
        overall_onsets.extend(rel_i[ppg_trs_i].tolist())

    peaks = tidy_beats(np.array(overall_peaks, dtype=int))
    onsets = tidy_beats(np.array(overall_onsets, dtype=int))
    return peaks, onsets
