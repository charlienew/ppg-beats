"""IMS - Incremental Merge Segmentation beat detector.

Reference:
    W. Karlen et al., 'Adaptive pulse segmentation and artifact detection in
    photoplethysmography for mobile applications,' in Proc. IEEE EMBS, 2012,
    pp. 3131-4. https://doi.org/10.1109/EMBC.2012.6346628

Ported from ims_beat_detector.m (GPL-3.0, Marco A. Pimentel / Peter H. Charlton).
"""

import numpy as np

from .._utils import tidy_beats


def _pulse_segment(y, fs, m):
    """Perform pulse segmentation with merge parameter m.

    Returns line segments and their signal representation.
    """
    N = len(y)
    if N < 2:
        return [], []

    # Step 1: Create line segments from slope changes
    segments = []
    seg_start = 0
    for i in range(1, N):
        if i == N - 1 or (y[i] - y[i - 1]) * (y[min(i + 1, N - 1)] - y[i]) <= 0:
            segments.append((seg_start, i, y[i] - y[seg_start]))
            seg_start = i

    if len(segments) < 2:
        return [], []

    # Step 2: Merge short segments (shorter than m samples)
    merged = True
    while merged:
        merged = False
        new_segments = []
        i = 0
        while i < len(segments):
            if i < len(segments) - 1:
                dur = segments[i][1] - segments[i][0]
                if dur < m:
                    # Merge with next segment
                    new_seg = (
                        segments[i][0],
                        segments[i + 1][1],
                        y[segments[i + 1][1]] - y[segments[i][0]],
                    )
                    new_segments.append(new_seg)
                    i += 2
                    merged = True
                    continue
            new_segments.append(segments[i])
            i += 1
        segments = new_segments

    return segments


def _detect_beats_ims(y, fs, m):
    """Detect beats using IMS with a specific m value."""
    segments = _pulse_segment(y, fs, m)
    if len(segments) < 3:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Find rising segments followed by falling segments
    peaks = []
    onsets = []

    # Adaptive threshold for amplitude
    slopes = [s[2] for s in segments]
    pos_slopes = [s for s in slopes if s > 0]
    if len(pos_slopes) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    mean_pos_slope = np.mean(pos_slopes)
    threshold = 0.3 * mean_pos_slope

    for i in range(len(segments) - 1):
        start_i, end_i, slope_i = segments[i]
        start_next, end_next, slope_next = segments[i + 1]

        # Rising followed by falling = peak at transition
        if slope_i > threshold and slope_next < 0:
            peaks.append(end_i)
            onsets.append(start_i)

    return np.array(peaks, dtype=int), np.array(onsets, dtype=int)


def ims_beat_detector(sig: np.ndarray, fs: float) -> tuple:
    """Detect PPG beats using Incremental Merge Segmentation.

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

    # Range of merge parameters (5ms to 100ms in 5ms steps)
    bounds = np.arange(0.005, 0.105, 0.005)
    m_values = np.unique(np.ceil(bounds * fs).astype(int))
    m_values = m_values[m_values >= 1]

    # Use default m = 30ms equivalent
    default_m = max(1, int(np.ceil(0.030 * fs)))

    peaks, onsets = _detect_beats_ims(sig, fs, default_m)

    if len(peaks) == 0:
        # Try other m values
        for m in m_values:
            peaks, onsets = _detect_beats_ims(sig, fs, m)
            if len(peaks) > 0:
                break

    if len(peaks) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Refine peak locations to actual signal maxima between onsets
    if len(onsets) >= 2:
        refined_peaks = np.empty(len(onsets) - 1, dtype=int)
        for i in range(len(onsets) - 1):
            segment = sig[onsets[i] : onsets[i + 1] + 1]
            refined_peaks[i] = onsets[i] + int(np.argmax(segment))
        peaks = refined_peaks

    peaks = tidy_beats(peaks)
    onsets = tidy_beats(onsets)
    return peaks, onsets
