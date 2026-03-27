"""Core utilities shared across all beat detectors.

Provides post-processing functions equivalent to the MATLAB toolbox's
tidy_beats.m, pulse_onsets_from_peaks.m, and pulse_peaks_from_onsets.m,
plus the full tidy_peaks_and_onsets pipeline from detect_ppg_beats.m.
"""

import numpy as np
from scipy.signal import argrelextrema


def tidy_beats(beat_indices: np.ndarray) -> np.ndarray:
    """Clean up beat indices: flatten, sort, and remove duplicates."""
    beat_indices = np.asarray(beat_indices).ravel()
    beat_indices = beat_indices[~np.isnan(beat_indices)]
    return np.unique(beat_indices.astype(int))


def pulse_onsets_from_peaks(sig: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Find pulse onsets (troughs) between consecutive peaks."""
    sig = np.asarray(sig).ravel()
    peaks = np.asarray(peaks).ravel().astype(int)
    onsets = np.empty(len(peaks) - 1, dtype=int)
    for i in range(len(peaks) - 1):
        segment = sig[peaks[i] : peaks[i + 1] + 1]
        onsets[i] = int(np.argmin(segment)) + peaks[i]
    return onsets


def pulse_peaks_from_onsets(sig: np.ndarray, onsets: np.ndarray) -> np.ndarray:
    """Find pulse peaks (maxima) between consecutive onsets."""
    sig = np.asarray(sig).ravel()
    onsets = np.asarray(onsets).ravel().astype(int)
    peaks = np.empty(len(onsets) - 1, dtype=int)
    for i in range(len(onsets) - 1):
        segment = sig[onsets[i] : onsets[i + 1] + 1]
        peaks[i] = int(np.argmax(segment)) + onsets[i]
    return peaks


def calc_mid_amp_points(
    sig: np.ndarray, peaks: np.ndarray, onsets: np.ndarray
) -> np.ndarray:
    """Calculate mid-amplitude points on the upslope between onsets and peaks."""
    sig = np.asarray(sig).ravel()
    n = min(len(onsets), len(peaks))
    mid_amps = np.empty(n, dtype=int)
    for i in range(n):
        desired_ht = np.mean([sig[onsets[i]], sig[peaks[i]]])
        segment = sig[onsets[i] : peaks[i] + 1]
        mid_amps[i] = int(np.argmin(np.abs(segment - desired_ht))) + onsets[i]
    return mid_amps


# ---------------------------------------------------------------------------
# Full tidy_peaks_and_onsets pipeline (from detect_ppg_beats.m)
# ---------------------------------------------------------------------------


def _remove_repeated(peaks: np.ndarray, onsets: np.ndarray):
    """Remove duplicates and shared indices between peaks and onsets."""
    peaks = np.unique(peaks)
    onsets = np.unique(onsets)
    shared = np.intersect1d(peaks, onsets)
    if len(shared):
        peaks = np.setdiff1d(peaks, shared)
        onsets = np.setdiff1d(onsets, shared)
    return peaks, onsets


def _ensure_extremum_between(sig, other_extrema, extrema_type):
    """Ensure at least one local extremum of the opposite type exists between
    consecutive extrema of the same type. Remove the lesser one if not."""
    if len(other_extrema) < 2:
        return other_extrema

    if extrema_type == "pk":
        local_ext = argrelextrema(sig, np.less, order=1)[0]
    else:
        local_ext = argrelextrema(sig, np.greater, order=1)[0]

    local_ext_set = set(local_ext)

    changed = True
    while changed:
        changed = False
        els_to_remove = []
        for i in range(len(other_extrema) - 1):
            start = other_extrema[i] + 1
            end = other_extrema[i + 1]
            has_extremum = any(start <= e < end for e in local_ext_set)
            if not has_extremum:
                vals = sig[other_extrema[[i, i + 1]]]
                if extrema_type == "pk":
                    remove_idx = i + int(np.argmin(vals))
                else:
                    remove_idx = i + int(np.argmax(vals))
                els_to_remove.append(remove_idx)
        if els_to_remove:
            other_extrema = np.delete(other_extrema, els_to_remove)
            changed = True
    return other_extrema


def _insert_extremum_between(sig, extrema, other_extrema, other_type):
    """If two consecutive other_extrema exist without an extremum between them,
    insert one."""
    if len(other_extrema) < 2:
        return extrema, other_extrema

    # Build sorted combined array with type labels
    combined = np.concatenate([other_extrema, extrema])
    is_other = np.concatenate(
        [np.ones(len(other_extrema), dtype=bool), np.zeros(len(extrema), dtype=bool)]
    )
    order = np.argsort(combined)
    combined_sorted = combined[order]
    is_other_sorted = is_other[order]

    # Find consecutive other_extrema
    bad_els = []
    for i in range(len(is_other_sorted) - 1):
        if is_other_sorted[i] and is_other_sorted[i + 1]:
            bad_els.append(i)

    for bad_idx in bad_els:
        start = combined_sorted[bad_idx]
        end = combined_sorted[bad_idx + 1]
        segment = sig[start : end + 1]
        # Remove baseline wander effect
        baseline = np.linspace(sig[start], sig[end], len(segment))
        detrended = segment - baseline
        if other_type == "pk":
            idx = int(np.argmin(detrended))
        else:
            idx = int(np.argmax(detrended))
        new_ext = start + idx
        if new_ext == start or new_ext == end:
            # Remove the first of the pair instead
            other_extrema = other_extrema[other_extrema != start]
        else:
            extrema = np.sort(np.append(extrema, new_ext))

    return extrema, other_extrema


def _ensure_starts_onset_ends_peak(peaks, onsets):
    """Ensure sequence starts with onset and ends with peak."""
    while len(onsets) > 0 and len(peaks) > 0 and onsets[0] > peaks[0]:
        peaks = peaks[1:]
    while len(peaks) > 0 and len(onsets) > 0 and peaks[-1] < onsets[-1]:
        onsets = onsets[:-1]
    return peaks, onsets


def _ensure_same_count(peaks, onsets):
    """If either is empty, empty both."""
    if len(peaks) == 0:
        onsets = np.array([], dtype=int)
    if len(onsets) == 0:
        peaks = np.array([], dtype=int)
    return peaks, onsets


def tidy_peaks_and_onsets(
    sig: np.ndarray, peaks: np.ndarray, onsets: np.ndarray
) -> tuple:
    """Full post-processing pipeline matching detect_ppg_beats.m rules:
    (i)   No two points at the same time
    (ii)  At least one local minimum between consecutive peaks
    (iii) At least one local maximum between consecutive onsets
    (iv)  Alternates between onsets and peaks
    (v)   Starts with onset, ends with peak
    (vi)  Same number of peaks and onsets
    """
    sig = np.asarray(sig, dtype=float).ravel()
    peaks = np.asarray(peaks).ravel().astype(int)
    onsets = np.asarray(onsets).ravel().astype(int)

    if len(peaks) == 0 or len(onsets) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    # (i) Remove duplicates and shared indices
    peaks, onsets = _remove_repeated(peaks, onsets)

    # (ii) At least one local minimum between consecutive peaks
    peaks = _ensure_extremum_between(sig, peaks, "pk")

    # (iii) At least one local maximum between consecutive onsets
    onsets = _ensure_extremum_between(sig, onsets, "on")

    # (iv) Alternates between onsets and peaks
    onsets, peaks = _insert_extremum_between(sig, onsets, peaks, "pk")
    peaks, onsets = _insert_extremum_between(sig, peaks, onsets, "on")

    # (v) Starts with onset, ends with peak
    peaks, onsets = _ensure_starts_onset_ends_peak(peaks, onsets)

    # (vi) Same number
    peaks, onsets = _ensure_same_count(peaks, onsets)

    return peaks, onsets
