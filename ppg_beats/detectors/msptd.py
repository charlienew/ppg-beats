"""MSPTD - Multi-Scale Peak and Trough Detection (and MSPTDfast variants).

Reference:
    S. M. Bishop and A. Ercole, 'Multi-scale peak and trough detection
    optimised for periodic and quasi-periodic neuroscience data,' in
    Intracranial Pressure and Neuromonitoring XVI, 2018.
    https://doi.org/10.1007/978-3-319-65798-1_39

MSPTDfast reference:
    P. H. Charlton et al., 'MSPTDfast: An Efficient Photoplethysmography
    Beat Detection Algorithm,' Computing in Cardiology, 2024.

Ported from msptd_beat_detector.m and msptdfastv2_beat_detector.m
(MIT License, Peter H. Charlton).
"""

import numpy as np
from scipy.signal import detrend, decimate

from .._utils import tidy_beats


def _detect_peaks_and_onsets_msptd(
    x: np.ndarray,
    fs: float = None,
    find_pks: bool = True,
    find_trs: bool = True,
    use_reduced_scales: bool = False,
    plaus_hr_bpm: tuple = (30, 200),
) -> tuple:
    """Core MSPTD algorithm on a single window."""
    N = len(x)
    L = N // 2 - 1
    if L < 1:
        return np.array([], dtype=int), np.array([], dtype=int)

    # Determine max scale based on plausible HR
    if use_reduced_scales and fs is not None:
        plaus_hr_hz = np.array(plaus_hr_bpm) / 60.0
        durn_signal = N / fs
        init_scales = np.arange(1, L + 1)
        init_scales_fs = (L / init_scales) / durn_signal
        inc_log = init_scales_fs >= plaus_hr_hz[0]
        max_scale = np.max(np.where(inc_log)[0]) + 1 if np.any(inc_log) else L
    else:
        max_scale = L

    x = detrend(x)

    # Initialize LMS matrices
    if find_pks:
        m_max = np.zeros((max_scale, N), dtype=bool)
    if find_trs:
        m_min = np.zeros((max_scale, N), dtype=bool)

    # Populate LMS matrices
    for k in range(1, max_scale + 1):
        for i in range(k + 1, N - k + 1):
            if find_pks and x[i - 1] > x[i - k - 1] and x[i - 1] > x[i + k - 1]:
                m_max[k - 1, i - 1] = True
            if find_trs and x[i - 1] < x[i - k - 1] and x[i - 1] < x[i + k - 1]:
                m_min[k - 1, i - 1] = True

    # Find optimal scale
    p = np.array([], dtype=int)
    t = np.array([], dtype=int)

    if find_pks:
        gamma_max = np.sum(m_max, axis=1)
        lambda_max = int(np.argmax(gamma_max))
        m_max = m_max[: lambda_max + 1, :]
        m_max_sum = np.sum(~m_max, axis=0)
        p = np.where(m_max_sum == 0)[0]

    if find_trs:
        gamma_min = np.sum(m_min, axis=1)
        lambda_min = int(np.argmax(gamma_min))
        m_min = m_min[: lambda_min + 1, :]
        m_min_sum = np.sum(~m_min, axis=0)
        t = np.where(m_min_sum == 0)[0]

    return p, t


def msptd_beat_detector(sig: np.ndarray, fs: float) -> tuple:
    """Detect PPG beats using MSPTD.

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
        Indices of detected pulse onsets (troughs).
    """
    return _msptd_generic(sig, fs, win_len=6, do_ds=False, use_reduced_scales=False)


def msptdfast_v1_beat_detector(sig: np.ndarray, fs: float) -> tuple:
    """MSPTDfast v1.1 - optimized MSPTD with downsampling to 20 Hz."""
    return _msptd_generic(
        sig, fs, win_len=8, do_ds=True, ds_freq=20, use_reduced_scales=True
    )


def msptdfast_v2_beat_detector(sig: np.ndarray, fs: float) -> tuple:
    """MSPTDfast v2.0 - optimized MSPTD with 6s window and downsampling."""
    return _msptd_generic(
        sig, fs, win_len=6, do_ds=True, ds_freq=20, use_reduced_scales=True
    )


def _msptd_generic(
    sig: np.ndarray,
    fs: float,
    win_len: float = 6,
    overlap: float = 0.2,
    do_ds: bool = False,
    ds_freq: float = 20,
    use_reduced_scales: bool = False,
) -> tuple:
    """Generic MSPTD implementation shared by all variants."""
    sig = np.asarray(sig, dtype=float).ravel()
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

    # Setup downsampling
    ds_factor = 1
    ds_fs = fs
    if do_ds and fs > ds_freq:
        ds_factor = int(fs // ds_freq)
        ds_fs = fs / ds_factor

    all_peaks = []
    all_onsets = []

    for ws, we in zip(win_starts, win_ends):
        win_sig = sig[ws:we]

        if ds_factor > 1:
            rel_sig = win_sig[::ds_factor]
            rel_fs = ds_fs
        else:
            rel_sig = win_sig
            rel_fs = fs

        p, t = _detect_peaks_and_onsets_msptd(
            rel_sig,
            fs=rel_fs,
            find_pks=True,
            find_trs=True,
            use_reduced_scales=use_reduced_scales,
        )

        # Upsample indices
        if ds_factor > 1:
            p = p * ds_factor
            t = t * ds_factor

        # Determine tolerance for refinement
        if ds_factor > 1:
            if ds_fs < 10:
                tol_durn = 0.2
            elif ds_fs < 20:
                tol_durn = 0.1
            else:
                tol_durn = 0.05
        else:
            tol_durn = 0.05
        tol = int(np.ceil(fs * tol_durn))

        # Refine peak locations
        for i in range(len(p)):
            lo = max(0, p[i] - tol)
            hi = min(len(win_sig), p[i] + tol + 1)
            p[i] = lo + int(np.argmax(win_sig[lo:hi]))

        # Refine onset locations
        for i in range(len(t)):
            lo = max(0, t[i] - tol)
            hi = min(len(win_sig), t[i] + tol + 1)
            t[i] = lo + int(np.argmin(win_sig[lo:hi]))

        all_peaks.extend(p + ws)
        all_onsets.extend(t + ws)

    peaks = tidy_beats(np.array(all_peaks, dtype=int))
    onsets = tidy_beats(np.array(all_onsets, dtype=int))
    return peaks, onsets
