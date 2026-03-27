"""Main PPG beat detection dispatcher - NeuroKit2-compatible API.

Provides ``ppg_detectbeats()`` which mirrors NeuroKit2's ``ppg_findpeaks()``
interface, and ``detect_ppg_beats()`` which mirrors the MATLAB toolbox's
``detect_ppg_beats.m``.
"""

from typing import Dict, Union

import numpy as np

from ._utils import tidy_peaks_and_onsets, calc_mid_amp_points
from .detectors import (
    ampd_beat_detector,
    msptd_beat_detector,
    msptdfast_v1_beat_detector,
    msptdfast_v2_beat_detector,
    erma_beat_detector,
    pda_beat_detector,
    heartpy_beat_detector,
    coppg_beat_detector,
    mmpdv2_beat_detector,
    qppg_beat_detector,
    atmax_beat_detector,
    atmin_beat_detector,
    swt_beat_detector,
    wepd_beat_detector,
    ims_beat_detector,
)

# Registry mapping method names to detector functions
DETECTORS = {
    "ampd": ampd_beat_detector,
    "msptd": msptd_beat_detector,
    "msptdfast": msptdfast_v2_beat_detector,
    "msptdfastv1": msptdfast_v1_beat_detector,
    "msptdfastv2": msptdfast_v2_beat_detector,
    "erma": erma_beat_detector,
    "pda": pda_beat_detector,
    "heartpy": heartpy_beat_detector,
    "coppg": coppg_beat_detector,
    "mmpdv2": mmpdv2_beat_detector,
    "qppg": qppg_beat_detector,
    "atmax": atmax_beat_detector,
    "atmin": atmin_beat_detector,
    "swt": swt_beat_detector,
    "wepd": wepd_beat_detector,
    "ims": ims_beat_detector,
}


def list_detectors() -> list:
    """Return a sorted list of available detector names."""
    return sorted(DETECTORS.keys())


def ppg_detectbeats(
    ppg_cleaned: Union[list, np.ndarray],
    sampling_rate: int = 1000,
    method: str = "msptdfastv2",
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Detect beats in a cleaned PPG signal (NeuroKit2-compatible API).

    This function provides a drop-in replacement for NeuroKit2's
    ``ppg_findpeaks()`` and can be used in NeuroKit2 processing pipelines.

    Parameters
    ----------
    ppg_cleaned : array_like
        Cleaned PPG signal (1-D).
    sampling_rate : int
        Sampling frequency in Hz (default: 1000).
    method : str
        Beat detection algorithm name. Use ``list_detectors()`` to see all
        available methods. Default: ``"msptdfastv2"``.
    **kwargs
        Additional keyword arguments (reserved for future per-method options).

    Returns
    -------
    dict
        Dictionary with at least:
        - ``"PPG_Peaks"`` : np.ndarray of peak sample indices
        - ``"PPG_Onsets"`` : np.ndarray of onset (trough) sample indices
        - ``"method"`` : name of the detection method used

    Examples
    --------
    Standalone usage::

        import ppg_beats
        info = ppg_beats.ppg_detectbeats(signal, sampling_rate=125, method="msptd")
        peaks = info["PPG_Peaks"]

    With NeuroKit2::

        import neurokit2 as nk
        import ppg_beats

        ppg_clean = nk.ppg_clean(ppg_raw, sampling_rate=125)
        info = ppg_beats.ppg_detectbeats(ppg_clean, sampling_rate=125, method="erma")
        # Use peaks with nk.signal_rate(), nk.signal_fixpeaks(), etc.
        hr = nk.signal_rate(info["PPG_Peaks"], sampling_rate=125)
    """
    sig = np.asarray(ppg_cleaned, dtype=float).ravel()
    method_lower = method.lower().replace("-", "").replace("_", "")

    if method_lower not in DETECTORS:
        available = ", ".join(list_detectors())
        raise ValueError(
            f"Unknown method '{method}'. Available methods: {available}"
        )

    detector_fn = DETECTORS[method_lower]
    peaks, onsets = detector_fn(sig, sampling_rate)

    # Apply post-processing
    peaks, onsets = tidy_peaks_and_onsets(sig, peaks, onsets)

    return {
        "PPG_Peaks": peaks,
        "PPG_Onsets": onsets,
        "method": method_lower,
    }


def detect_ppg_beats(
    sig: np.ndarray,
    fs: float,
    method: str = "msptdfastv2",
) -> tuple:
    """Detect beats in a PPG signal (MATLAB-toolbox-compatible API).

    This mirrors the MATLAB ``detect_ppg_beats.m`` function.

    Parameters
    ----------
    sig : array_like
        PPG signal values.
    fs : float
        Sampling frequency in Hz.
    method : str
        Beat detection algorithm name.

    Returns
    -------
    peaks : np.ndarray
        Indices of detected pulse peaks.
    onsets : np.ndarray
        Indices of detected pulse onsets.
    mid_amps : np.ndarray
        Indices of mid-amplitude points on the upslope.
    """
    sig = np.asarray(sig, dtype=float).ravel()
    info = ppg_detectbeats(sig, sampling_rate=int(fs), method=method)
    peaks = info["PPG_Peaks"]
    onsets = info["PPG_Onsets"]

    if len(peaks) > 0 and len(onsets) > 0:
        mid_amps = calc_mid_amp_points(sig, peaks, onsets)
    else:
        mid_amps = np.array([], dtype=int)

    return peaks, onsets, mid_amps
