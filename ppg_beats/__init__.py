"""ppg_beats - Python PPG beat detection library.

A Python port of the MATLAB ppg-beats toolbox, providing 15+ beat detection
algorithms with a NeuroKit2-compatible API.

Quick start::

    import ppg_beats

    # NeuroKit2-compatible interface
    info = ppg_beats.ppg_detectbeats(signal, sampling_rate=125, method="msptdfastv2")
    peaks = info["PPG_Peaks"]
    onsets = info["PPG_Onsets"]

    # MATLAB-compatible interface
    peaks, onsets, mid_amps = ppg_beats.detect_ppg_beats(signal, fs=125, method="erma")

    # List available detectors
    print(ppg_beats.list_detectors())

    # Call a detector directly
    peaks, onsets = ppg_beats.detectors.ampd_beat_detector(signal, fs=125)
"""

from .ppg_detectbeats import ppg_detectbeats, detect_ppg_beats, list_detectors, DETECTORS
from . import detectors
from ._utils import (
    tidy_beats,
    tidy_peaks_and_onsets,
    pulse_onsets_from_peaks,
    pulse_peaks_from_onsets,
    calc_mid_amp_points,
)

__version__ = "0.1.0"

__all__ = [
    # Main API
    "ppg_detectbeats",
    "detect_ppg_beats",
    "list_detectors",
    "DETECTORS",
    # Utilities
    "tidy_beats",
    "tidy_peaks_and_onsets",
    "pulse_onsets_from_peaks",
    "pulse_peaks_from_onsets",
    "calc_mid_amp_points",
    # Subpackage
    "detectors",
]
