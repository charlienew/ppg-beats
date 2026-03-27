# ppg_beats — Python PPG Beat Detection Library

A Python port of the [MATLAB ppg-beats toolbox](https://ppg-beats.readthedocs.io/), providing 16 beat detection algorithms with a [NeuroKit2](https://neuropsychology.github.io/NeuroKit/)-compatible API.

For algorithm details and references, see the [MATLAB toolbox documentation](https://ppg-beats.readthedocs.io/en/latest/toolbox/ppg_beat_detectors/).

---

## Installation

### From local source

```bash
cd ppg_beats
pip install -e .
```

### Dependencies

| Package | Version | Required for |
|---------|---------|-------------|
| numpy | >= 1.21 | All detectors |
| scipy | >= 1.7 | All detectors |
| pywavelets | >= 1.1 | SWT detector only |

Optional:

- **neurokit2** — for integration with NeuroKit2 processing pipelines
- **matplotlib** — for plotting

---

## Quick Start

```python
import numpy as np
import ppg_beats

# Load or generate a PPG signal
# sig = ...  (1-D numpy array)
fs = 125  # sampling frequency in Hz

# Detect beats (NeuroKit2-compatible interface)
info = ppg_beats.ppg_detectbeats(sig, sampling_rate=fs, method="msptdfastv2")
peaks = info["PPG_Peaks"]     # sample indices of systolic peaks
onsets = info["PPG_Onsets"]   # sample indices of pulse onsets (troughs)

# List all available methods
print(ppg_beats.list_detectors())
```

---

## API Reference

### `ppg_detectbeats` — NeuroKit2-compatible interface

```python
ppg_beats.ppg_detectbeats(ppg_cleaned, sampling_rate=1000, method="msptdfastv2")
```

**Parameters:**

- **ppg_cleaned** (`array_like`) — Cleaned PPG signal (1-D). Accepts `list`, `np.ndarray`, or `pd.Series`.
- **sampling_rate** (`int`) — Sampling frequency in Hz. Default: `1000`.
- **method** (`str`) — Beat detection algorithm name (case-insensitive). Default: `"msptdfastv2"`.

**Returns:** `dict` with keys:

| Key | Type | Description |
|-----|------|-------------|
| `"PPG_Peaks"` | `np.ndarray` | Sample indices of detected systolic peaks |
| `"PPG_Onsets"` | `np.ndarray` | Sample indices of detected pulse onsets |
| `"method"` | `str` | Name of the method used |

The output dictionary is directly compatible with NeuroKit2 functions such as `nk.signal_rate()` and `nk.signal_fixpeaks()`.

**Example:**

```python
info = ppg_beats.ppg_detectbeats(sig, sampling_rate=125, method="erma")
peaks = info["PPG_Peaks"]
```

---

### `detect_ppg_beats` — MATLAB-compatible interface

```python
ppg_beats.detect_ppg_beats(sig, fs, method="msptdfastv2")
```

Mirrors the MATLAB `detect_ppg_beats.m` function signature.

**Parameters:**

- **sig** (`array_like`) — PPG signal values.
- **fs** (`float`) — Sampling frequency in Hz.
- **method** (`str`) — Beat detection algorithm name.

**Returns:** `tuple` of `(peaks, onsets, mid_amps)`

| Output | Type | Description |
|--------|------|-------------|
| `peaks` | `np.ndarray` | Sample indices of systolic peaks |
| `onsets` | `np.ndarray` | Sample indices of pulse onsets |
| `mid_amps` | `np.ndarray` | Sample indices of mid-amplitude points on the upslope |

**Example:**

```python
peaks, onsets, mid_amps = ppg_beats.detect_ppg_beats(sig, fs=125, method="pda")
```

---

### `list_detectors`

```python
ppg_beats.list_detectors()
```

Returns a sorted list of all available detector method names.

---

### Calling detectors directly

Each detector can also be called directly from the `detectors` subpackage:

```python
from ppg_beats.detectors import msptd_beat_detector

peaks, onsets = msptd_beat_detector(sig, fs)
```

All detectors share the same signature:

```python
def detector_name(sig: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    sig : array_like
        PPG signal values (1-D).
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    peaks : np.ndarray
        Indices of detected pulse peaks.
    onsets : np.ndarray
        Indices of detected pulse onsets.
    """
```

---

## Using with NeuroKit2

The library is designed as a drop-in extension for NeuroKit2's PPG processing pipeline.

### Basic integration

```python
import neurokit2 as nk
import ppg_beats

# Clean the signal with NeuroKit2
ppg_clean = nk.ppg_clean(ppg_raw, sampling_rate=125)

# Detect beats with ppg_beats
info = ppg_beats.ppg_detectbeats(ppg_clean, sampling_rate=125, method="msptdfastv2")

# Use NeuroKit2's heart rate computation
hr = nk.signal_rate(info["PPG_Peaks"], sampling_rate=125)

# Optionally fix artifacts in detected peaks
peaks_corrected = nk.signal_fixpeaks(
    info["PPG_Peaks"], interval_min=0.4, interval_max=2.0, method="kubios"
)
```

### Compare multiple detectors

```python
import ppg_beats
import numpy as np

results = {}
for method in ppg_beats.list_detectors():
    info = ppg_beats.ppg_detectbeats(sig, sampling_rate=fs, method=method)
    ibi = np.diff(info["PPG_Peaks"]) / fs  # inter-beat intervals in seconds
    results[method] = {
        "n_beats": len(info["PPG_Peaks"]),
        "mean_hr": 60 / np.mean(ibi) if len(ibi) > 0 else 0,
    }
```

---

## Post-Processing

Both `ppg_detectbeats` and `detect_ppg_beats` automatically apply the full post-processing pipeline from the MATLAB toolbox (`tidy_peaks_and_onsets`), which enforces:

1. No duplicate peaks or onsets
2. At least one local minimum between consecutive peaks
3. At least one local maximum between consecutive onsets
4. Alternating onset-peak-onset-peak pattern
5. Sequence starts with an onset and ends with a peak
6. Equal number of peaks and onsets

These utilities are also available individually:

```python
from ppg_beats import tidy_beats, tidy_peaks_and_onsets, pulse_onsets_from_peaks

# Clean up raw beat indices
clean_indices = tidy_beats(raw_indices)

# Full post-processing
peaks, onsets = tidy_peaks_and_onsets(sig, raw_peaks, raw_onsets)

# Derive onsets from peaks (or vice versa)
onsets = pulse_onsets_from_peaks(sig, peaks)
peaks = ppg_beats.pulse_peaks_from_onsets(sig, onsets)
```

---

## Available Detectors

All algorithms are ported from the [MATLAB ppg-beats toolbox](https://ppg-beats.readthedocs.io/en/latest/toolbox/ppg_beat_detectors/). See the MATLAB documentation for detailed algorithm descriptions and original references.

| Method name | Full name | Approach | MATLAB docs |
|-------------|-----------|----------|-------------|
| `ampd` | Automatic Multiscale-based Peak Detection | Multiscale scalogram analysis | [link](https://ppg-beats.readthedocs.io/en/latest/functions/ampd_beat_detector/) |
| `atmax` | Adaptive Threshold (Vmax) | Adaptive slope-based threshold crossing (peaks) | [link](https://ppg-beats.readthedocs.io/en/latest/functions/atmax_beat_detector/) |
| `atmin` | Adaptive Threshold (Vmin) | Adaptive slope-based threshold crossing (troughs) | [link](https://ppg-beats.readthedocs.io/en/latest/functions/atmin_beat_detector/) |
| `coppg` | Percentile Peak Detector | Percentile-based thresholding in 10s windows | [link](https://ppg-beats.readthedocs.io/en/latest/functions/coppg_beat_detector/) |
| `erma` | Event-Related Moving Averages | Bandpass + squaring + dual moving averages | [link](https://ppg-beats.readthedocs.io/en/latest/functions/erma_beat_detector/) |
| `heartpy` | HeartPy | Iterative peak fitting with rolling mean optimization | [link](https://ppg-beats.readthedocs.io/en/latest/functions/heartpy_beat_detector/) |
| `ims` | Incremental Merge Segmentation | Adaptive slope-based pulse segmentation | [link](https://ppg-beats.readthedocs.io/en/latest/functions/ims_beat_detector/) |
| `mmpdv2` | Mountaineer's Method v2 | Ascending slope counting with adaptive refractory period | [link](https://ppg-beats.readthedocs.io/en/latest/functions/mmpdv2_beat_detector/) |
| `msptd` | Multi-Scale Peak and Trough Detection | Dual-scalogram for peaks and troughs | [link](https://ppg-beats.readthedocs.io/en/latest/functions/msptd_beat_detector/) |
| `msptdfast` | MSPTDfast (alias for v2) | Optimized MSPTD with downsampling | [link](https://ppg-beats.readthedocs.io/en/latest/functions/msptdfastv2_beat_detector/) |
| `msptdfastv1` | MSPTDfast v1.1 | MSPTD + downsampling to 20 Hz + 8s window | [link](https://ppg-beats.readthedocs.io/en/latest/functions/msptdfastv1_beat_detector/) |
| `msptdfastv2` | MSPTDfast v2.0 | MSPTD + downsampling to 20 Hz + 6s window | [link](https://ppg-beats.readthedocs.io/en/latest/functions/msptdfastv2_beat_detector/) |
| `pda` | Peak Detection Algorithm | Simple upslope threshold detection | [link](https://ppg-beats.readthedocs.io/en/latest/functions/pda_beat_detector/) |
| `qppg` | Adapted Onset Detector | Slope sum function with adaptive thresholding | [link](https://ppg-beats.readthedocs.io/en/latest/functions/qppg_beat_detector/) |
| `swt` | Stationary Wavelet Transform | Wavelet decomposition + Shannon entropy envelope | [link](https://ppg-beats.readthedocs.io/en/latest/functions/swt_beat_detector/) |
| `wepd` | Waveform Envelope Peak Detection | Elliptic filter + moving average + envelope detection | [link](https://ppg-beats.readthedocs.io/en/latest/functions/wepd_beat_detector/) |

### Detectors not yet ported

The following MATLAB detectors are not included in this Python library:

| Detector | Reason |
|----------|--------|
| `abd` | Complex multi-stage filtering with Kaiser filter design; feasible but lower priority |
| `ppgpulses` | Custom low-pass differentiator filter design |
| `pwd` | Bessel filter + complex adaptive windowing |
| `qppgfast` | Optimized variant of `qppg`; the standard `qppg` is included |
| `spar` | Nonlinear dynamical systems approach requiring custom phase-space reconstruction; proprietary license |
| `wfd` | CWT-based singularity detection |

---

## Algorithm Selection Guide

**Recommended default:** `msptdfastv2` — the most recent and optimized algorithm from the toolbox authors.

| Use case | Recommended methods |
|----------|-------------------|
| General purpose | `msptdfastv2`, `msptd` |
| Low-complexity / real-time | `pda`, `mmpdv2` |
| Noisy signals | `erma`, `heartpy` |
| Onset detection | `qppg`, `atmin` |
| Research benchmarking | Use `list_detectors()` to run all |

---

## Differences from the MATLAB Toolbox

- **0-indexed:** Python uses 0-based indexing. Peak/onset indices refer to positions in the input array starting from 0.
- **NumPy arrays:** All inputs and outputs are NumPy arrays (the MATLAB toolbox uses column vectors).
- **No `eval()`:** The Python dispatcher uses a dictionary registry instead of MATLAB's `eval()`.
- **NeuroKit2 API:** The primary interface (`ppg_detectbeats`) returns a dictionary matching NeuroKit2 conventions.
- **Post-processing included:** Both API functions automatically apply `tidy_peaks_and_onsets` (matching MATLAB's `detect_ppg_beats.m` behavior).

---

## License

This Python library follows the same licensing as the original MATLAB implementations. Each detector file includes the license from its original source. The majority of detectors use the MIT License; some use GPL-3.0. See individual detector source files for details.
