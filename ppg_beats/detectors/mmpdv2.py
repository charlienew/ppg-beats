"""MMPDv2 - Mountaineer's Method for Peak Detection v2.

Reference:
    E. J. A. Prada, 'The mountaineer's method for peak detection in
    photoplethysmographic signals,' Revista Facultad de Ingenieria,
    vol. 90, pp. 42-50, 2019. https://doi.org/10.17533/udea.redin.n90a06

Ported from mmpdv2_beat_detector.m (MIT License, E. J. Arguello Prada & Peter H. Charlton).
"""

import numpy as np

from .._utils import tidy_beats, pulse_onsets_from_peaks


def mmpdv2_beat_detector(sig: np.ndarray, fs: float) -> tuple:
    """Detect PPG beats using the Mountaineer's Method v2.

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
    y1 = sig
    num_samples = len(y1)

    cuenta_ascenso = 0
    ascenso_max = round(0.6 * fs * 0.15)
    primer_pulso = False
    ppi_value = 1.0
    refractory = 0.35

    t_pico = []
    amp_pico = []

    for i in range(1, num_samples):
        if y1[i] > y1[i - 1]:
            cuenta_ascenso += 1
        else:
            if cuenta_ascenso >= ascenso_max:
                if not primer_pulso:
                    primer_pulso = True
                    amp_pico.append(y1[i - 1])
                    t_pico.append(i - 1)
                    ascenso_max = round(0.6 * fs * 0.15)
                else:
                    time_since_last = (i - t_pico[-1]) / fs
                    if (time_since_last > 1.2 * ppi_value or
                            cuenta_ascenso > round((1.75 * ascenso_max) / 0.6)):
                        amp_pico.append(y1[i - 1])
                        t_pico.append(i - 1)
                        ascenso_max = round(0.6 * fs * 0.15)
                        ppi_value = (t_pico[-1] - t_pico[-2]) / fs
                        refractory = 0.35
                    elif time_since_last > refractory:
                        amp_pico.append(y1[i - 1])
                        t_pico.append(i - 1)
                        ascenso_max = round(0.6 * cuenta_ascenso)
                        ppi_value = (t_pico[-1] - t_pico[-2]) / fs
                        refractory = 0.75 * ppi_value

            cuenta_ascenso = 0

    peaks = tidy_beats(np.array(t_pico, dtype=int))
    onsets = pulse_onsets_from_peaks(sig, peaks)
    return peaks, onsets
