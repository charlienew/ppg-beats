"""QPPG - Adapted Onset Detector (Wei Zong's ABP/PPG beat detector).

Reference:
    A. N. Vest et al., 'An open source benchmarked toolbox for cardiovascular
    waveform and interval analysis,' Physiol Meas, vol. 39, no. 10, 2018.
    https://doi.org/10.1088/1361-6579/aae021

Ported from qppg_beat_detector.m (GPL-3.0, multiple authors / Peter H. Charlton).
"""

import numpy as np

from .._utils import tidy_beats, pulse_peaks_from_onsets


def _slpsamp(t, data, ebuf, lbuf, tt_2, aet, bufln, slpwindow):
    """Compute slope sum function value at time t."""
    while t > tt_2:
        if tt_2 > 0 and tt_2 - 1 >= 0 and tt_2 < len(data) and tt_2 - 1 < len(data):
            val1 = data[tt_2]
            val2 = data[tt_2 - 1]
        else:
            val1 = 0
            val2 = 0
        dy = val1 - val2
        if dy < 0:
            dy = 0
        tt_2 += 1
        M = int(tt_2 % (bufln - 1))
        ebuf[M] = dy
        aet = 0
        for j in range(slpwindow):
            p = M - j
            if p < 0:
                p += bufln
            aet += ebuf[p]
        lbuf[M] = aet
    M3 = int(t % (bufln - 1))
    return lbuf[M3], ebuf, lbuf, tt_2, aet


def qppg_beat_detector(sig: np.ndarray, fs: float) -> tuple:
    """Detect PPG beats using the QPPG onset detector.

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
    data = sig.copy()

    sps = fs
    BUFLN = 4096
    EYE_CLS = 0.34
    LPERIOD = int(sps * 8)
    SLPW = 0.17
    NDP = 2.5
    TmDEF = 5
    Tm = TmDEF

    INVALID_DATA = -32768

    # Handle invalid data
    if data[0] <= INVALID_DATA + 10:
        data[0] = np.mean(data)
    inv = np.where(data <= INVALID_DATA + 10)[0]
    for i in inv:
        if i > 0:
            data[i] = data[i - 1]

    # Rescale data to ~ +/- 2000
    if len(data) < 5 * 60 * sps:
        dmin, dmax = np.min(data), np.max(data)
        if dmax - dmin > 0:
            data = (data - dmin) / (dmax - dmin) * 4000 - 2000
    else:
        n_segments = int(np.ceil(len(data) / (5 * 60 * sps)))
        max_vals = []
        min_vals = []
        for i in range(n_segments):
            start = int(i * 5 * 60 * sps)
            end = min(int((i + 1) * 5 * 60 * sps), len(data))
            max_vals.append(np.max(data[start:end]))
            min_vals.append(np.min(data[start:end]))
        med_max = np.median(max_vals)
        med_min = np.median(min_vals)
        if med_max - med_min > 0:
            data = (data - med_min) / (med_max - med_min) * 4000 - 2000

    EyeClosing = round(sps * EYE_CLS)
    ExpectPeriod = round(sps * NDP)
    SLPwindow = round(sps * SLPW)

    ebuf = np.zeros(BUFLN)
    lbuf = np.zeros(BUFLN)
    tt_2 = 0
    aet = 0

    from_idx = 0
    to_idx = len(data) - 1

    # Learning period
    t1 = int(8 * sps) + from_idx
    T0 = 0
    n = 0
    for t in range(from_idx, min(t1 + 1, len(data))):
        val, ebuf, lbuf, tt_2, aet = _slpsamp(t, data, ebuf, lbuf, tt_2, aet, BUFLN, SLPwindow)
        if val > INVALID_DATA + 10:
            T0 += val
            n += 1
    if n > 0:
        T0 /= n
    Ta = 3 * T0

    learning = True
    t = from_idx
    idx_peaks = []
    beat_n = 0

    while t <= to_idx:
        if learning:
            if t > from_idx + LPERIOD:
                learning = False
                T1 = T0
                t = from_idx
                continue
            else:
                T1 = 2 * T0

        temp, ebuf, lbuf, tt_2, aet = _slpsamp(t, data, ebuf, lbuf, tt_2, aet, BUFLN, SLPwindow)

        if temp > T1:
            timer = 0
            maxd = temp
            mind = maxd
            tmax = t
            for tt in range(t + 1, min(t + EyeClosing, to_idx + 1)):
                temp2, ebuf, lbuf, tt_2, aet = _slpsamp(tt, data, ebuf, lbuf, tt_2, aet, BUFLN, SLPwindow)
                if temp2 > maxd:
                    maxd = temp2
                    tmax = tt

            if maxd == temp:
                t += 1
                continue

            for tt in range(tmax, max(t - EyeClosing // 2, -1), -1):
                temp2, ebuf, lbuf, tt_2, aet = _slpsamp(tt, data, ebuf, lbuf, tt_2, aet, BUFLN, SLPwindow)
                if temp2 < mind:
                    mind = temp2

            if maxd > mind + 10:
                onset = (maxd - mind) / 100 + 2
                tpq = t - round(0.04 * fs)
                maxmin_2_3 = (maxd - mind) * 2.0 / 3

                for tt in range(tmax, max(t - EyeClosing // 2, -1), -1):
                    temp2, ebuf, lbuf, tt_2, aet = _slpsamp(tt, data, ebuf, lbuf, tt_2, aet, BUFLN, SLPwindow)
                    if temp2 < maxmin_2_3:
                        break

                for tt2 in range(tt, max(t - EyeClosing // 2 + round(0.024 * fs), -1), -1):
                    temp2, ebuf, lbuf, tt_2, aet = _slpsamp(tt2, data, ebuf, lbuf, tt_2, aet, BUFLN, SLPwindow)
                    temp3, ebuf, lbuf, tt_2, aet = _slpsamp(tt2 - round(0.024 * fs), data, ebuf, lbuf, tt_2, aet, BUFLN, SLPwindow)
                    if temp2 - temp3 < onset:
                        tpq = tt2 - round(0.016 * fs)
                        break

                # Find valley near tpq
                valley_v = round(tpq)
                search_start = max(1, round(tpq - 0.20 * fs))
                search_end = min(round(tpq + 0.05 * fs), len(data) - 2)
                for vi in range(search_start, search_end + 1):
                    if valley_v <= 0:
                        continue
                    if (0 <= valley_v < len(data) and 0 < vi < len(data) - 1 and
                            data[valley_v] > data[vi] and
                            data[vi] <= data[vi - 1] and data[vi] <= data[vi + 1]):
                        valley_v = vi

                if not learning:
                    valley_v = round(valley_v)
                    if valley_v > 0:
                        if beat_n == 0 or valley_v > idx_peaks[-1]:
                            idx_peaks.append(valley_v)
                            beat_n += 1

                Ta = Ta + (maxd - Ta) / 10
                T1 = Ta / 3
                t = round(tpq) + EyeClosing
            else:
                t += 1
                continue
        else:
            if not learning:
                timer = getattr(qppg_beat_detector, '_timer', 0) + 1
                qppg_beat_detector._timer = timer
                if timer > ExpectPeriod and Ta > Tm:
                    Ta -= 1
                    T1 = Ta / 3

        t += 1

    onsets = tidy_beats(np.array(idx_peaks, dtype=int))
    peaks = pulse_peaks_from_onsets(sig, onsets)
    return peaks, onsets
