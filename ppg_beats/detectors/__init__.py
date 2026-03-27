"""PPG beat detection algorithms ported from the MATLAB ppg-beats toolbox."""

from .ampd import ampd_beat_detector
from .msptd import msptd_beat_detector, msptdfast_v1_beat_detector, msptdfast_v2_beat_detector
from .erma import erma_beat_detector
from .pda import pda_beat_detector
from .heartpy import heartpy_beat_detector
from .coppg import coppg_beat_detector
from .mmpdv2 import mmpdv2_beat_detector
from .qppg import qppg_beat_detector
from .atmax import atmax_beat_detector, atmin_beat_detector
from .swt import swt_beat_detector
from .wepd import wepd_beat_detector
from .ims import ims_beat_detector

__all__ = [
    "ampd_beat_detector",
    "msptd_beat_detector",
    "msptdfast_v1_beat_detector",
    "msptdfast_v2_beat_detector",
    "erma_beat_detector",
    "pda_beat_detector",
    "heartpy_beat_detector",
    "coppg_beat_detector",
    "mmpdv2_beat_detector",
    "qppg_beat_detector",
    "atmax_beat_detector",
    "atmin_beat_detector",
    "swt_beat_detector",
    "wepd_beat_detector",
    "ims_beat_detector",
]
