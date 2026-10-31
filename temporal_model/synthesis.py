"""Synthetic multi-person windows: add the dynamics of one-person recordings.

To first order, the people in a room add their scattered paths to the static
channel, H(t) = S + sum_p D_p(t). The distributed CSI is already stable from
frame to frame (common phase std < 0.01 rad, gain variation < 0.3 %), so a
person's contribution in a window is simply the window's deviation from its
time average. A k-person window is then a base window (one person, or an
empty room) plus the deviations of other one-person windows of the same room,
each rescaled to the base's static level so that the source's
dynamic-to-static ratio is kept.
"""

import numpy as np


def dynamic_part(window, lowpass_hz=None, intervals=None):
    """(T, ...) complex window -> (deviation from the time mean, static RMS).

    lowpass_hz keeps only Doppler components below the cutoff, which drops most
    of the source's measurement noise (otherwise k sources add k noise floors).
    """
    static = window.mean(axis=0, keepdims=True)
    d = window - static
    if lowpass_hz is not None:
        fs = 1e6 / float(np.mean(intervals[:-1])) if intervals is not None else 200.0
        freqs = np.fft.fftfreq(len(window), d=1.0 / fs)
        spectrum = np.fft.fft(d, axis=0)
        spectrum[np.abs(freqs) > lowpass_hz] = 0
        d = np.fft.ifft(spectrum, axis=0)
    return d, float(np.sqrt(np.mean(np.abs(static) ** 2)))


def synthesize(base, sources, lowpass_hz=None, source_intervals=None):
    """Base window (T, 1, 3, 1024) complex64 + list of source windows -> base + rescaled source dynamics."""
    out = base.astype(np.complex128)
    base_static = float(np.sqrt(np.mean(np.abs(out.mean(axis=0)) ** 2)))
    for k, source in enumerate(sources):
        intervals = None if source_intervals is None else source_intervals[k]
        d, static = dynamic_part(source.astype(np.complex128), lowpass_hz, intervals)
        out += d * (base_static / static)
    return out.astype(np.complex64)
