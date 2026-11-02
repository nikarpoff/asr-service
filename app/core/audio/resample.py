"""
Any rate and channel count to the 16 kHz mono the models take (ASR-2).

The resampler works on the spectrum with numpy only: the signal is padded so that both lengths
are whole for the rate ratio, and its spectrum is cut (or zero-padded) at the new Nyquist
frequency. It is an ideal low-pass filter, and it keeps the duration to a sample.
"""

from math import gcd

import numpy as np

from app.core.config import AUDIO_SETTINGS as CFG

EDGE_MARGIN_S = 0.02                    # mirrored around the signal: the FFT wrap-around rings outside of it


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Mean of the channels of a (frames, channels) array."""
    if samples.ndim == 1:
        return samples.astype(np.float32, copy=False)
    return samples.mean(axis=1, dtype=np.float32)


def to_model_input(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """16 kHz mono float32 of any (frames, channels) audio."""
    return resample(to_mono(samples), sample_rate, CFG.SAMPLE_RATE)


def resample(samples: np.ndarray, rate: int, target: int) -> np.ndarray:
    """Band-limited resampling of a mono signal; the result has round(len * target / rate) samples."""
    count = samples.shape[0]
    if rate == target or count == 0:
        return samples.astype(np.float32, copy=False)

    # 1. Pad by a whole number of steps: `step` input samples are exactly `target / gcd` output ones
    step = rate // gcd(rate, target)
    margin = step * -(-int(EDGE_MARGIN_S * rate) // step)
    total = step * _next_smooth(-(-(count + 2 * margin) // step))
    padded = np.pad(samples.astype(np.float64), (margin, total - count - margin), mode="reflect" if count > 1 else "edge")

    # 2. Cut or extend the spectrum at the new Nyquist frequency
    out_total = total * target // rate
    spectrum = np.fft.rfft(padded)
    bins = out_total // 2 + 1
    if bins <= spectrum.shape[0]:
        spectrum = spectrum[:bins]
    else:
        spectrum = np.pad(spectrum, (0, bins - spectrum.shape[0]))
    resampled = np.fft.irfft(spectrum, out_total) * (out_total / total)

    # 3. Drop the padding
    start = margin * target // rate
    return resampled[start:start + round(count * target / rate)].astype(np.float32)


def _next_smooth(n: int) -> int:
    """The smallest 2^a·3^b·5^c not less than n: the FFT of such a length is fast."""
    best = None
    power5 = 1
    while power5 < 2 * n:
        power35 = power5
        while power35 < 2 * n:
            candidate = power35
            while candidate < n:
                candidate *= 2
            if best is None or candidate < best:
                best = candidate
            power35 *= 3
        power5 *= 5
    return best
