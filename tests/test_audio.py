"""
Audio to the model input: any rate and channel count become 16 kHz mono of the same duration (ASR-2).
"""

import io
import struct
import wave

import numpy as np
import pytest

from app.core.audio import decode, parse_media_type, resample, to_model_input, to_mono
from app.core.audio.decode import AudioFormat
from app.core.exceptions import InputTooLargeError, InvalidInputError, UnsupportedMediaTypeError

RATES = [8000, 16000, 22050, 44100, 48000]
WAV = parse_media_type("audio/wav")


def tone(frequency: float, seconds: float, rate: int) -> np.ndarray:
    return (0.5 * np.sin(2 * np.pi * frequency * np.arange(int(seconds * rate)) / rate)).astype(np.float32)


def dominant_frequency(samples: np.ndarray, rate: int) -> float:
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(samples.shape[0])))
    return np.argmax(spectrum) * rate / samples.shape[0]


def pcm_wav(frames: np.ndarray, rate: int, width: int = 2) -> bytes:
    """Integer PCM WAV of (frames, channels) floats with `width` bytes per sample."""
    scale = float(1 << (8 * width - 1))
    values = np.clip(np.round(frames * scale), -scale, scale - 1).astype(np.int64)
    if width == 1:
        data = (values + 128).astype(np.uint8).tobytes()
    elif width == 3:
        data = b"".join(int(value).to_bytes(3, "little", signed=True) for value in values.ravel())
    else:
        data = values.astype(f"<i{width}").tobytes()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(frames.shape[1])
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(data)
    return buffer.getvalue()


@pytest.mark.parametrize("rate", RATES)
@pytest.mark.parametrize("channels", [1, 2], ids=["mono", "stereo"])
def test_any_rate_to_16k_mono(rate, channels):
    seconds = 1.5
    signal = tone(440, seconds, rate)
    frames = np.stack([signal] * channels, axis=1)

    audio = decode(pcm_wav(frames, rate), WAV, max_seconds=30)
    assert (audio.sample_rate, audio.samples.shape) == (rate, (signal.shape[0], channels))

    samples = to_model_input(audio.samples, audio.sample_rate)
    assert samples.dtype == np.float32 and samples.ndim == 1

    # The duration is kept to a sample, the pitch and the level as well
    assert abs(samples.shape[0] - seconds * 16000) <= 1
    assert abs(dominant_frequency(samples, 16000) - 440) < 2
    middle = samples[1600:-1600]
    assert abs(np.sqrt(np.mean(middle ** 2)) - 0.5 / np.sqrt(2)) < 0.01


@pytest.mark.parametrize("rate", [22050, 44100, 48000])
def test_downsampling_removes_frequencies_above_8k(rate):
    # A 10 kHz tone has no place at 16 kHz: it must not fold back as 6 kHz
    samples = resample(tone(10000, 1.0, rate), rate, 16000)
    assert np.sqrt(np.mean(samples[800:-800] ** 2)) < 0.01


def test_upsampling_keeps_the_signal():
    original = tone(1000, 1.0, 16000)
    restored = resample(resample(original, 16000, 8000), 8000, 16000)
    assert restored.shape == original.shape
    assert np.max(np.abs(restored[400:-400] - original[400:-400])) < 0.01


def test_stereo_to_mono_is_mean():
    left, right = tone(300, 0.5, 16000), tone(700, 0.5, 16000)
    mono = to_mono(np.stack([left, right], axis=1))
    np.testing.assert_allclose(mono, (left + right) / 2, atol=1e-6)


def test_16k_mono_passes_unchanged():
    signal = tone(440, 0.5, 16000)
    np.testing.assert_array_equal(to_model_input(signal[:, np.newaxis], 16000), signal)


def test_short_audio_resamples():
    for count in (1, 2, 3, 10):
        assert resample(np.ones(count, dtype=np.float32), 48000, 16000).shape[0] == round(count / 3)
        assert resample(np.ones(count, dtype=np.float32), 8000, 16000).shape[0] == count * 2


@pytest.mark.parametrize("width", [1, 2, 3, 4], ids=["8bit", "16bit", "24bit", "32bit"])
def test_pcm_widths_decode(width):
    signal = tone(440, 0.2, 16000)
    audio = decode(pcm_wav(signal[:, np.newaxis], 16000, width), WAV, max_seconds=30)
    tolerance = 1 / 64 if width == 1 else 1e-4
    np.testing.assert_allclose(audio.samples[:, 0], signal, atol=tolerance)


def test_wav_extra_chunks_and_extensible_format():
    signal = tone(440, 0.2, 16000)
    data = (signal * 32767).astype("<i2").tobytes()
    # WAVE_FORMAT_EXTENSIBLE with the PCM subformat, a LIST chunk before the data
    subformat = struct.pack("<H", 1) + bytes.fromhex("000000001000800000aa00389b71")
    fmt = struct.pack("<HHIIHHHHI", 0xFFFE, 1, 16000, 32000, 2, 16, 22, 16, 4) + subformat
    info = b"INFOISFT\x05\x00\x00\x00Lavf\x00\x00"
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"LIST" + struct.pack("<I", len(info)) + info \
        + b"data" + struct.pack("<I", len(data)) + data
    body = b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE" + chunks

    audio = decode(body, WAV, max_seconds=30)
    np.testing.assert_allclose(audio.samples[:, 0], signal, atol=1e-4)


def test_l16_decode():
    signal = tone(440, 0.5, 16000)
    stereo = np.stack([signal, -signal], axis=1)
    body = (stereo * 32767).astype("<i2").tobytes()

    media = parse_media_type("audio/L16; rate=16000; channels=2")
    assert (media.format, media.sample_rate, media.channels) == (AudioFormat.L16, 16000, 2)
    audio = decode(body, media, max_seconds=30)
    np.testing.assert_allclose(audio.samples, stereo, atol=1e-4)
    assert audio.duration_s == 0.5


def test_decode_errors():
    with pytest.raises(InvalidInputError, match="Audio is empty"):
        decode(b"", WAV, max_seconds=30)
    with pytest.raises(InvalidInputError, match="Audio could not be decoded"):
        decode(b"RIFF\x00\x00\x00\x00WAVEdata", WAV, max_seconds=30)
    with pytest.raises(InputTooLargeError, match="Audio is longer than 1 seconds"):
        decode(pcm_wav(np.zeros((16001, 1)), 16000), WAV, max_seconds=1)
    with pytest.raises(UnsupportedMediaTypeError):
        parse_media_type("audio/ogg")
