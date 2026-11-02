"""
Request body to samples: WAV, raw audio/L16 or either of them in a multipart field `file` (ASR-1).

The body is already in memory: no temporary files on the hot path (SVC-8). The RIFF reader is
our own, so a format the service does not take is told from a corrupt file: the first is 415,
the second is 400.
"""

import struct

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from python_multipart.exceptions import FormParserError
from python_multipart.multipart import MultipartParser, parse_options_header

from app.core.config import AUDIO_SETTINGS as CFG
from app.core.exceptions import (
    InputTooLargeError,
    InvalidInputError,
    InvalidParamsError,
    UnsupportedMediaTypeError,
)

WAV_TYPES = frozenset({"audio/wav", "audio/x-wav", "audio/wave", "audio/vnd.wave"})
L16_TYPE = "audio/l16"
MULTIPART_TYPE = "multipart/form-data"
OCTET_STREAM_TYPE = "application/octet-stream"      # curl -F sends a .wav file as it
FILE_FIELD = "file"

UNSUPPORTED_TYPE = "Content-Type must be audio/wav, audio/L16 or multipart/form-data"
CORRUPT_AUDIO = "Audio could not be decoded"
EMPTY_AUDIO = "Audio is empty"

_WAVE_FORMAT_PCM = 0x0001
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE


class AudioFormat(StrEnum):
    WAV = "wav"
    L16 = "l16"
    MULTIPART = "multipart"


@dataclass(frozen=True)
class MediaType:
    """A Content-Type the service takes, with its parameters checked"""
    format: AudioFormat
    sample_rate: int = 0                # audio/L16 only
    channels: int = 0                   # audio/L16 only
    boundary: bytes = b""               # multipart only


@dataclass(frozen=True)
class Audio:
    """Decoded PCM: float32 samples in [-1, 1], one column per channel"""
    samples: np.ndarray
    sample_rate: int

    @property
    def duration_s(self) -> float:
        return self.samples.shape[0] / self.sample_rate


@dataclass
class _Part:
    headers: Dict[str, str] = field(default_factory=dict)
    chunks: List[bytes] = field(default_factory=list)


def parse_media_type(content_type: Optional[str]) -> MediaType:
    """The body format of a Content-Type; UnsupportedMediaTypeError for any other."""
    if not content_type:
        raise UnsupportedMediaTypeError(UNSUPPORTED_TYPE)

    raw_type, raw_params = parse_options_header(content_type)
    media_type = raw_type.decode("latin-1").strip().lower()
    params = {key.decode("latin-1").lower(): value.decode("latin-1") for key, value in raw_params.items()}

    if media_type in WAV_TYPES:
        return MediaType(AudioFormat.WAV)
    if media_type == L16_TYPE:
        return _l16_media_type(params)
    if media_type == MULTIPART_TYPE:
        if not params.get("boundary"):
            raise InvalidInputError("Multipart body has no boundary")
        return MediaType(AudioFormat.MULTIPART, boundary=params["boundary"].encode("latin-1"))
    raise UnsupportedMediaTypeError(UNSUPPORTED_TYPE)


def decode(body: bytes, media: MediaType, max_seconds: float) -> Audio:
    """Samples of the body; 400 for an empty or corrupt one, 413 when longer than `max_seconds`. Blocking."""
    if media.format is AudioFormat.MULTIPART:
        body, media = _multipart_file(body, media.boundary)

    if not body:
        raise InvalidInputError(EMPTY_AUDIO)
    if media.format is AudioFormat.L16:
        return _decode_l16(body, media, max_seconds)
    return _decode_wav(body, max_seconds)


def _l16_media_type(params: Dict[str, str]) -> MediaType:
    try:
        # RFC 2586 makes the rate mandatory and one channel the default
        sample_rate = int(params["rate"])
        channels = int(params.get("channels", "1"))
    except (KeyError, ValueError):
        raise UnsupportedMediaTypeError("audio/L16 needs an integer rate parameter") from None
    _check_layout(sample_rate, channels)
    return MediaType(AudioFormat.L16, sample_rate=sample_rate, channels=channels)


def _check_layout(sample_rate: int, channels: int) -> None:
    if not CFG.MIN_SAMPLE_RATE <= sample_rate <= CFG.MAX_SAMPLE_RATE:
        raise UnsupportedMediaTypeError(
            f"Sample rate {sample_rate} Hz is not supported, expected {CFG.MIN_SAMPLE_RATE}-{CFG.MAX_SAMPLE_RATE} Hz"
        )
    if not 1 <= channels <= CFG.MAX_CHANNELS:
        raise UnsupportedMediaTypeError(f"{channels} channels are not supported, expected mono or stereo")


def _check_duration(frames: int, sample_rate: int, max_seconds: float) -> None:
    if frames > max_seconds * sample_rate:
        raise InputTooLargeError(f"Audio is longer than {max_seconds:g} seconds")


def _decode_l16(body: bytes, media: MediaType, max_seconds: float) -> Audio:
    # PCM s16le as the gateway sends it
    frame_size = 2 * media.channels
    if len(body) % frame_size:
        raise InvalidInputError(CORRUPT_AUDIO)

    frames = len(body) // frame_size
    _check_duration(frames, media.sample_rate, max_seconds)
    samples = np.frombuffer(body, dtype="<i2").reshape(frames, media.channels)
    return Audio(samples.astype(np.float32) / 32768.0, media.sample_rate)


def _decode_wav(body: bytes, max_seconds: float) -> Audio:
    if len(body) < 12 or body[:4] != b"RIFF" or body[8:12] != b"WAVE":
        raise InvalidInputError(CORRUPT_AUDIO)

    # 1. Walk the chunks: fmt describes the samples, data holds them; the rest is skipped
    layout, data, position = None, None, 12
    while position + 8 <= len(body) and data is None:
        chunk_id, size = struct.unpack_from("<4sI", body, position)
        start = position + 8
        if chunk_id == b"fmt ":
            layout = _parse_fmt(body[start:start + size])
        elif chunk_id == b"data":
            # A streamed WAV may declare a wrong size: take what is there
            data = body[start:start + size]
        position = start + size + (size & 1)

    if layout is None or data is None:
        raise InvalidInputError(CORRUPT_AUDIO)

    # 2. Whole frames only: a truncated file loses its last partial frame
    sample_rate, channels, width = layout
    frames = len(data) // (channels * width)
    if frames == 0:
        raise InvalidInputError(EMPTY_AUDIO)
    _check_duration(frames, sample_rate, max_seconds)

    samples = _PCM_DECODERS[width](data[:frames * channels * width])
    return Audio(samples.reshape(frames, channels), sample_rate)


def _parse_fmt(chunk: bytes) -> Tuple[int, int, int]:
    if len(chunk) < 16:
        raise InvalidInputError(CORRUPT_AUDIO)

    format_tag, channels, sample_rate, _, _, bits = struct.unpack_from("<HHIIHH", chunk)
    if format_tag == _WAVE_FORMAT_EXTENSIBLE and len(chunk) >= 26:
        # The real format is the first two bytes of the subformat GUID
        format_tag = struct.unpack_from("<H", chunk, 24)[0]
    if format_tag != _WAVE_FORMAT_PCM:
        raise UnsupportedMediaTypeError("Only PCM WAV is supported")
    if bits not in (8, 16, 24, 32):
        raise UnsupportedMediaTypeError(f"{bits}-bit WAV is not supported, expected 8, 16, 24 or 32 bits")
    if channels == 0 or sample_rate == 0:
        raise InvalidInputError(CORRUPT_AUDIO)

    _check_layout(sample_rate, channels)
    return sample_rate, channels, bits // 8


def _pcm8(data: bytes) -> np.ndarray:
    # 8-bit WAV is unsigned
    return (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0


def _pcm16(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def _pcm24(data: bytes) -> np.ndarray:
    triples = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
    values = triples[:, 0] | (triples[:, 1] << 8) | (triples[:, 2] << 16)
    values = np.where(values >= 1 << 23, values - (1 << 24), values)
    return values.astype(np.float32) / float(1 << 23)


def _pcm32(data: bytes) -> np.ndarray:
    return (np.frombuffer(data, dtype="<i4").astype(np.float64) / float(1 << 31)).astype(np.float32)


_PCM_DECODERS: Dict[int, Callable[[bytes], np.ndarray]] = {1: _pcm8, 2: _pcm16, 3: _pcm24, 4: _pcm32}


def _multipart_file(body: bytes, boundary: bytes) -> Tuple[bytes, MediaType]:
    """Content and media type of the multipart field `file`."""
    parts: List[_Part] = []
    header_name: List[bytes] = []
    header_value: List[bytes] = []

    def on_header_end() -> None:
        name = b"".join(header_name).decode("latin-1").lower()
        parts[-1].headers[name] = b"".join(header_value).decode("latin-1")
        header_name.clear()
        header_value.clear()

    callbacks = {
        "on_part_begin": lambda: parts.append(_Part()),
        "on_part_data": lambda data, start, end: parts[-1].chunks.append(data[start:end]),
        "on_header_field": lambda data, start, end: header_name.append(data[start:end]),
        "on_header_value": lambda data, start, end: header_value.append(data[start:end]),
        "on_header_end": on_header_end,
    }
    try:
        parser = MultipartParser(boundary, callbacks)
        parser.write(body)
        parser.finalize()
    except FormParserError as e:
        raise InvalidInputError("Multipart body could not be parsed") from e

    for part in parts:
        _, disposition = parse_options_header(part.headers.get("content-disposition", ""))
        if disposition.get(b"name") == FILE_FIELD.encode():
            return b"".join(part.chunks), _part_media_type(part.headers.get("content-type"))
    raise InvalidParamsError(f"Multipart field '{FILE_FIELD}' is missing")


def _part_media_type(content_type: Optional[str]) -> MediaType:
    # A file without its own type is taken for WAV: the RIFF header decides further
    if not content_type or parse_options_header(content_type)[0].decode("latin-1").lower() == OCTET_STREAM_TYPE:
        return MediaType(AudioFormat.WAV)

    media = parse_media_type(content_type)
    if media.format is AudioFormat.MULTIPART:
        raise UnsupportedMediaTypeError(UNSUPPORTED_TYPE)
    return media
