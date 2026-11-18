"""
Silero VAD on onnxruntime, without torch.

The weights `silero_vad.onnx` (v6.2.3, MIT, see LICENSE next to them) live in the repository:
the detector is a part of the core and runs in the fast tests. The window loop follows
`OnnxWrapper` and `get_speech_timestamps` of the silero-vad package.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort

from loguru import logger

from app.core.config import AUDIO_SETTINGS, VAD_SETTINGS as CFG

MODEL_PATH = Path(__file__).resolve().parent / "silero_vad.onnx"

WINDOW = 512                            # samples per call at 16 kHz, 32 ms
CONTEXT = 64                            # tail of the previous window the model sees with the next one
STATE_SHAPE = (2, 1, 128)


class SileroVad:
    """Finds the speech in 16 kHz mono audio; safe to share between threads"""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        options = ort.SessionOptions()
        # One window is tiny: more threads only add the synchronization cost
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        self._sample_rate = np.array(AUDIO_SETTINGS.SAMPLE_RATE, dtype=np.int64)
        logger.info("Silero VAD model ready: '{}'", model_path.name)

    def probabilities(self, audio: np.ndarray) -> np.ndarray:
        """Speech probability of every 32 ms window. Blocking: call it from a worker thread."""
        count = -(-audio.shape[0] // WINDOW)
        padded = np.zeros(count * WINDOW + CONTEXT, dtype=np.float32)
        padded[CONTEXT:CONTEXT + audio.shape[0]] = audio

        # The state is local to the call: one session serves all the worker threads
        state = np.zeros(STATE_SHAPE, dtype=np.float32)
        probabilities = np.empty(count, dtype=np.float32)
        for index in range(count):
            start = index * WINDOW
            window = padded[start:start + CONTEXT + WINDOW][np.newaxis, :]
            output, state = self._session.run(None, {"input": window, "state": state, "sr": self._sample_rate})
            probabilities[index] = output[0, 0]
        return probabilities

    def speech_segments(self, audio: np.ndarray) -> List[Tuple[int, int]]:
        """Speech segments as (start, end) samples, without padding."""
        rate = AUDIO_SETTINGS.SAMPLE_RATE
        min_speech = rate * CFG.MIN_SPEECH_MS // 1000
        min_silence = rate * CFG.MIN_SILENCE_MS // 1000
        neg_threshold = CFG.THRESHOLD - CFG.NEG_THRESHOLD_GAP

        segments: List[Tuple[int, int]] = []
        triggered, start, silence_from = False, 0, None
        for index, probability in enumerate(self.probabilities(audio)):
            position = index * WINDOW
            if probability >= CFG.THRESHOLD:
                silence_from = None
                if not triggered:
                    triggered, start = True, position
                continue
            if triggered and probability < neg_threshold:
                if silence_from is None:
                    silence_from = position
                if position - silence_from < min_silence:
                    continue
                if silence_from - start >= min_speech:
                    segments.append((start, silence_from))
                triggered, silence_from = False, None

        if triggered and audio.shape[0] - start >= min_speech:
            segments.append((start, audio.shape[0]))
        return segments

    def trim(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """The audio from the first speech to the last one with a pad; None when there is no speech."""
        segments = self.speech_segments(audio)
        if not segments:
            return None

        pad = AUDIO_SETTINGS.SAMPLE_RATE * CFG.SPEECH_PAD_MS // 1000
        start = max(segments[0][0] - pad, 0)
        end = min(segments[-1][1] + pad, audio.shape[0])
        return audio[start:end]
