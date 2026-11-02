# Model and heuristic thresholds live here, not in env: they are tuned together with the code


class ENGINE_SETTINGS:
    """Inference engine shared by the services."""
    WARMUP_CALLS = 3                    # idle calls at startup: kernels, allocators and caches are ready before the first request


class AUDIO_SETTINGS:
    """Audio the recognition takes (ASR-1, ASR-2)."""
    MAX_BYTES = 4 * 1024 * 1024         # request body limit, 4 MiB; longer audio gets 413 without decoding
    MIN_SAMPLE_RATE = 8000              # telephone audio
    MAX_SAMPLE_RATE = 48000
    MAX_CHANNELS = 2                    # mono or stereo
    SAMPLE_RATE = 16000                 # every model and the speech detector take 16 kHz mono
    WARMUP_SECONDS = 3.0                # a typical question: the warmup input length


class VAD_SETTINGS:
    """Silero VAD thresholds: the defaults of its get_speech_timestamps."""
    THRESHOLD = 0.5                     # a window with a higher speech probability is speech
    NEG_THRESHOLD_GAP = 0.15            # speech ends below THRESHOLD - gap: no flapping around the threshold
    MIN_SPEECH_MS = 250                 # a shorter segment is a click or a cough, not a word
    MIN_SILENCE_MS = 100                # a shorter pause keeps the segment going
    SPEECH_PAD_MS = 100                 # kept around the speech: the trimmed edges do not cut the consonants
