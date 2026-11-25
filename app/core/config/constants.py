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
    """Silero VAD thresholds; the speech threshold is below the Silero default of 0.5, the rest are its defaults."""
    THRESHOLD = 0.3                     # a window with a higher speech probability is speech; 0.5 cut the first vowel off a word in street noise
    NEG_THRESHOLD_GAP = 0.15            # speech ends below THRESHOLD - gap: no flapping around the threshold
    MIN_SPEECH_MS = 250                 # a shorter segment is a click or a cough, not a word
    MIN_SILENCE_MS = 100                # a shorter pause keeps the segment going
    SPEECH_PAD_MS = 100                 # kept around the speech: the trimmed edges do not cut the consonants


class TONE_SETTINGS:
    """T-one acoustic model: the constants of its ONNX export."""
    REPOSITORY = "t-tech/T-one"         # Hugging Face repository of the authors
    REVISION = "106f3b0b32a9e107eb613312e4ebc61ff3d53926"       # pinned: the answers do not change under a measurement
    FILENAME = "model.onnx"
    SAMPLE_RATE = 8000                  # telephony model: the 16 kHz input is resampled down
    CHUNK_SAMPLES = 2400                # 300 ms, the fixed input of the streaming model
    STATE_SIZE = 219729                 # flat state carried between the chunks
    LABELS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя "      # CTC tokens; indices past them are blank and service ones


class WHISPER_SETTINGS:
    """faster-whisper decoding (ASR-3)."""
    BEAM_SIZE = 1                       # greedy search: speed over the last percent of quality
    COMPUTE_TYPE_CPU = "int8"           # quantized weights: several times faster on CPU
    COMPUTE_TYPE_CUDA = "float16"


class VOSK_SETTINGS:
    """Vosk models: MODEL_NAME to the archive of the model list."""
    URL = "https://alphacephei.com/vosk/models"
    ARCHIVES = {
        "small-ru": "vosk-model-small-ru-0.22",        # 45 MB, for phones and small boards
        "ru": "vosk-model-ru-0.42",                    # 1.8 GB, for servers
    }


class GIGAAM_SETTINGS:
    """GigaAM loading."""
    FP16_ENCODER = True                 # half precision encoder on CUDA; on CPU the library keeps float32


class GIGAAM_ONNX_SETTINGS:
    """GigaAM-v3 exported to ONNX for onnx-asr: MODEL_NAME to the onnx-asr model and its files."""
    REPOSITORY = "istupakov/gigaam-v3-onnx"     # ONNX export of ai-sage/GigaAM-v3 that onnx-asr maps its gigaam-v3 models to
    REVISION = "322c3b29492673eb7d0b434bfa9dfb8653e34d02"       # pinned: the answers do not change under a measurement
    MODELS = {
        "v3_ctc": "gigaam-v3-ctc",
        "v3_rnnt": "gigaam-v3-rnnt",
    }
    # Full precision only: no *.int8.onnx; the e2e models are out of the scope
    FILES = {
        "v3_ctc": ("v3_ctc.onnx", "v3_ctc.yaml"),
        "v3_rnnt": ("v3_rnnt_encoder.onnx", "v3_rnnt_decoder.onnx", "v3_rnnt_joint.onnx", "v3_rnnt.yaml"),
    }
    COMMON_FILES = ("config.json", "v3_vocab.txt")     # config.json gives onnx-asr the v3 feature extractor
