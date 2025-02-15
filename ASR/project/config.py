import os

# Model configuration
DEFAULT_WHISPER_MODEL_ID = "openai/whisper-large-v3-turbo"
DISTIL_WHISPER_MODEL_ID = "distil-whisper/distil-large-v3"
#DIARIZATION_MODEL_ID = "pyannote/speaker-diarization@2.1"
DIARIZATION_MODEL_ID = "pyannote/speaker-diarization-3.1"
VAD_MODEL_ID = "pyannote/voice-activity-detection"  # Not used separately in combined mode

# Hugging Face access token (set via environment variable or fallback)
HF_TOKEN = os.getenv("HF_HUB_TOKEN", "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")

# Device configuration: use "cuda" if available, otherwise CPU
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# ASR model selection: "whisper" or "distil"
SELECTED_WHISPER_MODEL = "distil"

# Audio processing configuration
SAMPLE_RATE = 16000

# Processing mode: Allowed values: "diarization", "transcription", "combined"
PROCESSING_MODE = os.getenv("PROCESSING_MODE", "combined").lower()
if PROCESSING_MODE not in ("diarization", "transcription", "combined"):
    raise ValueError("Invalid PROCESSING_MODE. Choose from 'diarization', 'transcription', or 'combined'.")

# Processing type: "batch" or "realtime"
PROCESSING_TYPE = os.getenv("PROCESSING_TYPE", "batch").lower()

# (Optional) Auto-detect number of speakers (None means let the model decide)
AUTO_NUM_SPEAKERS = None

# Temporary output directory for debugging and RTTM file storage.
TEMP_OUTPUT_DIR = os.getenv("TEMP_OUTPUT_DIR", "temp_debug")
if not os.path.exists(TEMP_OUTPUT_DIR):
    os.makedirs(TEMP_OUTPUT_DIR)
