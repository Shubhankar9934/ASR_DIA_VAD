import logging
from transformers import pipeline
from project.config import DEVICE, DEFAULT_WHISPER_MODEL_ID, DISTIL_WHISPER_MODEL_ID, SELECTED_WHISPER_MODEL

class ASRModel:
    def __init__(self, model_key: str = SELECTED_WHISPER_MODEL):
        self.model_key = model_key
        self.pipeline = self.load_model()

    def load_model(self):
        model_id = DEFAULT_WHISPER_MODEL_ID if self.model_key == "whisper" else DISTIL_WHISPER_MODEL_ID
        logging.info("Loading ASR pipeline '%s' on device %s...", model_id, DEVICE)
        asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            chunk_length_s=30,
            return_timestamps=True,
            device=DEVICE
        )
        logging.info("ASR pipeline for '%s' loaded successfully.", model_id)
        return asr_pipeline

    def transcribe(self, wav_file: str, english_only: bool = False):
        generate_kwargs = {} if english_only else {"language": "english"}
        logging.info("Starting ASR transcription on '%s'...", wav_file)
        result = self.pipeline(wav_file, generate_kwargs=generate_kwargs)
        if isinstance(result, list) and len(result) > 0:
            result = result[0]
        chunks = result.get("chunks", [])
        logging.info("Obtained %d transcription chunks.", len(chunks))
        return chunks
