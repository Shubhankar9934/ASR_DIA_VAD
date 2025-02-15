import logging
from pyannote.audio import Pipeline
from project.config import DIARIZATION_MODEL_ID, HF_TOKEN

class DiarizationModel:
    def __init__(self):
        self.pipeline = self.load_model()

    def load_model(self):
        logging.info("Loading diarization pipeline '%s'...", DIARIZATION_MODEL_ID)
        diar_pipeline = Pipeline.from_pretrained(
            DIARIZATION_MODEL_ID,
            use_auth_token=HF_TOKEN
        )
        logging.info("Diarization pipeline loaded successfully.")
        return diar_pipeline

    def diarize(self, wav_file: str):
        logging.info("Running diarization on '%s'...", wav_file)
        result = self.pipeline(wav_file)
        logging.info("Diarization completed.")
        return result
