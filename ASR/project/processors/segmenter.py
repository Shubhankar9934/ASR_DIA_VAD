import logging
from project.models.vad import VADModel

def segment_audio(wav_file: str):
    """
    Use the VAD model to segment the audio into speech and silence segments.
    """
    vad = VADModel()
    segments = vad.apply_vad(wav_file)
    logging.info("Segmented audio into %d segments.", len(segments))
    return segments
