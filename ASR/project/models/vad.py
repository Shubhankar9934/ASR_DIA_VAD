import logging
from pyannote.audio import Pipeline
from project.config import HF_TOKEN, VAD_MODEL_ID
from pydub import AudioSegment

class VADModel:
    def __init__(self):
        self.pipeline = self.load_model()

    def load_model(self):
        logging.info("Loading VAD pipeline '%s'...", VAD_MODEL_ID)
        vad_pipeline = Pipeline.from_pretrained(
            VAD_MODEL_ID,
            use_auth_token=HF_TOKEN
        )
        logging.info("VAD pipeline loaded successfully.")
        return vad_pipeline

    def apply_vad(self, wav_file: str):
        logging.info("Applying VAD on '%s'...", wav_file)
        result = self.pipeline(wav_file)
        audio = AudioSegment.from_file(wav_file)
        duration_sec = len(audio) / 1000.0
        segments = []
        for segment, _, _ in result.itertracks(yield_label=True):
            segments.append({
                "type": "speech",
                "start": segment.start,
                "end": segment.end
            })
        if not segments:
            segments.append({
                "type": "silence",
                "start": 0.0,
                "end": duration_sec
            })
        # Fill gaps between speech segments with silence.
        segments.sort(key=lambda x: x["start"])
        final_segments = []
        if segments and segments[0]["start"] > 0:
            final_segments.append({
                "type": "silence",
                "start": 0.0,
                "end": segments[0]["start"]
            })
        for i, seg in enumerate(segments):
            final_segments.append(seg)
            if i < len(segments) - 1:
                if segments[i+1]["start"] > seg["end"]:
                    final_segments.append({
                        "type": "silence",
                        "start": seg["end"],
                        "end": segments[i+1]["start"]
                    })
        if segments and segments[-1]["end"] < duration_sec:
            final_segments.append({
                "type": "silence",
                "start": segments[-1]["end"],
                "end": duration_sec
            })
        logging.info("VAD produced %d segments.", len(final_segments))
        return final_segments
