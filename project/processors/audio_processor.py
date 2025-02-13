import os
import logging
import threading
import json
from project.config import SAMPLE_RATE, SELECTED_WHISPER_MODEL, DEVICE, TEMP_OUTPUT_DIR
from project.utils.file_utils import convert_to_wav, remove_file, dump_temp, write_rttm
from project.models.asr import ASRModel
from project.models.diarization import DiarizationModel
from project.processors.aligner import align_transcription_with_diarization

class AudioProcessor:
    def __init__(self, file_path: str, mode: str = "combined"):
        self.file_path = file_path
        self.mode = mode  # "diarization", "transcription", or "combined"
        self.asr_model = ASRModel(SELECTED_WHISPER_MODEL)
        self.diar_model = DiarizationModel()
        self.diar_result = None
        self.transcript_chunks = None

    def process(self):
        base_name = os.path.splitext(self.file_path)[0]
        wav_file = base_name + "_16k.wav"
        convert_to_wav(self.file_path, wav_file, SAMPLE_RATE)

        output_segments = []

        if self.mode == "diarization":
            diar_result = self.diar_model.diarize(wav_file)
            dump_temp(str(diar_result), "diar_result.txt")
            rttm_path = os.path.join(TEMP_OUTPUT_DIR, os.path.basename(wav_file).replace("_16k.wav", ".rttm"))
            write_rttm(diar_result, rttm_path, file_id=os.path.basename(wav_file))
            dump_temp("RTTM written to: " + rttm_path, "rttm_info.txt")
            for turn, _, speaker in diar_result.itertracks(yield_label=True):
                output_segments.append({
                    "type": "speech",
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                    "text": ""
                })
        elif self.mode == "transcription":
            transcript_chunks = self.asr_model.transcribe(wav_file, english_only=(SELECTED_WHISPER_MODEL=="distil"))
            dump_temp(json.dumps(transcript_chunks, indent=2), "transcript_chunks.json")
            for chunk in transcript_chunks:
                chunk_start, chunk_end = chunk.get("timestamp", (0.0, 0.0))
                output_segments.append({
                    "type": "speech",
                    "start": chunk_start,
                    "end": chunk_end,
                    "speaker": "",
                    "text": chunk.get("text", "")
                })
        elif self.mode == "combined":
            threads = []
            def run_diar():
                self.diar_result = self.diar_model.diarize(wav_file)
            def run_transcript():
                self.transcript_chunks = self.asr_model.transcribe(wav_file, english_only=(SELECTED_WHISPER_MODEL=="distil"))
            t1 = threading.Thread(target=run_diar)
            t2 = threading.Thread(target=run_transcript)
            threads.extend([t1, t2])
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            dump_temp(str(self.diar_result), "diar_result.txt")
            dump_temp(json.dumps(self.transcript_chunks, indent=2), "transcript_chunks.json")
            # Write RTTM file.
            rttm_path = os.path.join(TEMP_OUTPUT_DIR, os.path.basename(wav_file).replace("_16k.wav", ".rttm"))
            write_rttm(self.diar_result, rttm_path, file_id=os.path.basename(wav_file))
            dump_temp("RTTM written to: " + rttm_path, "rttm_info.txt")
            # Align transcription chunks with diarization.
            aligned_transcript = align_transcription_with_diarization(self.transcript_chunks, self.diar_result)
            dump_temp(json.dumps(aligned_transcript, indent=2), "aligned_transcript.json")
            # Preserve all transcription text by using aligned transcript chunks.
            # First, sort aligned_transcript by their start timestamp.
            aligned_transcript.sort(key=lambda x: x.get("timestamp", (0.0,))[0])
            # Generate silence segments between transcription chunks.
            final_segments = []
            if aligned_transcript:
                first_start = aligned_transcript[0].get("timestamp", (0.0,))[0]
                if first_start > 0:
                    final_segments.append({
                        "type": "silence",
                        "start": 0.0,
                        "end": first_start
                    })
                for i in range(len(aligned_transcript)):
                    chunk = aligned_transcript[i]
                    c_start, c_end = chunk.get("timestamp", (0.0, 0.0))
                    final_segments.append({
                        "type": "speech",
                        "start": c_start,
                        "end": c_end,
                        "speaker": chunk.get("speaker", "unknown"),
                        "text": chunk.get("text", "").strip()
                    })
                    if i < len(aligned_transcript) - 1:
                        next_start = aligned_transcript[i+1].get("timestamp", (0.0,))[0]
                        if next_start > c_end:
                            final_segments.append({
                                "type": "silence",
                                "start": c_end,
                                "end": next_start
                            })
                # Add trailing silence if needed.
                from pydub import AudioSegment
                audio = AudioSegment.from_file(wav_file)
                duration_sec = len(audio) / 1000.0
                if aligned_transcript[-1].get("timestamp", (0.0, 0.0))[1] < duration_sec:
                    final_segments.append({
                        "type": "silence",
                        "start": aligned_transcript[-1].get("timestamp", (0.0, 0.0))[1],
                        "end": duration_sec
                    })
            else:
                # If no transcription is present, mark entire audio as silence.
                from pydub import AudioSegment
                audio = AudioSegment.from_file(wav_file)
                duration_sec = len(audio) / 1000.0
                final_segments.append({
                    "type": "silence",
                    "start": 0.0,
                    "end": duration_sec
                })
            output_segments = final_segments
            dump_temp(json.dumps(output_segments, indent=2), "final_alined_transcript.json")
        else:
            raise ValueError("Invalid processing mode.")

        remove_file(wav_file)
        remove_file(self.file_path)
        return output_segments

def format_segments(segments):
    def map_speaker(label):
        return label or ""
    formatted_segments = []
    for seg in segments:
        formatted_seg = {
            "type": seg.get("type"),
            "start": round(seg.get("start", 0.0), 2),
            "end": round(seg.get("end", 0.0), 2)
        }
        if seg.get("type") == "speech":
            formatted_seg["speaker"] = map_speaker(seg.get("speaker", ""))
            formatted_seg["text"] = seg.get("text", "").strip()
        formatted_segments.append(formatted_seg)
    return formatted_segments

def generate_output_json(segments):
    import json
    return json.dumps(format_segments(segments), indent=2)
