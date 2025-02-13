import logging

def align_transcription_with_diarization(transcript_chunks, diarization_result):
    """
    For each transcription chunk, compute its midpoint and assign a speaker
    based on the diarization result (using pyannote's itertracks interface).
    Returns a list of transcription chunks with an added 'speaker' field.
    """
    aligned_chunks = []
    for chunk in transcript_chunks:
        ts = chunk.get("timestamp", (0.0, 0.0))
        midpoint = (ts[0] + ts[1]) / 2
        assigned_speaker = "unknown"
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            if turn.start <= midpoint <= turn.end:
                assigned_speaker = speaker
                break
        new_chunk = dict(chunk)
        new_chunk["speaker"] = assigned_speaker
        aligned_chunks.append(new_chunk)
    logging.info("Aligned %d transcription chunks with diarization.", len(aligned_chunks))
    return aligned_chunks
