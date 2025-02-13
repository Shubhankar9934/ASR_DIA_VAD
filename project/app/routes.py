import uuid
import threading
import logging
from flask import Blueprint, request, jsonify
from project.config import PROCESSING_MODE
from project.processors.audio_processor import AudioProcessor, generate_output_json, format_segments

bp = Blueprint('api', __name__)
jobs = {}

def process_job(job_id, file_path, mode):
    try:
        processor = AudioProcessor(file_path, mode)
        segments = processor.process()
        formatted = format_segments(segments)
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = formatted
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        logging.error("Job %s failed: %s", job_id, str(e))

@bp.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    mode = request.form.get("mode", PROCESSING_MODE).lower()
    if mode not in ("diarization", "transcription", "combined"):
        return jsonify({"error": "Invalid mode. Choose 'diarization', 'transcription', or 'combined'."}), 400

    job_id = str(uuid.uuid4())
    temp_filename = f"{job_id}_{file.filename}"
    file.save(temp_filename)
    jobs[job_id] = {"status": "processing", "result": None, "error": None}

    thread = threading.Thread(target=process_job, args=(job_id, temp_filename, mode))
    thread.start()

    return jsonify({"job_id": job_id, "message": f"Processing started in '{mode}' mode."}), 200

@bp.route("/result/<job_id>", methods=["GET"])
def get_result(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Invalid job ID"}), 404
    return jsonify(jobs[job_id])
