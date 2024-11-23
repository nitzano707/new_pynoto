from flask import Flask, request, jsonify
from pyannote.audio import Pipeline
import os

app = Flask(__name__)

# אתחול המודל של pyannote.audio עם Hugging Face Token
HUGGING_FACE_TOKEN = os.getenv("HF_TOKEN")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=HUGGING_FACE_TOKEN)

@app.route('/diarize', methods=['POST'])
def diarize():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    # שמירת קובץ האודיו
    audio_file = request.files['audio']
    file_path = f"/tmp/{audio_file.filename}"
    audio_file.save(file_path)

    # הפעלת המודל
    diarization = pipeline(file_path)
    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
