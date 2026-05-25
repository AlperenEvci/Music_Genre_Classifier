# flask_api/app.py
"""
Flask Fallback API — Plan C
Kullanım: python -m flask_api.app
Endpoint: POST http://127.0.0.1:5000/predict
"""
import os
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify
from scripts.predict import predict_file

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB


@app.route('/predict', methods=['POST'])
def predict():
    """
    Audio dosyasını kabul eder, tahmin yap, JSON döndür.

    Expected multipart form-data:
      - audio: binary file (.wav veya .mp3)

    Response:
      {
        "genre": "rock",
        "confidence": 0.85,
        "recommendations": [...],
        "features": {...},
        "error": null
      }
    """
    if 'audio' not in request.files:
        return jsonify({
            'error': 'audio alanı gerekli',
            'genre': None,
            'confidence': 0.0,
            'recommendations': []
        }), 400

    audio_file = request.files['audio']
    ext = Path(audio_file.filename).suffix.lower() if audio_file.filename else ''

    if ext not in ('.wav', '.mp3'):
        return jsonify({
            'error': 'Yalnızca .wav/.mp3 kabul edilir',
            'genre': None,
            'confidence': 0.0,
            'recommendations': []
        }), 422

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        result = predict_file(tmp_path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return jsonify(result)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
