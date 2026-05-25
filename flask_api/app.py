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

# stdlib magic bytes kontrolü — python-magic bağımlılığı gerektirmez
AUDIO_MAGIC_BYTES = {
    b'RIFF': 'audio/wav',       # WAV
    b'\xff\xfb': 'audio/mpeg',  # MP3
    b'\xff\xf3': 'audio/mpeg',  # MP3
    b'\xff\xf2': 'audio/mpeg',  # MP3
    b'ID3\x02': 'audio/mpeg',   # MP3 ID3v2.2
    b'ID3\x03': 'audio/mpeg',   # MP3 ID3v2.3
    b'ID3\x04': 'audio/mpeg',   # MP3 ID3v2.4
}


def _check_audio_magic(file_path: str) -> bool:
    """İlk 4 byte ile ses dosyası magic number kontrolü."""
    try:
        with open(file_path, 'rb') as f:
            header = f.read(4)
        for magic in AUDIO_MAGIC_BYTES:
            if header.startswith(magic):
                return True
        return False
    except OSError:
        return False

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

        # Magic byte kontrolü — uzantı sahteciliğini engeller
        if not _check_audio_magic(tmp_path):
            os.unlink(tmp_path)
            tmp_path = None
            return jsonify({
                'error': 'Geçersiz ses dosyası',
                'genre': None,
                'confidence': 0.0,
                'recommendations': []
            }), 422

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
