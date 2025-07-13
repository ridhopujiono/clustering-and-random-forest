from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import sys
import os

# Impor fungsi utama dan model dari file predict.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from predict import proses_input_anak
app = Flask(__name__)

CORS(app)  # Mengizinkan akses dari berbagai origin (opsional, jika frontend beda domain)

@app.route('/')
def index():
    return jsonify({"message": "Stunting Prediction API is running."})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data JSON dari body request
        data = request.json

        # Validasi input (pastikan field penting ada)
        required_fields = ['jk', 'usia', 'berat', 'tinggi']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f"Missing required field: {field}"}), 400

        # Jalankan prediksi
        hasil = proses_input_anak({
            'jk': data['jk'],
            'usia': data['usia'],
            'berat': data['berat'],
            'tinggi': data['tinggi']
        })
        return jsonify(hasil)

    
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e), 'line': traceback.format_exc()}), 500

if __name__ == '__main__':
    app.run(debug=True)
