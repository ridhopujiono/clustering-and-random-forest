from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback
import sys
import os
import pandas as pd
import json

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



@app.route('/map-data', methods=['GET'])
def get_map_data():
    # 1. PERSIAPAN DATA (DILAKUKAN SEKALI SAAT API DIMULAI)
    print(">>> Memuat dan mempersiapkan data untuk API...")

    # Memuat data hasil clustering
    try:
        df_dominant = pd.read_csv('data/profil_dominan_desa.csv')
        df_percentage = pd.read_csv('data/profil_persentase_desa.csv')
        # Menggabungkan dua dataframe menjadi satu berdasarkan desa_kel
        df_map_data = pd.merge(df_dominant, df_percentage, on='desa_kel')
    except FileNotFoundError:
        print("Error: Pastikan file 'profil_dominan_desa.csv' dan 'profil_persentase_desa.csv' ada di folder 'data/'.")
        exit()

    # Memuat data GeoJSON
    try:
        with open('geojson/all.json', 'r') as f:
            geojson_data = json.load(f)
    except FileNotFoundError:
        print("Error: Pastikan file 'all.json' ada di dalam folder 'geojson/'.")
        exit()

    # Membuat "peta" atau dictionary dari GeoJSON agar mudah dicari berdasarkan nama desa
    # Ini jauh lebih cepat daripada mencari di dalam list setiap kali ada request
    geojson_map = {item['name']: item['geojson'] for item in geojson_data}

    """
    Endpoint untuk menyajikan data clustering yang sudah digabung dengan GeoJSON.
    """
    # Mengubah dataframe pandas menjadi format list of dictionaries
    data_list = df_map_data.to_dict(orient='records')
    
    # Menambahkan data geojson ke setiap entri desa
    for item in data_list:
        desa_name = item['desa_kel']
        # Mengambil geojson dari 'peta' yang sudah kita buat
        item['geojson'] = geojson_map.get(desa_name, None) # Beri None jika tidak ketemu

    # Mengembalikan data dalam format JSON
    return jsonify(data_list)
    
if __name__ == '__main__':
    app.run(debug=True)
