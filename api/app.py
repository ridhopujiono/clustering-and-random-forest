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
    
# --- ENDPOINT BARU UNTUK RINGKASAN CLUSTER ---
@app.route('/cluster-summary', methods=['GET'])
def get_cluster_summary():
    """
    Endpoint untuk menyajikan data karakteristik dan distribusi setiap cluster.
    """
    # --- DATA BARU UNTUK RINGKASAN CLUSTER ---
    try:
        df_means = pd.read_csv('data/cluster_summary_means.csv').set_index('cluster')
        df_dist = pd.read_csv('data/cluster_summary_distribution.csv').set_index('cluster')
    except FileNotFoundError:
        print("Error: Pastikan file 'cluster_summary_means.csv' dan 'cluster_summary_distribution.csv' sudah ada.")
        df_means, df_dist = (pd.DataFrame(), pd.DataFrame()) # Buat dataframe kosong jika file tidak ada

    print(">>> Data siap disajikan oleh API.")
    summary_data = {}
    # Loop melalui setiap cluster yang ada di index dataframe
    for cluster_id in df_means.index:
        summary_data[cluster_id] = {
            'karakteristik': df_means.loc[cluster_id].round(2).to_dict(),
            'distribusi': df_dist.loc[cluster_id].round(4).to_dict()
        }
    return jsonify(summary_data)


@app.route('/clustering-dashboard', methods=['GET'])
def get_clustering_dashboard():
    """
    Endpoint baru untuk menyajikan SEMUA data ringkasan untuk dashboard clustering.
    """
    try:
        # Membaca semua file CSV yang dibutuhkan
        df_means = pd.read_csv('data/cluster_summary_means.csv')
        df_dist = pd.read_csv('data/cluster_summary_distribution.csv')
        df_dominant = pd.read_csv('data/profil_dominan_desa.csv')
        df_percentage = pd.read_csv('data/profil_persentase_desa.csv')

        # Mengubah setiap dataframe menjadi format JSON yang sesuai
        response = {
            'karakteristik_cluster': df_means.to_dict(orient='records'),
            'distribusi_stunting': df_dist.to_dict(orient='records'),
            'profil_dominan_desa': df_dominant.to_dict(orient='records'),
            'persentase_komposisi_desa': df_percentage.to_dict(orient='records')
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": f"Gagal memuat data ringkasan: {str(e)}"}), 500

@app.route('/dashboard-summary', methods=['GET'])
def get_dashboard_summary():
    try:
        # Kita hanya butuh satu file ini yang berisi semua informasi
        df_lengkap = pd.read_csv('data/data_lengkap_teranalisis.csv')
    except Exception as e:
        print(f"Error memuat data_lengkap_teranalisis.csv: {e}")
        df_lengkap = pd.DataFrame() # Buat dataframe kosong jika gagal

    if df_lengkap.empty:
        return jsonify({"error": "Data tidak tersedia"}), 500

    # 1. Hitung KPI Cards
    total_anak = len(df_lengkap)
    jumlah_desa = df_lengkap['desa_kel'].nunique()
    
    jumlah_stunting = len(df_lengkap[df_lengkap['status_stunting'] == 'STUNTING'])
    prevalensi_stunting = (jumlah_stunting / total_anak) * 100 if total_anak > 0 else 0
    
    jumlah_berisiko = len(df_lengkap[df_lengkap['status_stunting'] == 'RESIKO STUNTING'])

    # 2. Siapkan data untuk Grafik Ringkasan
    distribusi_status = df_lengkap['status_stunting'].value_counts().to_dict()
    distribusi_cluster = df_lengkap['cluster'].value_counts().sort_index().to_dict()
    
    # 3. Hitung Peringkat Wilayah Berisiko
    # Hitung total anak per desa
    total_per_desa = df_lengkap['desa_kel'].value_counts()
    # Hitung anak stunting + berisiko per desa
    df_risiko = df_lengkap[df_lengkap['status_stunting'].isin(['STUNTING', 'RESIKO STUNTING'])]
    risiko_per_desa = df_risiko['desa_kel'].value_counts()
    # Hitung persentase dan ambil 5 teratas
    persen_risiko_desa = (risiko_per_desa / total_per_desa * 100).fillna(0).sort_values(ascending=False).head(5)
    top_5_desa = persen_risiko_desa.round(2).to_dict()

    # Gabungkan semua dalam satu response JSON
    response = {
        'kpi': {
            'total_anak': total_anak,
            'prevalensi_stunting': round(prevalensi_stunting, 2),
            'jumlah_berisiko': jumlah_berisiko,
            'jumlah_desa': jumlah_desa
        },
        'charts': {
            'distribusi_status': distribusi_status,
            'distribusi_cluster': distribusi_cluster,
            'top_5_desa_risiko': top_5_desa
        }
    }
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)
