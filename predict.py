import joblib
import pandas as pd
import numpy as np

# Impor fungsi kustom kita dari folder lib
try:
    from lib.zscore import hitung_zscore_balita
except ImportError:
    print("Error: Pastikan file 'zscore.py' ada di dalam folder 'lib/'.")
    exit()

# ==============================================================================
# 1. MEMUAT SEMUA MODEL DAN KOMPONEN
# ==============================================================================
try:
    print(">>> Memuat semua model dan komponen...")
    # PERBAIKAN: Menggunakan nama file dari script training terakhir
    rf_model = joblib.load('models/rf_realistis.pkl')
    kmeans_model = joblib.load('models/kmeans_final.pkl')
    scaler = joblib.load('models/scaler_final.pkl')
    le_stunting = joblib.load('models/label_encoder_multiclass.pkl')
    model_columns = joblib.load('models/model_columns_realistis.pkl')
    print("Semua komponen berhasil dimuat.\n")
except FileNotFoundError:
    print("Error: Pastikan semua file model (.pkl) ada di dalam folder 'models/'. Jalankan script training terakhir terlebih dahulu.")
    exit()

# ==============================================================================
# 2. FUNGSI HELPER
# ==============================================================================
def tentukan_status_gizi(zs_bb_tb):
    """Menentukan status gizi berdasarkan Z-score BB/TB."""
    if zs_bb_tb is None: return 'Tidak Dapat Dihitung'
    if zs_bb_tb < -3: return 'GIZI BURUK'
    elif zs_bb_tb < -2: return 'GIZI KURANG'
    elif zs_bb_tb <= 1: return 'GIZI BAIK'
    elif zs_bb_tb <= 2: return 'RESIKO GIZI LEBIH'
    elif zs_bb_tb <= 3: return 'GIZI LEBIH'
    else: return 'OBESITAS'

# ==============================================================================
# 3. FUNGSI UTAMA UNTUK PREDIKSI & ANALISIS
# ==============================================================================
def proses_input_anak(data_anak):
    """Fungsi lengkap untuk melakukan prediksi dan analisis dari input dasar."""
    df_baru = pd.DataFrame([data_anak])

    # --- Bagian 1: PREDIKSI STUNTING DENGAN RANDOM FOREST ---
    # Pra-pemrosesan hanya dengan fitur yang digunakan saat training
    df_processed = pd.get_dummies(df_baru, columns=['jk'], drop_first=True)
    df_aligned = df_processed.reindex(columns=model_columns, fill_value=0)

    # Lakukan Prediksi Stunting
    prediksi_kode = rf_model.predict(df_aligned)
    prediksi_label = le_stunting.inverse_transform(prediksi_kode)[0]
    prediksi_proba = rf_model.predict_proba(df_aligned)[0]

    # --- Bagian 2: ANALISIS TAMBAHAN (Z-SCORE, STATUS GIZI, CLUSTER) ---
    zscores = hitung_zscore_balita(
        jk=data_anak['jk'],
        usia_bulan=data_anak['usia'],
        berat_kg=data_anak['berat'],
        tinggi_cm=data_anak['tinggi']
    )
    
    status_gizi_label = "N/A"
    prediksi_cluster = "N/A"

    if zscores:
        status_gizi_label = tentukan_status_gizi(zscores['zs_bb_tb'])
        
        # Prediksi Cluster
        cluster_features = ['usia', 'berat', 'tinggi', 'zs_bb_u', 'zs_tb_u', 'zs_bb_tb']
        # Buat DataFrame kecil untuk clustering
        df_cluster = pd.DataFrame([{
            'usia': data_anak['usia'],
            'berat': data_anak['berat'],
            'tinggi': data_anak['tinggi'],
            **zscores
        }])
        data_scaled = scaler.transform(df_cluster[cluster_features])
        prediksi_cluster = kmeans_model.predict(data_scaled)[0]
    else:
        print("\n[INFO] Z-score tidak dapat dihitung (mungkin usia di luar rentang 0-60 bulan). Analisis Gizi & Cluster dilewati.")

    # # --- Bagian 3: TAMPILKAN SEMUA HASIL ---
    # print("\n" + "="*20 + " HASIL ANALISIS " + "="*20)
    # print(f"Hasil Prediksi      : Anak terdeteksi '{prediksi_label}'")
    # print(f"Tingkat Keyakinan   : {max(prediksi_proba)*100:.2f}%")
    # print("-" * 58)
    # print(f"Profil Cluster      : Masuk ke dalam Cluster {prediksi_cluster}")
    # print(f"Status Gizi (BB/TB) : {status_gizi_label}")
    # print("="*58)
    return {
        'prediksi_label': str(prediksi_label),
        'prediksi_proba': f"{float(max(prediksi_proba)*100):.2f}%",
        'status_gizi_label': str(status_gizi_label),
        'prediksi_cluster': int(prediksi_cluster) if isinstance(prediksi_cluster, (np.integer, np.int32, np.int64)) else str(prediksi_cluster),
    }
